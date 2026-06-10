"""
Маска речи преподавателя m(t) и построение остаточного (студенческого) трека.

Основной (зафиксированный) подход:
  - intervals — речь препода, найденная Silero VAD по треку extracted
    (intervals_from_silero_vad);
  - intervals расширяются с pad_ms (~75мс), чтобы fade-ramp ложился на
    паузу вокруг речи, а не подрезал её;
  - построенная маска используется с fill="zero": в репликах препода (m=1)
    подставляется тишина, в паузах (m=0) — чистый mix:

        out = (1 - m) * mix + m * 0 = (1 - m) * mix

  - переходы 0<->1 сглажены raised-cosine кроссфейдом fade_ms (~50мс) —
    без щелчков на стыках.

Альтернатива (НЕ основной путь, оставлена как опция): fill="residual" —
смешивание с compute_residual (mix с ослабленным, но не занулённым преподом)
вместо тишины: out = (1 - m) * mix + m * residual.

intervals (откуда взялись — VAD/ручной ввод/что угодно) передаются снаружи,
смеситель про их источник ничего не знает.
"""
import numpy as np

IntervalList = list[tuple[float, float]]


def pad_intervals(intervals: IntervalList, pad_sec: float, duration_sec: float) -> IntervalList:
    """Расширяет каждый интервал на pad_sec в обе стороны, клиппит по [0, duration_sec]."""
    return [(max(0.0, s - pad_sec), min(duration_sec, e + pad_sec)) for s, e in intervals]


def normalize_intervals(intervals: IntervalList) -> IntervalList:
    """Сортирует по start и сливает пересекающиеся/смежные интервалы."""
    if not intervals:
        return []
    items = sorted((float(s), float(e)) for s, e in intervals if e > s)
    merged = [items[0]]
    for start, end in items[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _ramp(length: int, shape: str) -> np.ndarray:
    """Кривая 0->1 длиной length (используется и в обратном порядке для 1->0)."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if shape == "linear":
        return np.linspace(0.0, 1.0, length, dtype=np.float32)
    if shape == "hann":
        # raised-cosine: ровно 0 на левом крае, ровно 1 на правом
        return ((1 - np.cos(np.linspace(0.0, np.pi, length))) / 2).astype(np.float32)
    raise ValueError(f"unknown fade_shape: {shape!r}")


def build_mask(n: int, sr: int, intervals: IntervalList,
               fade_ms: float = 50.0, fade_shape: str = "hann",
               pad_ms: float = 75.0) -> np.ndarray:
    """Строит m(t) длиной n, значения в [0,1].

    Внутри интервала (за вычетом краёв) m=1, вне всех интервалов m=0.
    На первых/последних fade_ms каждого интервала — плавный переход
    0->1 / 1->0 (целиком внутри интервала, поэтому на стыке с m=0
    снаружи разрыва нет). Если интервал короче 2*fade, ramps уменьшаются
    до половины длины интервала.

    pad_ms: каждый интервал расширяется на pad_ms в обе стороны ДО
    normalize_intervals (чтобы соседние после расширения корректно
    слились) — fade-ramp тогда ложится на padding-зону, а не на саму
    речь препода (которую VAD мог слегка подрезать по краям).
    """
    mask = np.zeros(n, dtype=np.float32)
    if n <= 0:
        return mask

    fade_samples = int(round(fade_ms / 1000.0 * sr))
    duration_sec = n / sr

    padded = pad_intervals(intervals, pad_ms / 1000.0, duration_sec)
    for start_sec, end_sec in normalize_intervals(padded):
        start = int(round(max(0.0, start_sec) * sr))
        end = int(round(min(duration_sec, end_sec) * sr))
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if end <= start:
            continue

        seg = np.ones(end - start, dtype=np.float32)
        fade = min(fade_samples, (end - start) // 2)

        if fade > 0:
            seg[:fade] = _ramp(fade, fade_shape)
            seg[-fade:] = _ramp(fade, fade_shape)[::-1]

        mask[start:end] = np.maximum(mask[start:end], seg)

    return mask


def build_residual_masked(mix: np.ndarray, residual: np.ndarray = None,
                          intervals: IntervalList = (), sr: int = 16000,
                          fade_ms: float = 50.0, fade_shape: str = "hann",
                          fill: str = "zero", duck_gain: float = 0.1,
                          pad_ms: float = 75.0) -> np.ndarray:
    """(1 - m) * mix + m * filling.

    fill: что подставляется в зонах препода (m=1):
      - "zero" (по умолчанию, рабочая конфигурация) — тишина (полное
        зануление речи препода).
      - "residual" — НЕ основной путь, оставлен как опция: текущий
        compute_residual (mix с ослабленным преподом). Требует передать
        residual.
      - "duck"  — сильно приглушённый mix: duck_gain * mix.
    """
    if fill == "residual":
        if residual is None:
            raise ValueError("residual обязателен при fill='residual'")
        n = min(len(mix), len(residual))
    else:
        n = len(mix)
    mix = mix[:n].astype(np.float32)

    if fill == "residual":
        filling = residual[:n].astype(np.float32)
    elif fill == "zero":
        filling = np.zeros(n, dtype=np.float32)
    elif fill == "duck":
        filling = duck_gain * mix
    else:
        raise ValueError(f"unknown fill: {fill!r}")

    m = build_mask(n, sr, intervals, fade_ms, fade_shape, pad_ms)
    return (1 - m) * mix + m * filling


# ===========================================================================
# Источники интервалов (опциональные, инъектируются снаружи)
# ===========================================================================
def _load_silero_vad():
    """Лениво грузит Silero VAD (torch.hub).

    Вызывающий код сам решает, переиспользовать ли (model, get_speech_timestamps)
    между несколькими вызовами — здесь модель не кэшируется.
    """
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def intervals_from_silero_vad(speech_track: np.ndarray, sr: int = 16000,
                               threshold: float = 0.5,
                               min_speech_sec: float = 0.25,
                               min_silence_sec: float = 0.25) -> IntervalList:
    """Опционально: Silero VAD по дорожке речи препода (extracted/pred_wav).

    torch.hub импортируется лениво — модуль mask.py грузится без Silero,
    если эта функция не вызывается.
    """
    import torch

    model, get_speech_timestamps = _load_silero_vad()

    wav = torch.tensor(speech_track, dtype=torch.float32)
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=sr, threshold=threshold,
        min_speech_duration_ms=int(min_speech_sec * 1000),
        min_silence_duration_ms=int(min_silence_sec * 1000),
    )

    intervals = [(ts["start"] / sr, ts["end"] / sr) for ts in timestamps]
    return normalize_intervals(intervals)


def speech_fraction(audio_segment: np.ndarray, sr: int = 16000,
                     threshold: float = 0.5,
                     model=None, get_speech_timestamps=None) -> float:
    """Доля audio_segment, помеченная Silero VAD как речь (0..1).

    model/get_speech_timestamps можно передать готовыми (см. make_vad_validator),
    чтобы не грузить модель на каждый вызов. Если не переданы — грузятся лениво.
    """
    import torch

    if len(audio_segment) == 0:
        return 0.0
    if model is None or get_speech_timestamps is None:
        model, get_speech_timestamps = _load_silero_vad()

    wav = torch.tensor(audio_segment, dtype=torch.float32)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=threshold)

    speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
    return speech_samples / len(audio_segment)


def make_vad_validator(audio: np.ndarray, sr: int = 16000,
                        threshold: float = 0.5,
                        min_speech_frac: float = 0.8):
    """Строит validator(start_sec, end_sec) -> bool для ref_picker.pick_reference.

    Silero грузится один раз при создании валидатора и переиспользуется на
    всех попытках pick_reference (max_tries), а не на каждый кандидат заново.
    """
    model, get_speech_timestamps = _load_silero_vad()

    def validator(start_sec: float, end_sec: float) -> bool:
        start = int(start_sec * sr)
        end = int(end_sec * sr)
        frac = speech_fraction(audio[start:end], sr, threshold, model, get_speech_timestamps)
        return frac >= min_speech_frac

    return validator
