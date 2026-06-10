"""
Этап 5: вторичный (остаточный) сигнал = всё, кроме препода.

Наивное mix - extracted не работает: SoloSpeech генеративный, его выход
не выровнен по фазе/амплитуде с оригиналом (в отличие от масочных моделей).

Два режима:
  - "global" (старый): один лаг по первым 10с + один alpha МНК на весь файл.
  - "local" (по умолчанию): рассинхрон SoloSpeech локальный (независимый
    diffusion по 15с-чанкам), поэтому лаг и alpha оцениваются в скользящем
    окне (window_sec, перекрытие overlap):
      1. для каждого окна ищется свой лаг (кросс-корреляция, ±max_shift) —
         сдвинутые версии extracted накладываются друг на друга через
         Hann-overlap-add (плавный переход между разными лагами без щелчков);
      2. для каждого окна считается свой alpha (МНК), клиппится вокруг
         глобального alpha (±alpha_tol) и линейно интерполируется по
         сэмплам — получаем непрерывную кривую alpha(t).
Затем residual = mix - alpha(t) * extracted_aligned(t).
"""
import numpy as np
from scipy.signal import correlate


def _align_time(mix: np.ndarray, extracted: np.ndarray, max_shift: int = 800):
    """Находит сдвиг extracted относительно mix по кросс-корреляции и
    возвращает extracted, сдвинутый и подрезанный под длину mix.
    max_shift=800 сэмплов = 50 мс на 16 кГц (с запасом).
    """
    n = min(len(mix), len(extracted))
    a = mix[:n].astype(np.float64)
    b = extracted[:n].astype(np.float64)

    # Кросс-корреляция в окне ±max_shift (полную считать дорого для часовых файлов).
    # Берём центральный сегмент для оценки сдвига — этого достаточно.
    seg = min(n, 16000 * 10)  # до 10 c для оценки
    corr = correlate(a[:seg], b[:seg], mode="full")
    lag = np.argmax(corr) - (seg - 1)
    lag = int(np.clip(lag, -max_shift, max_shift))

    out = np.zeros(n, dtype=np.float32)
    if lag >= 0:
        out[lag:] = extracted[:n - lag]
    else:
        out[:n + lag] = extracted[-lag:n]
    return out, lag


def _best_scale(mix: np.ndarray, extracted: np.ndarray, fallback: float = 1.0) -> float:
    """alpha по МНК: проекция mix на extracted. Если extracted ~ 0 (denom
    слишком мал), возвращает fallback вместо случайного значения."""
    denom = float(np.dot(extracted, extracted))
    if denom < 1e-12:
        return fallback
    return float(np.dot(mix, extracted) / denom)


def _local_lag(mix_seg: np.ndarray, ext: np.ndarray, start: int, end: int,
               max_shift: int) -> int:
    """Локальный лаг extracted относительно mix для окна [start:end).

    Сдвиг определяется так, что выровненный сегмент extracted равен
    ext[start+lag : end+lag] (см. применение в _compute_residual_local).
    """
    if max_shift <= 0 or len(mix_seg) == 0:
        return 0
    lo = max(0, start - max_shift)
    hi = min(len(ext), end + max_shift)
    ext_region = ext[lo:hi]
    if len(ext_region) < len(mix_seg):
        return 0

    corr = correlate(mix_seg.astype(np.float64), ext_region.astype(np.float64),
                      mode="valid")
    if len(corr) == 0:
        return 0
    k = int(np.argmax(corr))
    # scipy correlate(a, b, "valid") с len(a) <= len(b) даёт
    # corr[k] = sum_i a[i] * b[i + (len(b) - len(a) - k)]
    # т.е. corr[k] = sum_i mix_seg[i] * ext_region[i + (L_diff - k)]
    #              = sum_i mix_seg[i] * ext[lo + i + (L_diff - k)]
    # хотим ext[start+i+lag] -> lag = lo + (L_diff - k) - start
    L_diff = len(ext_region) - len(mix_seg)
    lag = lo + (L_diff - k) - start
    return int(np.clip(lag, -max_shift, max_shift))


def _compute_residual_global(mix: np.ndarray, ext: np.ndarray,
                              align: bool, scale: bool):
    lag = 0
    if align:
        ext, lag = _align_time(mix, ext)

    alpha = 1.0
    if scale:
        alpha = _best_scale(mix, ext)

    residual = mix - alpha * ext

    peak = np.abs(residual).max()
    if peak > 1.0:
        residual = residual / peak * 0.99

    info = {"mode": "global", "lag_samples": lag, "alpha": round(alpha, 4)}
    return residual.astype(np.float32), info


def _compute_residual_local(mix: np.ndarray, ext: np.ndarray,
                             align: bool, scale: bool, sr: int,
                             window_sec: float, overlap: float,
                             max_shift_sec: float, alpha_tol: float):
    n = len(mix)
    window = max(1, int(window_sec * sr))
    hop = max(1, int(window * (1 - overlap)))
    max_shift = int(max_shift_sec * sr)

    global_alpha = _best_scale(mix, ext) if scale else 1.0

    ext_aligned = np.zeros(n, dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)
    centers, alphas, lags = [], [], []

    # Hann-окно для overlap-add. На границах full Hann уходит в 0, что при
    # одном-единственном окне (короткая запись) обнулило бы всё — на этот
    # случай используем плоский вес.
    if window > 1:
        fade = np.hanning(window)
    else:
        fade = np.ones(window)

    start = 0
    while True:
        end = min(start + window, n)
        mix_seg = mix[start:end]
        seg_len = end - start

        lag = _local_lag(mix_seg, ext, start, end, max_shift) if align else 0

        seg = np.zeros(seg_len, dtype=np.float32)
        src_lo, src_hi = start + lag, end + lag
        clip_lo, clip_hi = max(0, src_lo), min(len(ext), src_hi)
        if clip_hi > clip_lo:
            seg[clip_lo - src_lo: clip_hi - src_lo] = ext[clip_lo:clip_hi]

        w = fade[:seg_len]
        if seg_len < window:
            # последнее (укороченное) окно: полный Hann не подходит по
            # длине — берём плоский вес, чтобы не занижать края.
            w = np.ones(seg_len)

        ext_aligned[start:end] += seg * w
        weight_sum[start:end] += w

        local_alpha = _best_scale(mix_seg, seg, fallback=global_alpha) if scale else 1.0
        delta = alpha_tol * max(abs(global_alpha), 1e-6)
        local_alpha = max(0.0, float(np.clip(local_alpha,
                                              global_alpha - delta,
                                              global_alpha + delta)))

        centers.append(start + seg_len / 2)
        alphas.append(local_alpha)
        lags.append(lag)

        if end == n:
            break
        start += hop

    # Защита от деления на ~0 на краях (если веса там всё же малы/нулевые).
    weight_sum[weight_sum < 1e-8] = 1.0
    ext_aligned = ext_aligned / weight_sum

    if scale:
        alpha_curve = np.interp(np.arange(n), centers, alphas)
    else:
        alpha_curve = np.ones(n, dtype=np.float64)

    residual = mix - alpha_curve * ext_aligned.astype(np.float32)

    peak = np.abs(residual).max()
    if peak > 1.0:
        residual = residual / peak * 0.99

    info = {
        "mode": "local",
        "global_alpha": round(global_alpha, 4),
        "n_windows": len(centers),
        "window_sec": window_sec,
        "overlap": overlap,
        "lag_samples_min": int(min(lags)),
        "lag_samples_max": int(max(lags)),
        "alpha_min": round(min(alphas), 4),
        "alpha_max": round(max(alphas), 4),
    }
    return residual.astype(np.float32), info


def compute_residual(mix: np.ndarray, extracted: np.ndarray,
                     align: bool = True, scale: bool = True,
                     residual_mode: str = "local",
                     sr: int = 16000,
                     window_sec: float = 2.0,
                     overlap: float = 0.5,
                     max_shift_sec: float = 0.05,
                     alpha_tol: float = 0.5):
    """Возвращает (residual, info).

    mix, extracted — 1D numpy на одной частоте (sr).
    residual_mode  — "local" (по умолчанию, скользящее окно) или "global"
                      (старое поведение: один лаг + один alpha на весь файл).
    info — словарь с диагностикой для логов/GUI.
    """
    n = min(len(mix), len(extracted))
    mix = mix[:n].astype(np.float32)
    ext = extracted[:n].astype(np.float32)

    if residual_mode == "global":
        return _compute_residual_global(mix, ext, align, scale)

    return _compute_residual_local(mix, ext, align, scale, sr,
                                    window_sec, overlap, max_shift_sec, alpha_tol)
