"""
Сэмплер позиции референсного фрагмента: равномерный выбор в центральной
зоне записи.

Чистая математика (random/numpy) — без зависимостей от VAD/torch/librosa.
См. CLAUDE.md, разделы «Выбор референсного фрагмента» и «Профиль памяти encode».

frag_len_sec по умолчанию 5.0: на 8 ГБ карте encode() компрессора растёт
линейно (~313 МБ/с + ~240 МБ), 5с -> ~1.8 ГБ — безопасно с большим запасом
(порог ~24с).
"""
import numpy as np


def propose(duration_sec: float, frag_len_sec: float = 5.0,
            central_frac: float = 0.6,
            rng: np.random.Generator = None) -> tuple[float, float]:
    """Предлагает (start_sec, end_sec) длиной frag_len_sec.

    Старт выбирается РАВНОМЕРНО так, чтобы весь фрагмент целиком лежал в
    центральной зоне [zone_lo, zone_hi] = [(1-central_frac)/2 * duration,
    duration - (1-central_frac)/2 * duration]:

        start ~ Uniform(zone_lo, zone_hi - frag_len_sec)

    Граничные случаи:
    - запись короче frag_len_sec -> вернуть [0, duration_sec];
    - frag_len_sec не влезает в центральную зону -> падаем на всю запись
      [0, duration_sec - frag_len_sec];
    - результат всегда клиппится по [0, duration_sec].
    """
    if rng is None:
        rng = np.random.default_rng()

    if duration_sec <= 0:
        return 0.0, 0.0

    if frag_len_sec >= duration_sec:
        return 0.0, duration_sec

    margin = (1.0 - central_frac) / 2.0 * duration_sec
    zone_lo, zone_hi = margin, duration_sec - margin

    available = (zone_hi - zone_lo) - frag_len_sec
    if available < 0:
        # центральная зона меньше фрагмента -> вся запись
        zone_lo, zone_hi = 0.0, duration_sec
        available = duration_sec - frag_len_sec

    start = float(rng.uniform(zone_lo, zone_lo + available))
    start = float(np.clip(start, 0.0, duration_sec - frag_len_sec))

    return start, start + frag_len_sec


def pick_reference(duration_sec: float, frag_len_sec: float = 5.0,
                   validator=None, max_tries: int = 20,
                   central_frac: float = 0.6,
                   rng: np.random.Generator = None) -> tuple[float, float]:
    """Сэмплер + опциональный VAD-валидатор через инъекцию зависимости.

    validator(start_sec, end_sec) -> bool. validator=None — легитимный режим
    (чистый рандом без VAD), а не ошибка. После max_tries неудачных попыток
    возвращается последний кандидат (fallback).
    """
    if rng is None:
        rng = np.random.default_rng()

    candidate = None
    for _ in range(max_tries):
        candidate = propose(duration_sec, frag_len_sec, central_frac, rng)
        if validator is None or validator(*candidate):
            return candidate
    return candidate


if __name__ == "__main__":
    duration = 30 * 60.0  # 30 минут
    frag_len = 5.0
    central_frac = 0.6
    rng = np.random.default_rng(42)

    print(f"duration={duration:.1f}s, frag_len={frag_len}s, central_frac={central_frac}\n")
    starts = []
    for i in range(15):
        start, end = propose(duration, frag_len_sec=frag_len, central_frac=central_frac, rng=rng)
        starts.append(start)
        ok_range = 0.0 <= start and end <= duration
        ok_len = abs((end - start) - frag_len) < 1e-9
        print(f"{i + 1:2d}: start={start:8.2f}  end={end:8.2f}  "
              f"len={end - start:.2f}  in_range={ok_range}  len_ok={ok_len}")

    margin = (1.0 - central_frac) / 2.0 * duration
    print(f"\nzone = [{margin:.1f}, {duration - margin:.1f}]")
    print(f"start range = [{margin:.1f}, {duration - margin - frag_len:.1f}]")
    print(f"observed min/max start = {min(starts):.2f} / {max(starts):.2f}")

    print("\n--- короткие записи ---")
    for d in (8.0, 3.0):
        s, e = propose(d, frag_len_sec=frag_len, central_frac=central_frac,
                       rng=np.random.default_rng(0))
        print(f"duration={d}s -> start={s:.2f}, end={e:.2f}, len={e - s:.2f}")
