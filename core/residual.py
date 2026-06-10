"""
Этап 5: вторичный (остаточный) сигнал = всё, кроме препода.

Наивное mix - extracted не работает: SoloSpeech генеративный, его выход
не выровнен по фазе/амплитуде с оригиналом (в отличие от масочных моделей).
Поэтому перед вычитанием:
  1. выравниваем по времени (кросс-корреляция) — убираем возможный сдвиг;
  2. подбираем масштаб alpha методом наименьших квадратов:
     alpha = <mix, extracted> / <extracted, extracted>,
     минимизирует ||mix - alpha * extracted||^2.
Затем residual = mix - alpha * extracted_aligned.
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


def _best_scale(mix: np.ndarray, extracted: np.ndarray) -> float:
    """alpha по МНК: проекция mix на extracted."""
    denom = float(np.dot(extracted, extracted))
    if denom < 1e-12:
        return 1.0
    return float(np.dot(mix, extracted) / denom)


def compute_residual(mix: np.ndarray, extracted: np.ndarray,
                     align: bool = True, scale: bool = True):
    """Возвращает (residual, info).

    mix, extracted — 1D numpy на одной частоте.
    info — словарь с диагностикой (lag, alpha) для логов/GUI.
    """
    n = min(len(mix), len(extracted))
    mix = mix[:n].astype(np.float32)
    ext = extracted[:n].astype(np.float32)

    lag = 0
    if align:
        ext, lag = _align_time(mix, ext)

    alpha = 1.0
    if scale:
        alpha = _best_scale(mix, ext)

    residual = mix - alpha * ext

    # Защита от клиппинга
    peak = np.abs(residual).max()
    if peak > 1.0:
        residual = residual / peak * 0.99

    info = {"lag_samples": lag, "alpha": round(alpha, 4)}
    return residual.astype(np.float32), info
