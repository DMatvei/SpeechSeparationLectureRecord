import librosa
import numpy as np
import torch

from  solospeech.scripts.solospeech.utils import save_audio as _solo_save_audio
from . import config



def load_audio(path: str, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr)
    return wav


def save_audio(path: str, wav, sr: int = config.SAMPLE_RATE) -> None:

    if isinstance(wav, np.ndarray):
        wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    _solo_save_audio(path, sr, wav)


def chunk_audio(
        wav: np.ndarray,
        sr: int,
        chunk_sec: float,
        overlap_sec: float = 0.0):

    n = len(wav)
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step = chunk_samples - overlap_samples
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_samples, n)
        chunks.append((wav[start:end], start, end))
        if end >= n:
            break
        start += step
    return chunks

def crossfade_concat(
        chunks_with_starts,
        sr: int,
        overlap_sec: float) -> np.ndarray:
    if overlap_sec <= 0 or len(chunks_with_starts) == 1:
        return np.concatenate([c[0] for c in chunks_with_starts])

    overlap_samples = int(overlap_sec * sr)
    out = chunks_with_starts[0][0].copy().astype(np.float32)
    for i in range(1, len(chunks_with_starts)):
        prev_wav = out
        cur_wav, _ = chunks_with_starts[i]
        fade_len = min(overlap_samples, len(prev_wav), len(cur_wav))
        if fade_len <= 0:
            out = np.concatenate([prev_wav, cur_wav])
            continue
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = 1.0 - fade_in
        head_cur = cur_wav[:fade_len] * fade_in
        tail_prev = prev_wav[-fade_len:] * fade_out
        merged_overlap = head_cur + tail_prev
        out = np.concatenate([prev_wav[:-fade_len], merged_overlap,
                              cur_wav[fade_len:]])
    return out

















