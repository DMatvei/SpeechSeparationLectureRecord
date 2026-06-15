"""
Оркестратор пайплайна.

Дирижирует порядком загрузки/выгрузки компонентов (sequential loading —
тот же приём, что победил OOM на 8 ГБ VRAM):

  Этап 1: compressor  -> encode reference + все чанки mix -> выгрузить
  Этап 2: extractor   -> diffusion на каждом латенте      -> выгрузить
  Этап 3: compressor  -> decode каждого латента в wav      -> выгрузить
  Этап 4: corrector   -> коррекция каждого чанка           -> выгрузить
  Этап 5: склейка + (опц.) вторичный сигнал

Короткая запись = частный случай: один чанк на всю длину.
Промежуточные данные (латенты, wav-чанки) держим в RAM на CPU — для
дипломных длительностей это ок (десятки чанков).

Прогресс: on_progress(percent:int, message:str).
Отмена:   cancel_check() -> bool; если True, кидаем PipelineCancelled.
"""
import os
import torch
import numpy as np

from core import config, audio_io, solospeech
from core.vram import free_vram, get_device
from core.converter import convert_to_wav
from core.mask import intervals_from_silero_vad, build_residual_masked


class PipelineCancelled(Exception):
    pass


def _noop(*_args, **_kwargs):
    pass


def process(input_path: str,
            reference_path: str,
            output_dir: str,
            quality: str = config.DEFAULT_QUALITY,
            chunk_sec: float = config.DEFAULT_CHUNK_SEC,
            overlap_sec: float = config.DEFAULT_OVERLAP_SEC,
            make_residual: bool = True,
            on_progress=None,
            cancel_check=None):
    """Полный прогон. Возвращает словарь с путями к результатам.

    input_path     — исходный файл (любой формат, сконвертируем).
    reference_path — wav с референсом препода (уже выбран в GUI).
    output_dir     — куда писать результаты.
    quality        — ключ из QUALITY_PRESETS (low/medium/high).
    make_residual  — считать ли вторичный сигнал (mix - extracted).
    """
    on_progress = on_progress or _noop
    cancel_check = cancel_check or (lambda: False)

    def check_cancel():
        if cancel_check():
            raise PipelineCancelled()

    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    steps = config.QUALITY_PRESETS[quality]["num_infer_steps"]

    # ЭТАП 0: конвертация в 16 кГц моно wav --------------------
    on_progress(2, "Конвертация…")
    converted_wav = os.path.join(output_dir, "input_16k.wav")
    convert_to_wav(input_path, converted_wav, sr=config.SAMPLE_RATE)
    check_cancel()

    mixture = audio_io.load_audio(converted_wav)
    reference = audio_io.load_audio(reference_path)
    sr = config.SAMPLE_RATE
    total_dur = len(mixture) / sr

    # Чанкинг (короткая запись -> один чанк)
    if total_dur <= config.SHORT_AUDIO_THRESHOLD_SEC:
        mix_chunks_raw = [(mixture, 0, len(mixture))]
        eff_overlap = 0.0
    else:
        mix_chunks_raw = audio_io.chunk_audio(mixture, sr, chunk_sec, overlap_sec)
        eff_overlap = overlap_sec
    chunk_starts = [c[1] for c in mix_chunks_raw]
    n_chunks = len(mix_chunks_raw)
    on_progress(5, f"Чанков: {n_chunks}")

    tse_config = solospeech.load_extractor_config()

    # Этап 1: compressor -> encode reference + чанки mix ------
    on_progress(8, "Сжатие (компрессор)…")
    autoencoder = solospeech.load_compressor(device)
    free_vram("compressor loaded")

    ref_latent_cpu = solospeech.encode(autoencoder, reference, device)[0].cpu()

    mix_latents_cpu, stds_cpu = [], []
    for idx, (chunk, _, _) in enumerate(mix_chunks_raw):
        check_cancel()
        lat, std = solospeech.encode(autoencoder, chunk, device)
        mix_latents_cpu.append(lat.cpu())
        stds_cpu.append(std.cpu() if std is not None else None)
        del lat, std
        on_progress(8 + int(12 * (idx + 1) / n_chunks), "Сжатие…")

    del autoencoder
    free_vram("compressor freed")

    # Этап 2: extractor -> diffusion --------------------------
    on_progress(20, "Извлечение речи (диффузия)…")
    tse_model = solospeech.load_extractor(device, tse_config)
    scheduler = solospeech.make_scheduler(tse_config, device)
    free_vram("extractor loaded")

    pred_latents_cpu = []
    for idx, mix_lat_cpu in enumerate(mix_latents_cpu):
        check_cancel()
        pred = solospeech.run_diffusion(
            tse_model, scheduler, device,
            mix_lat_cpu, ref_latent_cpu,
            num_infer_steps=steps, seed=config.RANDOM_SEED + idx,
        )
        pred_latents_cpu.append(pred.cpu())
        del pred
        if idx % 5 == 0:
            torch.cuda.empty_cache()
        on_progress(20 + int(45 * (idx + 1) / n_chunks),
                    f"Извлечение… чанк {idx + 1}/{n_chunks}")

    del tse_model, scheduler, mix_latents_cpu
    free_vram("extractor freed")

    # Этап 3: compressor -> decode ----------------------------
    on_progress(65, "Декомпрессия…")
    autoencoder = solospeech.load_compressor(device)
    free_vram("compressor reloaded")

    pred_wavs_cpu = []
    for idx, pred_lat_cpu in enumerate(pred_latents_cpu):
        check_cancel()
        wav = solospeech.decode(autoencoder, pred_lat_cpu, stds_cpu[idx], device)
        pred_wavs_cpu.append(wav.cpu())
        del wav
        on_progress(65 + int(10 * (idx + 1) / n_chunks), "Декомпрессия…")

    del autoencoder, pred_latents_cpu
    free_vram("compressor freed (final)")

    # Этап 4: corrector ---------------------------------------
    on_progress(75, "Коррекция…")
    geco_model = solospeech.load_corrector(device)
    free_vram("corrector loaded")

    corrected_chunks = []
    for idx, (orig_chunk, _, _) in enumerate(mix_chunks_raw):
        check_cancel()
        x_hat = solospeech.correct(geco_model, pred_wavs_cpu[idx], orig_chunk, device)
        corrected_chunks.append(x_hat)
        if idx % 5 == 0:
            torch.cuda.empty_cache()
        on_progress(75 + int(15 * (idx + 1) / n_chunks),
                    f"Коррекция… чанк {idx + 1}/{n_chunks}")

    del geco_model, pred_wavs_cpu
    free_vram("corrector freed")

    # Этап 5: склейка + вторичный сигнал ----------------------
    on_progress(92, "Склейка…")
    chunks_with_starts = list(zip(corrected_chunks, chunk_starts))
    extracted = audio_io.crossfade_concat(chunks_with_starts, sr, eff_overlap)

    peak = np.abs(extracted).max()
    if peak > 1.0:
        extracted = extracted / peak * 0.99

    base = os.path.splitext(os.path.basename(input_path))[0]
    extracted_path = os.path.join(output_dir, f"{base}_speech.wav")
    audio_io.save_audio(extracted_path, extracted, sr)

    result = {"extracted": extracted_path}

    if make_residual:
        on_progress(95, "Вторичный сигнал…")
        intervals = intervals_from_silero_vad(extracted, sr)
        residual = build_residual_masked(mixture, intervals=intervals, sr=sr, fill="zero")
        residual_path = os.path.join(output_dir, f"{base}_residual.wav")
        audio_io.save_audio(residual_path, residual, sr)
        result["residual"] = residual_path
        result["residual_info"] = {"mode": "masked_zero", "n_intervals": len(intervals)}

    on_progress(100, "Готово")
    return result
