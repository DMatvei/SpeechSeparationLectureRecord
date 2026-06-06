# Low-memory SoloSpeech inference for LONG audio (e.g. 30 min lectures).
# Simple time-based chunking (no diarization, no JSON), batched stage processing.
#
# Pipeline:
#   1. Split mix into N chunks of fixed duration (with optional overlap)
#   2. Load compressor → encode all chunks + reference → save latents to CPU → free
#   3. Load extractor → diffusion on each latent → save predicted latents to CPU → free
#   4. Load compressor again → decode each predicted latent → save wavs to CPU → free
#   5. Load corrector → correct each wav → save final wavs to CPU → free
#   6. Concatenate (with optional crossfade) and save final wav
#
# Usage:
#   python -u scripts/test_long_lowmem.py \
#       --test-wav "../../output/input_30min.wav" \
#       --enroll-wav "../../output/refs/ref_000.wav" \
#       --output-path "../../output/tse_out/solospeech_30min.wav" \
#       --chunk-sec 15 --num-infer-steps 50

import yaml
import gc
import argparse
import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler
from solospeech.model.solospeech.conditioners import SoloSpeech_TSE
from solospeech.scripts.solospeech.utils import save_audio
from solospeech.vae_modules.autoencoder_wrapper import Autoencoder
from solospeech.corrector.fastgeco.model import ScoreModel
from solospeech.corrector.geco.util.other import pad_spec


def free_vram(label=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved() / 1e9
        print(f'  [VRAM after {label}] allocated={alloc:.2f}GB reserved={reserv:.2f}GB', flush=True)


def chunk_audio(wav, sr, chunk_sec, overlap_sec=0.0):
    """Split 1D numpy wav into chunks. Returns list of (chunk, start_idx, end_idx)."""
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


def crossfade_concat(chunks_with_starts, sr, overlap_sec):
    """Склейка чанков с crossfade. chunks_with_starts: list of (wav, start_idx)."""
    if overlap_sec <= 0 or len(chunks_with_starts) == 1:
        # просто конкатенация
        return np.concatenate([c[0] for c in chunks_with_starts])

    overlap_samples = int(overlap_sec * sr)
    out = chunks_with_starts[0][0].copy().astype(np.float32)
    for i in range(1, len(chunks_with_starts)):
        prev_wav = out
        cur_wav, cur_start = chunks_with_starts[i]
        # Crossfade длиной overlap_samples
        fade_len = min(overlap_samples, len(prev_wav), len(cur_wav))
        if fade_len <= 0:
            out = np.concatenate([prev_wav, cur_wav])
            continue
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = 1.0 - fade_in
        # Хвост prev и голова cur пересекаются
        head_cur = cur_wav[:fade_len] * fade_in
        tail_prev = prev_wav[-fade_len:] * fade_out
        merged_overlap = head_cur + tail_prev
        out = np.concatenate([prev_wav[:-fade_len], merged_overlap, cur_wav[fade_len:]])
    return out


@torch.no_grad()
def run_diffusion_one(tse_model, scheduler, device, mix_latent, ref_latent,
                     ddim_steps, eta, seed):
    """Один проход diffusion для одного латента mix + reference."""
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(ddim_steps)

    mixture = mix_latent.transpose(2, 1).to(device)
    reference = ref_latent.transpose(2, 1).to(device)
    lengths = torch.LongTensor([mixture.shape[1]]).to(device)
    ref_lengths = torch.LongTensor([reference.shape[1]]).to(device)

    pred = torch.randn(mixture.shape, generator=generator, device=device)
    for t in scheduler.timesteps:
        pred = scheduler.scale_model_input(pred, t)
        model_output, _ = tse_model(
            x=pred, timesteps=t, mixture=mixture, reference=reference,
            x_len=lengths, ref_len=ref_lengths
        )
        pred = scheduler.step(
            model_output=model_output, timestep=t, sample=pred,
            eta=eta, generator=generator
        ).prev_sample

    return pred  # на GPU, ещё не декодирован


def main(args):
    out_dir = os.path.dirname(args.output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    local_dir = args.local_dir
    tse_config_path = os.path.join(local_dir, "config_extractor.yaml")
    vae_config_path = os.path.join(local_dir, "config_compressor.json")
    autoencoder_path = os.path.join(local_dir, "compressor.ckpt")
    tse_ckpt = os.path.join(local_dir, "extractor.pt")
    geco_ckpt = os.path.join(local_dir, "corrector.ckpt")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    with open(tse_config_path, 'r') as fp:
        tse_config = yaml.safe_load(fp)

    # ============================================================
    # Загрузка аудио
    # ============================================================
    print("\n[Load] Reading audio files...", flush=True)
    mixture, sr = librosa.load(args.test_wav, sr=args.sample_rate)
    reference, _ = librosa.load(args.enroll_wav, sr=args.sample_rate)
    total_dur = len(mixture) / sr
    ref_dur = len(reference) / sr
    print(f'  mixture: {total_dur:.1f}s, reference: {ref_dur:.1f}s', flush=True)

    # Чанкинг mix
    mix_chunks_raw = chunk_audio(mixture, sr, args.chunk_sec, args.overlap_sec)
    print(f'  Total chunks: {len(mix_chunks_raw)} (chunk={args.chunk_sec}s, overlap={args.overlap_sec}s)', flush=True)
    chunk_starts = [c[1] for c in mix_chunks_raw]

    # ============================================================
    # ЭТАП 1: Compressor → encode всех чанков + reference
    # ============================================================
    print("\n[Stage 1] Loading compressor for encoding...", flush=True)
    autoencoder = Autoencoder(autoencoder_path, vae_config_path, 'stft_vae', quantization_first=True)
    autoencoder.eval().to(device)
    free_vram("compressor loaded")

    print("  Encoding reference...", flush=True)
    with torch.no_grad():
        ref_tensor = torch.tensor(reference, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        ref_latent, _ = autoencoder(audio=ref_tensor)
        ref_latent_cpu = ref_latent.cpu()  # переносим на CPU
        del ref_tensor, ref_latent

    print("  Encoding mixture chunks...", flush=True)
    mix_latents_cpu = []
    stds_cpu = []
    for chunk, _, _ in tqdm(mix_chunks_raw, desc="encode"):
        with torch.no_grad():
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            lat, std = autoencoder(audio=chunk_tensor)
            mix_latents_cpu.append(lat.cpu())
            stds_cpu.append(std.cpu() if std is not None else None)
            del chunk_tensor, lat, std

    del autoencoder
    free_vram("compressor freed")

    # ============================================================
    # ЭТАП 2: Extractor → diffusion для каждого латента
    # ============================================================
    print("\n[Stage 2] Loading extractor...", flush=True)
    tse_model = SoloSpeech_TSE(
        tse_config['diffwrap']['UDiT'],
        tse_config['diffwrap']['ViT'],
    ).to(device)
    tse_model.load_state_dict(torch.load(tse_ckpt, map_location=device)['model'])
    tse_model.eval()
    free_vram("extractor loaded")

    noise_scheduler = DDIMScheduler(**tse_config["ddim"]['diffusers'])
    _dummy = torch.randn((1, 128, 128), device=device)
    _ = noise_scheduler.add_noise(_dummy, torch.randn_like(_dummy),
                                  torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long())
    del _dummy

    pred_latents_cpu = []
    print(f"  Running diffusion on {len(mix_latents_cpu)} chunks ({args.num_infer_steps} steps each)...", flush=True)
    for i, mix_lat_cpu in enumerate(tqdm(mix_latents_cpu, desc="diffuse")):
        mix_lat = mix_lat_cpu.to(device)
        pred = run_diffusion_one(
            tse_model, noise_scheduler, device,
            mix_lat, ref_latent_cpu.to(device),
            ddim_steps=args.num_infer_steps, eta=args.eta,
            seed=args.random_seed + i,  # разный seed на каждый чанк
        )
        pred_latents_cpu.append(pred.cpu())
        del mix_lat, pred
        if i % 5 == 0:
            torch.cuda.empty_cache()

    del tse_model, noise_scheduler, mix_latents_cpu
    free_vram("extractor freed")

    # ============================================================
    # ЭТАП 3: Compressor → decode латентов в wav
    # ============================================================
    print("\n[Stage 3] Reloading compressor for decoding...", flush=True)
    autoencoder = Autoencoder(autoencoder_path, vae_config_path, 'stft_vae', quantization_first=True)
    autoencoder.eval().to(device)
    free_vram("compressor reloaded")

    pred_wavs_cpu = []
    for i, pred_lat_cpu in enumerate(tqdm(pred_latents_cpu, desc="decode")):
        with torch.no_grad():
            pred_lat = pred_lat_cpu.to(device)
            std = stds_cpu[i].to(device) if stds_cpu[i] is not None else None
            wav = autoencoder(embedding=pred_lat.transpose(2, 1), std=std).squeeze(1)
            pred_wavs_cpu.append(wav.cpu())
            del pred_lat, wav

    del autoencoder, pred_latents_cpu
    free_vram("compressor freed (final)")

    # ============================================================
    # ЭТАП 4: Corrector → correction каждого wav-чанка
    # ============================================================
    if not args.skip_corrector:
        print("\n[Stage 4] Loading corrector...", flush=True)
        geco_model = ScoreModel.load_from_checkpoint(
            geco_ckpt, batch_size=1, num_workers=0, kwargs=dict(gpu=False)
        )
        geco_model.eval(no_ema=False)
        geco_model.to(device)
        free_vram("corrector loaded")

        corrected_chunks_cpu = []
        for i, (orig_chunk, _, _) in enumerate(tqdm(mix_chunks_raw, desc="correct")):
            with torch.no_grad():
                pred = pred_wavs_cpu[i].to(device)
                m = torch.tensor(orig_chunk, dtype=torch.float32, device=device).unsqueeze(0)
                min_leng = min(pred.shape[-1], m.shape[-1])
                x = pred[..., :min_leng]
                m = m[..., :min_leng]
                norm_factor = m.abs().max()
                if norm_factor < 1e-9:
                    norm_factor = torch.tensor(1.0, device=device)
                x = x / norm_factor
                m = m / norm_factor

                X = torch.unsqueeze(geco_model._forward_transform(geco_model._stft(x)), 0)
                X = pad_spec(X)
                M = torch.unsqueeze(geco_model._forward_transform(geco_model._stft(m)), 0)
                M = pad_spec(M)

                timesteps = torch.linspace(0.5, 0.03, 1, device=M.device)
                std_corr = geco_model.sde._std(0.5 * torch.ones((M.shape[0],), device=M.device))
                z = torch.randn_like(M)
                X_t = M + z * std_corr[:, None, None, None]

                for idx in range(len(timesteps)):
                    t = timesteps[idx]
                    dt = t - timesteps[idx + 1] if idx != len(timesteps) - 1 else timesteps[-1]
                    f, g = geco_model.sde.sde(X_t, t, M)
                    vec_t = torch.ones(M.shape[0], device=M.device) * t
                    mean_x = X_t - (f - g**2 * geco_model.forward(X_t, vec_t, M, X, vec_t[:, None, None, None])) * dt
                    if idx == len(timesteps) - 1:
                        X_t = mean_x
                        break
                    z = torch.randn_like(X)
                    X_t = mean_x + z * g * torch.sqrt(dt)

                sample = X_t.squeeze()
                x_hat = geco_model.to_audio(sample.squeeze(), min_leng)
                x_hat = x_hat * norm_factor / x_hat.abs().max()
                corrected_chunks_cpu.append(x_hat.detach().cpu().numpy())
                del pred, m, X, M, X_t, x_hat
            if i % 5 == 0:
                torch.cuda.empty_cache()

        del geco_model
        free_vram("corrector freed")
        final_chunks = corrected_chunks_cpu
    else:
        print("\n[Stage 4] Skipped (--skip-corrector)", flush=True)
        final_chunks = [w.squeeze().numpy() for w in pred_wavs_cpu]

    # ============================================================
    # ЭТАП 5: Склейка чанков
    # ============================================================
    print("\n[Stage 5] Concatenating chunks...", flush=True)
    chunks_with_starts = list(zip(final_chunks, chunk_starts))
    final_wav = crossfade_concat(chunks_with_starts, sr, args.overlap_sec)

    # Нормализация на всякий случай
    peak = np.abs(final_wav).max()
    if peak > 1.0:
        final_wav = final_wav / peak * 0.99

    save_audio(args.output_path, sr, torch.tensor(final_wav, dtype=torch.float32).unsqueeze(0))
    print(f"\n[Done] Saved to: {args.output_path}", flush=True)
    print(f"  Total length: {len(final_wav)/sr:.1f}s", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--test-wav', type=str, required=True)
    parser.add_argument('--enroll-wav', type=str, required=True)
    parser.add_argument('--local-dir', type=str, default='./checkpoints',
                        help='Path to checkpoints folder')
    parser.add_argument('--chunk-sec', type=float, default=15.0,
                        help='Chunk length in seconds')
    parser.add_argument('--overlap-sec', type=float, default=1.0,
                        help='Overlap between chunks (for crossfade)')
    parser.add_argument('--num-infer-steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--skip-corrector', action='store_true',
                        help='Skip the final corrector stage (faster, чуть хуже качество)')
    args = parser.parse_args()
    main(args)