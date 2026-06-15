# Low-memory version of test_v2.py
# Loads components sequentially, frees VRAM between stages
import yaml
import gc
import argparse
import os
import torch
import torch.nn.functional as F
import librosa
from diffusers import DDIMScheduler


print("import solospeech_tse")
from solospeech.model.solospeech.conditioners import SoloSpeech_TSE
print('import save_audio')
from solospeech.scripts.solospeech.utils import save_audio
print('import autoencoder')
from solospeech.vae_modules.autoencoder_wrapper import Autoencoder
print('import scoreModel')
from solospeech.corrector.fastgeco.model import ScoreModel
print('import pad_spec')
from solospeech.corrector.geco.util.other import pad_spec

print('end import')

def free_vram(label=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved() / 1e9
        print(f'  [VRAM after {label}] allocated={alloc:.2f}GB reserved={reserv:.2f}GB', flush=True)


@torch.no_grad()
def run_diffusion(tse_model, scheduler, device, mixture, reference, lengths, reference_lengths,
                  ddim_steps, eta, seed):
    """Только diffusion-цикл, БЕЗ финального autoencoder decode (его сделаем отдельно)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(ddim_steps)
    tse_pred = torch.randn(mixture.shape, generator=generator, device=device)

    for i, t in enumerate(scheduler.timesteps):
        tse_pred = scheduler.scale_model_input(tse_pred, t)
        model_output, _ = tse_model(
            x=tse_pred, timesteps=t, mixture=mixture, reference=reference,
            x_len=lengths, ref_len=reference_lengths
        )
        tse_pred = scheduler.step(
            model_output=model_output, timestep=t, sample=tse_pred,
            eta=eta, generator=generator
        ).prev_sample
        if i % 10 == 0:
            print(f'    diffusion step {i}/{ddim_steps}', flush=True)

    return tse_pred  # латент, ещё не декодированный


def main(args):
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    print('start main')
    # Локальные пути к чекпоинтам
    local_dir = "./checkpoints"
    args.tse_config = os.path.join(local_dir, "config_extractor.yaml")
    args.vae_config = os.path.join(local_dir, "config_compressor.json")
    args.autoencoder_path = os.path.join(local_dir, "compressor.ckpt")
    args.tse_ckpt = os.path.join(local_dir, "extractor.pt")
    args.geco_ckpt = os.path.join(local_dir, "corrector.ckpt")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    with open(args.tse_config, 'r') as fp:
        tse_config = yaml.safe_load(fp)

    print("Loading audio...", flush=True)
    mixture, _ = librosa.load(args.test_wav, sr=16000)
    reference, _ = librosa.load(args.enroll_wav, sr=16000)
    mixture_wav_cpu = torch.tensor(mixture).unsqueeze(0)         # (1, T) на CPU
    reference_wav_cpu = torch.tensor(reference).unsqueeze(0)

    # ============================================================
    # ЭТАП 1: Encode mixture и reference в латенты через compressor
    # ============================================================
    print("\n[Stage 1] Loading compressor for encoding...", flush=True)
    autoencoder = Autoencoder(args.autoencoder_path, args.vae_config, 'stft_vae', quantization_first=True)
    autoencoder.eval().to(device)
    free_vram("compressor loaded")

    with torch.no_grad():
        ref_latent, _ = autoencoder(audio=reference_wav_cpu.to(device).unsqueeze(1))
        mix_latent, std = autoencoder(audio=mixture_wav_cpu.to(device).unsqueeze(1))

    # Сохраняем std для финального decode, и латенты на GPU (они маленькие)
    print(f'  ref_latent: {ref_latent.shape}, mix_latent: {mix_latent.shape}', flush=True)

    # Выгружаем compressor (понадобится снова для decode, но пока можно убрать)
    del autoencoder
    free_vram("compressor freed")

    # ============================================================
    # ЭТАП 2: Diffusion через extractor
    # ============================================================
    print("\n[Stage 2] Loading extractor for diffusion...", flush=True)
    tse_model = SoloSpeech_TSE(
        tse_config['diffwrap']['UDiT'],
        tse_config['diffwrap']['ViT'],
    ).to(device)
    tse_model.load_state_dict(torch.load(args.tse_ckpt, map_location=device)['model'])
    tse_model.eval()
    free_vram("extractor loaded")

    noise_scheduler = DDIMScheduler(**tse_config["ddim"]['diffusers'])
    # сброс dtype параметров шедулера
    _dummy = torch.randn((1, 128, 128), device=device)
    _ = noise_scheduler.add_noise(_dummy, torch.randn_like(_dummy),
                                  torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long())
    del _dummy

    print("  Running diffusion...", flush=True)
    lengths = torch.LongTensor([mix_latent.shape[-1]]).to(device)
    reference_lengths = torch.LongTensor([ref_latent.shape[-1]]).to(device)

    tse_pred_latent = run_diffusion(
        tse_model, noise_scheduler, device,
        mixture=mix_latent.transpose(2, 1),
        reference=ref_latent.transpose(2, 1),
        lengths=lengths, reference_lengths=reference_lengths,
        ddim_steps=args.num_infer_steps, eta=args.eta, seed=args.random_seed,
    )

    # Выгружаем extractor — он больше не нужен
    del tse_model, noise_scheduler
    del mix_latent, ref_latent
    free_vram("extractor freed")

    # ============================================================
    # ЭТАП 3: Decode латента обратно в wav через compressor
    # ============================================================
    print("\n[Stage 3] Reloading compressor for decoding...", flush=True)
    autoencoder = Autoencoder(args.autoencoder_path, args.vae_config, 'stft_vae', quantization_first=True)
    autoencoder.eval().to(device)
    free_vram("compressor reloaded")

    with torch.no_grad():
        pred = autoencoder(embedding=tse_pred_latent.transpose(2, 1), std=std).squeeze(1)

    del autoencoder, tse_pred_latent
    free_vram("compressor freed (final)")

    # ============================================================
    # ЭТАП 4: Correction через corrector
    # ============================================================
    print("\n[Stage 4] Loading corrector...", flush=True)
    geco_model = ScoreModel.load_from_checkpoint(
        args.geco_ckpt, batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    geco_model.eval(no_ema=False)
    geco_model.to(device)
    free_vram("corrector loaded")

    with torch.no_grad():
        min_leng = min(pred.shape[-1], mixture_wav_cpu.shape[-1])
        x = pred[..., :min_leng]
        m = mixture_wav_cpu.to(device)[..., :min_leng]
        norm_factor = m.abs().max()
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
        x_hat = x_hat.detach().cpu()

    del geco_model
    free_vram("corrector freed")

    save_audio(args.output_path, 16000, x_hat)
    print(f"\nSaved to: {args.output_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--test-wav', type=str, required=True)
    parser.add_argument('--enroll-wav', type=str, required=True)
    parser.add_argument('--eta', type=int, default=0)
    parser.add_argument("--num_infer_steps", type=int, default=50)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    main(args)