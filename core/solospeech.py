import yaml
import torch
from diffusers import DDIMScheduler

from solospeech.model.solospeech.conditioners import SoloSpeech_TSE
from solospeech.vae_modules.autoencoder_wrapper import Autoencoder
from solospeech.corrector.fastgeco.model import ScoreModel
from solospeech.corrector.geco.util.other import pad_spec

from . import config


def load_extractor_config() -> dict:
    with open(config.EXTRACTOR_CONFIG, "r") as fp:
        return yaml.safe_load(fp)


# ===========================================================================
# ЗАГРУЗЧИКИ КОМПОНЕНТОВ — каждый грузит один компонент на device
# ===========================================================================
def load_compressor(device: str) -> Autoencoder:
    """Compressor (STFT-VAE). Используется и для encode, и для decode."""
    autoencoder = Autoencoder(
        config.COMPRESSOR_CKPT, config.COMPRESSOR_CONFIG,
        "stft_vae", quantization_first=True,
    )
    autoencoder.eval().to(device)
    return autoencoder


def load_extractor(device: str, tse_config: dict) -> SoloSpeech_TSE:
    """Extractor (U-DiT диффузионная модель) — главный компонент."""
    tse_model = SoloSpeech_TSE(
        tse_config["diffwrap"]["UDiT"],
        tse_config["diffwrap"]["ViT"],
    ).to(device)
    tse_model.load_state_dict(
        torch.load(config.EXTRACTOR_CKPT, map_location=device)["model"]
    )
    tse_model.eval()
    return tse_model


def load_corrector(device: str) -> ScoreModel:
    """Corrector (Fast-GeCo) — финальная шлифовка в T-F домене."""
    geco_model = ScoreModel.load_from_checkpoint(
        config.CORRECTOR_CKPT, batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    geco_model.eval(no_ema=False)
    geco_model.to(device)
    return geco_model


def make_scheduler(tse_config: dict, device: str) -> DDIMScheduler:
    """DDIM-шедулер + прогрев dtype (как в обоих скриптах)."""
    noise_scheduler = DDIMScheduler(**tse_config["ddim"]["diffusers"])
    _dummy = torch.randn((1, 128, 128), device=device)
    _ = noise_scheduler.add_noise(
        _dummy, torch.randn_like(_dummy),
        torch.randint(0, noise_scheduler.config.num_train_timesteps,
                      (1,), device=device).long()
    )
    del _dummy
    return noise_scheduler


# ===========================================================================
# ОПЕРАЦИИ — чистые функции над тензорами
# ===========================================================================
@torch.no_grad()
def encode(autoencoder: Autoencoder, wav_1d, device: str):
    """wav (1D numpy или (T,) тензор) -> (latent, std). Латент остаётся на device."""
    if not torch.is_tensor(wav_1d):
        wav_1d = torch.tensor(wav_1d, dtype=torch.float32)
    audio = wav_1d.to(device).reshape(1, 1, -1)   # (B=1, C=1, T)
    latent, std = autoencoder(audio=audio)
    return latent, std


@torch.no_grad()
def run_diffusion(tse_model, scheduler, device, mix_latent, ref_latent,
                  num_infer_steps: int, eta: float = 0.0,
                  seed: int = config.RANDOM_SEED, on_step=None):
    """Diffusion-цикл для одного латента. Возвращает предсказанный латент (на device,
    ещё не декодированный). Логика перенесена из run_diffusion_one().

    on_step: опциональный callback(i, total) для прогресса GUI.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(num_infer_steps)

    mixture = mix_latent.transpose(2, 1).to(device)
    reference = ref_latent.transpose(2, 1).to(device)
    lengths = torch.LongTensor([mixture.shape[1]]).to(device)
    ref_lengths = torch.LongTensor([reference.shape[1]]).to(device)

    pred = torch.randn(mixture.shape, generator=generator, device=device)
    for i, t in enumerate(scheduler.timesteps):
        pred = scheduler.scale_model_input(pred, t)
        model_output, _ = tse_model(
            x=pred, timesteps=t, mixture=mixture, reference=reference,
            x_len=lengths, ref_len=ref_lengths,
        )
        pred = scheduler.step(
            model_output=model_output, timestep=t, sample=pred,
            eta=eta, generator=generator,
        ).prev_sample
        if on_step is not None:
            on_step(i + 1, num_infer_steps)

    return pred


@torch.no_grad()
def decode(autoencoder: Autoencoder, pred_latent, std, device: str):
    """Предсказанный латент -> wav (1, T) на device. std из соответствующего encode."""
    pred_latent = pred_latent.to(device)
    std = std.to(device) if std is not None else None
    wav = autoencoder(embedding=pred_latent.transpose(2, 1), std=std).squeeze(1)
    return wav


@torch.no_grad()
def correct(geco_model, pred_wav, mix_wav_1d, device: str):
    """Финальная коррекция Fast-GeCo. pred_wav: (1, T) или (T,) на любом device;
    mix_wav_1d: исходный mix-чанк (1D numpy или тензор). Возвращает 1D numpy.

    Логика перенесена ДОСЛОВНО из этапа correction обоих скриптов.
    """
    pred = pred_wav.to(device)
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if not torch.is_tensor(mix_wav_1d):
        mix_wav_1d = torch.tensor(mix_wav_1d, dtype=torch.float32)
    m = mix_wav_1d.to(device)
    if m.dim() == 1:
        m = m.unsqueeze(0)

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
        mean_x = X_t - (f - g**2 * geco_model.forward(
            X_t, vec_t, M, X, vec_t[:, None, None, None])) * dt
        if idx == len(timesteps) - 1:
            X_t = mean_x
            break
        z = torch.randn_like(X)
        X_t = mean_x + z * g * torch.sqrt(dt)

    sample = X_t.squeeze()
    x_hat = geco_model.to_audio(sample.squeeze(), min_leng)
    x_hat = x_hat * norm_factor / x_hat.abs().max()
    return x_hat.detach().cpu().numpy()
