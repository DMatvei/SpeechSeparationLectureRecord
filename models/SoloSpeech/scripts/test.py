# @ hwang258@jhu.edu
import yaml
import random
import argparse
import os
import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler
from solospeech.model.solospeech.conditioners import SoloSpeech_TSE
from solospeech.model.solospeech.conditioners import SoloSpeech_TSR
from solospeech.scripts.solospeech.utils import save_audio
import shutil
from solospeech.vae_modules.autoencoder_wrapper import Autoencoder
import pandas as pd
from speechbrain.pretrained.interfaces import Pretrained
from solospeech.corrector.fastgeco.model import ScoreModel
from solospeech.corrector.geco.util.other import pad_spec
from huggingface_hub import snapshot_download

class Encoder(Pretrained):

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings


@torch.no_grad()
def sample_diffusion(tse_model, tsr_model, autoencoder, std, scheduler, device,
                     mixture=None, reference=None, lengths=None, reference_lengths=None, 
                     ddim_steps=50, eta=0, seed=2025
                     ):

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(ddim_steps)
    tse_pred = torch.randn(mixture.shape, generator=generator, device=device)
    tsr_pred = torch.randn(mixture.shape, generator=generator, device=device)

    for t in scheduler.timesteps:
        tse_pred = scheduler.scale_model_input(tse_pred, t)
        model_output, _ = tse_model(
            x=tse_pred, 
            timesteps=t, 
            mixture=mixture, 
            reference=reference, 
            x_len=lengths, 
            ref_len=reference_lengths
            )
        tse_pred = scheduler.step(model_output=model_output, timestep=t, sample=tse_pred,
                                eta=eta, generator=generator).prev_sample
    
    for t in scheduler.timesteps:
        tsr_pred = scheduler.scale_model_input(tsr_pred, t)
        model_output, _ = tsr_model(
            x=tsr_pred, 
            timesteps=t, 
            mixture=mixture, 
            reference=tse_pred, 
            x_len=lengths, 
            )
        tsr_pred = scheduler.step(model_output=model_output, timestep=t, sample=tsr_pred,
                                eta=eta, generator=generator).prev_sample

    tse_pred = autoencoder(embedding=tse_pred.transpose(2,1), std=std).squeeze(1)
    tsr_pred = autoencoder(embedding=tsr_pred.transpose(2,1), std=std).squeeze(1)

    return tse_pred, tsr_pred



def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("Downloading model from Huggingface...")
    local_dir = snapshot_download(
        repo_id="OpenSound/SoloSpeech-models"
    )
    args.tse_config = os.path.join(local_dir, "config_extractor.yaml")
    args.tsr_config = os.path.join(local_dir, "config_tsr.yaml")
    args.vae_config = os.path.join(local_dir, "config_compressor.json")
    args.autoencoder_path = os.path.join(local_dir, "compressor.ckpt")
    args.tse_ckpt = os.path.join(local_dir, "extractor.pt")
    args.tsr_ckpt = os.path.join(local_dir, "tsr.pt")
    args.geco_ckpt = os.path.join(local_dir, "corrector.ckpt")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    # load config
    print("Loading models...")
    with open(args.tse_config, 'r') as fp:
        args.tse_config = yaml.safe_load(fp)
    with open(args.tsr_config, 'r') as fp:
        args.tsr_config = yaml.safe_load(fp)
    args.v_prediction = args.tse_config["ddim"]["v_prediction"]
    # load compressor
    autoencoder = Autoencoder(args.autoencoder_path, args.vae_config, 'stft_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(device)
    # load extractor
    tse_model = SoloSpeech_TSE(
        args.tse_config['diffwrap']['UDiT'],
        args.tse_config['diffwrap']['ViT'],
    ).to(device)
    tse_model.load_state_dict(torch.load(args.tse_ckpt)['model'])
    tse_model.eval()
    # load tsr model
    tsr_model = SoloSpeech_TSR(
        args.tsr_config['diffwrap']['UDiT']
    ).to(device)
    tsr_model.load_state_dict(torch.load(args.tsr_ckpt)['model'])
    tsr_model.eval()
    # load corrector
    geco_model = ScoreModel.load_from_checkpoint(
        args.geco_ckpt,
        batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    geco_model.eval(no_ema=False)
    geco_model.cuda()
    # load sid model
    ecapatdnn_model = Encoder.from_hparams(source="yangwang825/ecapa-tdnn-vox2")
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    # load diffusion tools
    noise_scheduler = DDIMScheduler(**args.tse_config["ddim"]['diffusers'])
    # these steps reset dtype of noise_scheduler params
    latents = torch.randn((1, 128, 128),
                          device=device)
    noise = torch.randn(latents.shape).to(device)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (noise.shape[0],),
                              device=latents.device).long()
    _ = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # 
    print("Start Extraction...")
    mixture, _ = librosa.load(args.test_wav, sr=16000)
    reference, _ = librosa.load(args.enroll_wav, sr=16000)
    reference_wav = reference
    reference = torch.tensor(reference).unsqueeze(0).to(device)
    with torch.no_grad():
        # compressor
        reference, _ = autoencoder(audio=reference.unsqueeze(1))
        reference_lengths = torch.LongTensor([reference.shape[-1]]).to(device)
        mixture_input = torch.tensor(mixture).unsqueeze(0).to(device)
        mixture_wav = mixture_input
        mixture_input, std = autoencoder(audio=mixture_input.unsqueeze(1))
        lengths = torch.LongTensor([mixture_input.shape[-1]]).to(device)   
        # extractor
        tse_pred, tsr_pred = sample_diffusion(tse_model, tsr_model, autoencoder, std, noise_scheduler, device, mixture_input.transpose(2,1), reference.transpose(2,1), lengths, reference_lengths, ddim_steps=args.num_infer_steps, eta=args.eta, seed=args.random_seed)
        ecapatdnn_embedding1 = ecapatdnn_model.encode_batch(tse_pred.squeeze()).squeeze()
        ecapatdnn_embedding2 = ecapatdnn_model.encode_batch(tsr_pred.squeeze()).squeeze()
        ecapatdnn_embedding3 = ecapatdnn_model.encode_batch(torch.tensor(reference_wav)).squeeze()
        sim1 = cosine_sim(ecapatdnn_embedding1, ecapatdnn_embedding3).item()
        sim2 = cosine_sim(ecapatdnn_embedding2, ecapatdnn_embedding3).item()
        pred = tse_pred if sim1 > sim2 else tsr_pred
        # corrector
        min_leng = min(pred.shape[-1], mixture_wav.shape[-1])
        x = pred[...,:min_leng]
        m = mixture_wav[...,:min_leng]
        norm_factor = m.abs().max()
        x = x / norm_factor
        m = m / norm_factor 
        X = torch.unsqueeze(geco_model._forward_transform(geco_model._stft(x.cuda())), 0)
        X = pad_spec(X)
        M = torch.unsqueeze(geco_model._forward_transform(geco_model._stft(m.cuda())), 0)
        M = pad_spec(M)
        timesteps = torch.linspace(0.5, 0.03, 1, device=M.device)
        std = geco_model.sde._std(0.5*torch.ones((M.shape[0],), device=M.device))
        z = torch.randn_like(M)
        X_t = M + z * std[:, None, None, None]

        for idx in range(len(timesteps)):
            t = timesteps[idx]
            if idx != len(timesteps) - 1:
                dt = t - timesteps[idx+1]
            else:
                dt = timesteps[-1]
            with torch.no_grad():
                f, g = geco_model.sde.sde(X_t, t, M)
                vec_t = torch.ones(M.shape[0], device=M.device) * t 
                mean_x_tm1 = X_t - (f - g**2*geco_model.forward(X_t, vec_t, M, X, vec_t[:,None,None,None]))*dt
                if idx == len(timesteps) - 1:
                    X_t = mean_x_tm1 
                    break
                z = torch.randn_like(X) 
                X_t = mean_x_tm1 + z*g*torch.sqrt(dt)

        sample = X_t
        sample = sample.squeeze()
        x_hat = geco_model.to_audio(sample.squeeze(), min_leng)
        x_hat = x_hat * norm_factor / x_hat.abs().max()
        x_hat = x_hat.detach().cpu()
        
        save_audio(args.output_path, 16000, x_hat)
        print(f"Save to : {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--test-wav', type=str, required=True)
    parser.add_argument('--enroll-wav', type=str, required=True)
    # pre-trained model path
    parser.add_argument('--eta', type=int, default=0)
    parser.add_argument("--num_infer_steps", type=int, default=200)
    parser.add_argument('--sample-rate', type=int, default=16000)
    # random seed
    parser.add_argument('--random-seed', type=int, default=42, help="Fixed seed")
    args = parser.parse_args()
    main(args)