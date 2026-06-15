# @ hwang258@jhu.edu
# Fast and light-weight inference. We removed tsr and corrector models.
import yaml
import random
import argparse
import os
import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler
from solospeech.model.solospeech.conditioners import SoloSpeech_TSE
from solospeech.scripts.solospeech.utils import save_audio
import shutil
from solospeech.vae_modules.autoencoder_wrapper import Autoencoder
import pandas as pd
from huggingface_hub import snapshot_download


@torch.no_grad()
def sample_diffusion(tse_model, autoencoder, std, scheduler, device,
                     mixture=None, reference=None, lengths=None, reference_lengths=None, 
                     ddim_steps=50, eta=0, seed=2025
                     ):

    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(ddim_steps)
    tse_pred = torch.randn(mixture.shape, generator=generator, device=device)

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

    tse_pred = autoencoder(embedding=tse_pred.transpose(2,1), std=std).squeeze(1)

    return tse_pred


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("Downloading model from Huggingface...")
    local_dir = snapshot_download(
        repo_id="OpenSound/SoloSpeech-models"
    )
    args.tse_config = os.path.join(local_dir, "config_extractor.yaml")
    args.vae_config = os.path.join(local_dir, "config_compressor.json")
    args.autoencoder_path = os.path.join(local_dir, "compressor.ckpt")
    args.tse_ckpt = os.path.join(local_dir, "extractor.pt")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    # load config
    print("Loading models...")
    with open(args.tse_config, 'r') as fp:
        args.tse_config = yaml.safe_load(fp)
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
        pred = sample_diffusion(tse_model, autoencoder, std, noise_scheduler, device, mixture_input.transpose(2,1), reference.transpose(2,1), lengths, reference_lengths, ddim_steps=args.num_infer_steps, eta=args.eta, seed=args.random_seed)
        
        save_audio(args.output_path, 16000, pred)
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
