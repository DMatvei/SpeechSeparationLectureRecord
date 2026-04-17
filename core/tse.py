import os
import sys

import torch
import librosa
import soundfile as sf
import numpy as np
from collections import OrderedDict

from attr.validators import max_len
from torch.utils.data import Dataset, DataLoader


USER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'USEF-TSE')
if USER_DIR not in sys.path:
    sys.path.insert(0, USER_DIR)


class InferenceDataset(Dataset):
    """Датасет для инференса, из оригинала убрали метрики"""

    def __init__(self, mix_scp, aux_scp, fs):
        self.mix = {x.split()[0]: x.split()[1] for x in open(mix_scp)}
        self.aux = {x.split()[0]: x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.aux)


        self.wav_id = list(self.mix.keys())
        self.fs = fs

    def __getitem__(self, item):
        utt = self.wav_id[item]
        mix_wav, _ = librosa.load(self.mix[utt], sr=self.fs)
        aux_wav, _ = librosa.load(self.aux[utt], sr=self.fs)

        return utt, torch.from_numpy(mix_wav), torch.from_numpy(aux_wav)

    def __len__(self):
        return len(self.wav_id)


def load_model(config_path: str, chkpt_path: str):
    """Загрузка модели и чекпоинта"""
    from  hyperpyyaml import  load_hyperpyyaml

    with open(config_path, "r") as f:
        config = load_hyperpyyaml(f.read())

    model = config['modules']['masknet']

    # загрузка весов
    model_info = torch.load(chkpt_path, map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace('module.', '').replace('convolution_', 'convolution_module.')
        state_dict[name] = v
    model.load_state_dict(state_dict)


    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    return model, config.get('sample_rate', 8000)


def run_tse(model, mix_scp: str, aux_scp: str, output_dir: str,
            sample_rate: int = 8000, chunk_seconds: int = 15):
    """ЗАпуск TSE"""
    os.makedirs(output_dir, exist_ok=True)

    dataset = InferenceDataset(mix_scp, aux_scp, fs=sample_rate)

    results = []

    with torch.no_grad():
        for i in range(len(dataset)):
            utt_id, mix_wav, aux_wav = dataset[i]

            chunk_size = chunk_seconds * sample_rate
            mix_len = mix_wav.shape[0]
            chunks_out = []

            for start in range(0, mix_len, chunk_size):
                end = min(start + chunk_size, mix_len)
                mix_chunk = mix_wav[start:end].unsqueeze(0)

                if torch.cuda.is_available():
                    mix_chunk = mix_chunk.cuda()
                    aux = aux_wav.unsqueeze(0).cuda()
                else:
                    mix_chunk = mix_chunk
                    aux = aux_wav.unsqueeze(0)

                est = model(mix_chunk, aux)
                chunks_out.append(est.squeeze().cpu().numpy())

                del mix_chunk, est
                torch.cuda.empty_cache()

            full_output = np.concatenate(chunks_out)
            out_path = os.path.join(output_dir, f'{utt_id}.wav')
            sf.write(out_path, full_output, sample_rate)
            results.append(out_path)
            print(f'Save: {out_path}')

    return results


