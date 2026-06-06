import os
from core.converter import convert_to_wav

import soundfile as sf



_sample_rate = 16000













def process(input_path: str, output_dir: str, on_progress=None):
    os.makedirs(output_dir, exist_ok=True)

    # конвертация
    info = sf.info(input_path)
    check_sr = info.samplerate == 8000
    if input_path.lower().endswith(".wav") and info.channels == 1:
        wav_path = input_path
    else:
        wav_path = os.path.join(output_dir, "input.wav")
        convert_to_wav(input_path, wav_path)

    if on_progress: on_progress(20)





    if on_progress: on_progress(40)





    # TSE

    if on_progress: on_progress(80)

    if on_progress: on_progress(100)
    return wav_path