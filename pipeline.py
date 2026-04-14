import os
from core.converter import convert_to_wav

def process(input_path: str, output_dir: str, on_progress=None):
    os.makedirs(output_dir, exist_ok=True)

    # конвертация

    wav_path = os.path.join(output_dir, "input.wav")
    convert_to_wav(input_path, wav_path)
    if on_progress: on_progress(20)


    # todo диаризация
    # todo scp
    # todo TSE

    if on_progress: on_progress(100)
    return wav_path