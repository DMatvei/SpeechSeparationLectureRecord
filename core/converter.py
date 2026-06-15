import subprocess
import os
from core.config import FFMPEG_PATH, SAMPLE_RATE


print(f"Ищу ffmpeg: {FFMPEG_PATH}")
print(f"Файл существует: {os.path.exists(FFMPEG_PATH)}")


def convert_to_wav(input_path: str, output_path: str, sr: int = SAMPLE_RATE):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-ar", str(sr),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ], check=True)

