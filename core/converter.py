import subprocess
import os

# путь к ffmpeg

FFMPEG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),"..", "tools", "ffmpeg.exe"
)
FFMPEG_PATH = os.path.normpath(FFMPEG_PATH)
print(f"Ищу ffmpeg: {FFMPEG_PATH}")
print(f"Файл существует: {os.path.exists(FFMPEG_PATH)}")


def convert_to_wav(input_path: str, output_path: str, sr: int = 16_000):
    subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-ar", str(sr),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ], check=True)

