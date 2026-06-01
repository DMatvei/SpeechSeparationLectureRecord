import os

import torch
import soundfile as sf
from pyannote.audio import  Pipeline
import noisereduce as nr




# todo Вытащить модель, чтобы была доступна локально

def load_diarization_pipeline(model_path :str = None, token: str = None):
    """Загрузка пайплайна диаризации"""
    if model_path and os.path.exists(model_path):
        pipeline = Pipeline.from_pretrained(model_path)
    else:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token
        )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    return pipeline

def diarize(pipeline, wav_path: str):
    """Запуск диарилизации, возвращает объект с результатами обнаружение спикеров"""
    audio, sample_rate = sf.read(wav_path)
    waveform = torch.tensor(audio).float().unsqueeze(0)

    result = pipeline({"waveform":waveform, "sample_rate": sample_rate})
    return result.speaker_diarization


def find_lector(diarization):
    """Спикер с самым большим суммарным временем - это преподаватель"""
    speaker_time = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_time[speaker] = speaker_time.get(speaker, 0) + (turn.end - turn.start)

    lector = max(speaker_time, key=speaker_time.get)
    print(f"Преподаватель: {lector} ({speaker_time[lector]:.1f} сек)")
    return lector


def extract_refs(wav_path: str, diarization, lector: str,
                 output_path: str, num_refs: int = 5,
                 min_dur: float = 3.0, max_dur: float = 10.0
    ):
    """Нарезка референсов из основной записи"""
    audio, sr = sf.read(wav_path)
    os.makedirs(output_path, exist_ok=True)

    ref_paths = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker  != lector:
            continue

        duration = turn.end - turn.start
        if min_dur <= duration <= max_dur:
            start = int(turn.start * sr)
            end = int(turn.end * sr)
            path = os.path.join(output_path, f"ref_{len(ref_paths):03d}.wav")
            sf.write(path, audio[start:end], sr)
            ref_paths.append(path)
        if len(ref_paths) >= num_refs:
            break
    print(f"Сохранено референсов: {len(ref_paths)}")
    return ref_paths


def save_speaker_timeline(diarizatoin, lector: str, output_path: str):
    """Выводит моменты с не основными спикерами"""
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, f'speaker.txt')
    with open(path, "w", encoding='utf-8') as f:
        f.write(f'Преподаватель: {lector}\n')
        f.write(f'{"=" * 60}\n')

        for turn, _, speaker in diarizatoin.itertracks(yield_label=True):
            if speaker != lector:
                mins_s = int(turn.start // 60)
                secs_s = turn.start % 60
                mins_e = int(turn.end // 60)
                secs_e = turn.end % 60
                dur = turn.end - turn.start
                f.write(f'{speaker}: {mins_s:02d}:{secs_s:05.2f} - '
                        f'{mins_e:02d}:{secs_e:05.2f} ({dur:.1f} сек)\n')
    print(f'Таймлайн сохранён: {path}')


def clean_ref(input_path: str, output_path: str):
    """Очистка референса от шума"""
    audio, sr = sf.read(input_path)
    cleaned = nr.reduce_noise(y=audio, sr=sr)
    sf.write(output_path, cleaned, sr)
    return output_path