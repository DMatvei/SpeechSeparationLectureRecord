import os
from core.converter import convert_to_wav
from core.diarization import load_diarization_pipeline, diarize, find_lector, extract_refs

_diar_pipeline = None

def _ger_diar_pipeline():
    global _diar_pipeline
    if _diar_pipeline is None:
        _diar_pipeline = load_diarization_pipeline(token="***REMOVED***")
    return _diar_pipeline


def process(input_path: str, output_dir: str, on_progress=None):
    os.makedirs(output_dir, exist_ok=True)

    # конвертация
    if input_path.lower().endswith(".wav"):
        wav_path = input_path
    else:
        wav_path = os.path.join(output_dir, "input.wav")
        convert_to_wav(input_path, wav_path)

    if on_progress: on_progress(20)


    # диаризация
    pipeline = _ger_diar_pipeline()
    diarization = diarize(pipeline, wav_path)
    if on_progress: on_progress(30)

    # создаю референсы голоса преподавателя
    lector = find_lector(diarization)
    refs_dir = os.path.join(output_dir, "refs")
    ref_paths = extract_refs(wav_path, diarization, lector, refs_dir)
    if on_progress: on_progress(40)

    # todo scp
    # todo TSE

    if on_progress: on_progress(100)
    return wav_path