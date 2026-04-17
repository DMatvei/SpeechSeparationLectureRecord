import os
from core.converter import convert_to_wav
from core.diarization import load_diarization_pipeline, diarize, find_lector, extract_refs
from core.scp import generate_scp_files
from core.tse import load_model, run_tse



_diar_pipeline = None
_tse_model = None
_sample_rate = 8000






def _ger_diar_pipeline():
    global _diar_pipeline
    if _diar_pipeline is None:
        _diar_pipeline = load_diarization_pipeline(token="***REMOVED***")
    return _diar_pipeline


def _get_tse_model():
    global _tse_model, _sample_rate
    if _tse_model is None:
        _tse_model, _sample_rate = load_model(
            config_path="USEF-TSE/chkpt/USEF-TFGridNet/config.yaml",
            chkpt_path="USEF-TSE/chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar"
        )

    return _tse_model



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
    global _diar_pipeline
    pipeline = _ger_diar_pipeline()
    diarization = diarize(pipeline, wav_path)
    if on_progress: on_progress(30)

    # создаю референсы голоса преподавателя
    lector = find_lector(diarization)
    refs_dir = os.path.join(output_dir, "refs")
    ref_paths = extract_refs(wav_path, diarization, lector, refs_dir,
                             min_dur=20.0, num_refs=1, max_dur=50.0)

    # чистю видеопамять
    import torch, gc
    del pipeline
    _diar_pipeline = None
    gc.collect()
    torch.cuda.empty_cache()

    if on_progress: on_progress(40)


    # Получение scp Файлов
    mix_scp, aux_scp = generate_scp_files(wav_path, ref_paths, output_dir)
    if on_progress: on_progress(50)

    # TSE
    model = _get_tse_model()
    results_tse = run_tse(
        model, mix_scp, aux_scp,
        os.path.join(output_dir, 'tse_out'), _sample_rate
    )
    if on_progress: on_progress(80)

    if on_progress: on_progress(100)
    return wav_path