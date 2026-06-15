import os

# пути =============================================================

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


# Чекпоинты SoloSpeech --------------------------------------------
CHECKPOINTS_DIR = os.path.join(
    PROJECT_ROOT, 'models', "SoloSpeech", 'checkpoints'
)

COMPRESSOR_CKPT = os.path.join(CHECKPOINTS_DIR, "compressor.ckpt")
EXTRACTOR_CKPT = os.path.join(CHECKPOINTS_DIR, "extractor.pt")
CORRECTOR_CKPT = os.path.join(CHECKPOINTS_DIR, "corrector.ckpt")

# конфиги чекпоинтова ---------------------------------------------

COMPRESSOR_CONFIG = os.path.join(CHECKPOINTS_DIR, 'config_compressor.json')
EXTRACTOR_CONFIG = os.path.join(CHECKPOINTS_DIR, 'config_extractor.yaml')


# ffmpeg ----------------------------------------------------------
#TODO подумать над решением для линукс

FFMPEG_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, 'tools', 'ffmpeg.exe'))


# настройки для аудио ==============================================

SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SEC = 15.0
DEFAULT_OVERLAP_SEC = 1.0
RANDOM_SEED = 42

SHORT_AUDIO_THRESHOLD_SEC = 20.0

# пресеты качества =================================================
QUALITY_PRESETS = {
    "low":    {"num_infer_steps": 20, "label": "Быстрый"},
    "medium": {"num_infer_steps": 30, "label": "Стандарт"},
    "high":   {"num_infer_steps": 50, "label": "Максимум"},
}

DEFAULT_QUALITY = 'medium'

# проверка чекпоинтов. пустой возврат - значит всё ок ------------------
def check_checkpoints() -> list[str]:

    required = {
        "compressor.ckpt": COMPRESSOR_CKPT,
        "extractor.pt": EXTRACTOR_CKPT,
        "corrector.ckpt": CORRECTOR_CKPT,
        "config_compressor.json": COMPRESSOR_CONFIG,
        "config_extractor.yaml": EXTRACTOR_CONFIG,
    }

    return [name for name, path in required.items() if not os.path.exists(path)]
