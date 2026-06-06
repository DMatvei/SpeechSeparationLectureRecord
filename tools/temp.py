import torch
import soundfile as sf
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="***REMOVED***"
)

pipeline.to(torch.device("cuda"))
# Через soundfile — никакой torchcodec не нужен
audio, sample_rate = sf.read("output/input.wav")
waveform = torch.tensor(audio).float().unsqueeze(0)  # (1, samples)

diarization = pipeline({
    "waveform": waveform,
    "sample_rate": sample_rate
})

# Посмотри что за объект вернулся
print(type(diarization))
print(dir(diarization))

for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f} - {turn.end:.1f}: {speaker}")