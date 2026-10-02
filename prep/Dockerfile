# Audio clean-up before Whisper: Demucs (htdemucs) vocal separation and Silero VAD, on CPU.
# Built on speaker-id:local so torch/torchaudio 2.5.1 and ffmpeg are shared; the constraints
# file stops pip from upgrading torch while installing demucs and silero-vad.
FROM speaker-id:local

RUN pip freeze | grep -E '^(torch|torchaudio|numpy)==' > /tmp/pins.txt \
    && pip install --no-cache-dir -c /tmp/pins.txt demucs==4.0.1 silero-vad==5.1.2

ENV TORCH_HOME=/cache/torch
WORKDIR /prep
