# Speaker verification toolbox: SpeechBrain ECAPA-TDNN on CPU, ffmpeg for decoding.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pinned: newer torchaudio (2.9+) drops the audio backends SpeechBrain 1.0 probes for.
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir speechbrain==1.0.3 huggingface_hub==0.25.2 requests

ENV HF_HOME=/cache/hf
WORKDIR /app
