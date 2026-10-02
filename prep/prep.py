"""Clean a batch's audio before Whisper: Demucs vocal separation, then Silero VAD.

For every video with an audio track in <src>/manifest.jsonl this writes two 16 kHz mono WAVs,
both on the original timeline so transcript timestamps still match the video:
  <out>/demucs/<name>.wav          vocals stem from htdemucs (music and background removed)
  <out>/demucs_silero/<name>.wav   the same vocals, silenced outside Silero VAD speech
Each folder gets a manifest.jsonl pointing at its WAVs, so transcribe.py-style runners and
speaker_id.py score can use it as a batch. Per-video timings and VAD stats go to <out>/prep.jsonl.
Videos already in prep.jsonl are skipped.

Runs in the audio-prep:local image (see Dockerfile):
  docker run --rm -v "C:\\Users\\wayan\\Downloads:/data" -v "D:\\audio-converter\\prep:/prep:ro" -v spk-cache:/cache
         audio-prep:local python /prep/prep.py --src /data/<batch> --out /data/<experiment>
"""
import argparse
import json
import os
import pathlib
import subprocess
import time
import wave

import numpy as np
import torch
import torchaudio.functional as AF
from demucs.apply import apply_model
from demucs.pretrained import get_model
from silero_vad import get_speech_timestamps, load_silero_vad

SR = 16000
# Silero defaults except the padding: 30 ms clips word onsets, 200 ms keeps them.
VAD = {"threshold": 0.5, "min_speech_duration_ms": 250, "min_silence_duration_ms": 300, "speech_pad_ms": 200}


def decode(path, sr, channels):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", str(channels),
                          "-ar", str(sr), "-f", "f32le", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T.copy()


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def vocals(model, path):
    """Mono 16 kHz vocals stem and the mono 16 kHz mix it came from."""
    mix = torch.from_numpy(decode(path, model.samplerate, 2))
    if mix.shape[1] == 0:
        raise subprocess.CalledProcessError(0, "ffmpeg", b"", b"no audio samples")
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    out = apply_model(model, ((mix - mean) / std)[None], shifts=1, split=True, overlap=0.25,
                      progress=False, device="cpu")[0] * std + mean
    voc = out[model.sources.index("vocals")].mean(0)
    return (AF.resample(voc, model.samplerate, SR).numpy(),
            AF.resample(mix.mean(0), model.samplerate, SR).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    dirs = {k: out / k for k in ("demucs", "demucs_silero")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    stats_path = out / "prep.jsonl"
    done = {json.loads(l)["_id"] for l in stats_path.read_text(encoding="utf-8").splitlines() if l} if stats_path.exists() else set()
    docs = {}
    for line in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line and (d := json.loads(line)).get("file"):
            docs[d["_id"]] = d  # last record per _id wins, as in transcribe.py
    torch.set_num_threads(os.cpu_count() or 4)
    model = get_model("htdemucs").cpu().eval()
    vad = load_silero_vad()
    with open(stats_path, "a", encoding="utf-8") as stats:
        for n, d in enumerate(docs.values(), 1):
            if d["_id"] in done:
                continue
            rec = {"_id": d["_id"], "source_file": d["file"]}
            name = pathlib.Path(d["file"]).stem + ".wav"
            t0 = time.time()
            try:
                with torch.no_grad():
                    voc, mix = vocals(model, src / d["file"])
            except subprocess.CalledProcessError:
                rec["status"] = "no_audio"
            else:
                rec["demucs_seconds"] = round(time.time() - t0, 1)
                rec["audio_seconds"] = round(len(mix) / SR, 1)
                # share of the mix's energy left in the vocals stem: low = mostly music or noise
                rec["vocals_energy_share"] = round(float((voc ** 2).sum() / max((mix ** 2).sum(), 1e-9)), 3)
                t1 = time.time()
                spans = get_speech_timestamps(torch.from_numpy(voc), vad, sampling_rate=SR, return_seconds=True, **VAD)
                mask = np.zeros_like(voc)
                for s in spans:
                    mask[int(s["start"] * SR):int(s["end"] * SR)] = 1
                rec["vad_seconds"] = round(time.time() - t1, 1)
                rec["vad_speech_seconds"] = round(float(mask.sum() / SR), 1)
                rec["vad_spans"] = [[round(s["start"], 2), round(s["end"], 2)] for s in spans]
                write_wav(dirs["demucs"] / name, voc)
                write_wav(dirs["demucs_silero"] / name, voc * mask)
                rec["status"] = "ok"
            stats.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.flush()
            print(f"[{n:2}/{len(docs)}] {d['_id'][:24]:24} {rec['status']:8} audio={rec.get('audio_seconds', 0):6}s "
                  f"demucs={rec.get('demucs_seconds', 0):5}s vocals={rec.get('vocals_energy_share', 0):.2f} "
                  f"vad_speech={rec.get('vad_speech_seconds', 0)}s", flush=True)
    recs = {r["_id"]: r for r in (json.loads(l) for l in stats_path.read_text(encoding="utf-8").splitlines() if l)}
    for d in dirs.values():
        with open(d / "manifest.jsonl", "w", encoding="utf-8") as m:
            for doc in docs.values():
                ok = recs.get(doc["_id"], {}).get("status") == "ok"
                m.write(json.dumps(doc | {"file": pathlib.Path(doc["file"]).stem + ".wav" if ok else None,
                                          "source_file": doc["file"]}, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
