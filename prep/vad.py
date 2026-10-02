"""Silero VAD on the original audio (no Demucs).

Without --out (pipeline gate): per-video VAD stats go to <src>/vad.jsonl and no audio is written.
transcribe.py reads that file and marks videos with almost no Silero speech as no_speech without
calling Whisper.

With --out (experiment): for every video with an audio track in <src>/manifest.jsonl this also
writes a 16 kHz mono WAV on the original timeline, so transcript timestamps still match the video:
  <out>/silero/<name>.wav   original mix, silenced outside Silero VAD speech
The folder gets a manifest.jsonl pointing at its WAVs (same layout as prep.py), and the stats go
to <out>/silero/vad.jsonl.

Videos already in vad.jsonl are skipped.

Runs in the audio-prep:local image (see Dockerfile):
  docker run --rm -v "C:\\Users\\wayan\\Downloads:/data" -v "D:\\audio-converter\\prep:/prep:ro" -v spk-cache:/cache
         audio-prep:local python /prep/vad.py --src /data/<batch> [--out /data/<experiment>]
"""
import argparse
import json
import pathlib
import subprocess
import time

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from prep import SR, VAD, decode, write_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", help="experiment folder for silenced WAVs; omit to only write <src>/vad.jsonl")
    a = ap.parse_args()
    src = pathlib.Path(a.src)
    out = pathlib.Path(a.out) / "silero" if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    stats_path = (out or src) / "vad.jsonl"
    done = {json.loads(l)["_id"] for l in stats_path.read_text(encoding="utf-8").splitlines() if l} if stats_path.exists() else set()
    docs = {}
    for line in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line and (d := json.loads(line)).get("file"):
            docs[d["_id"]] = d  # last record per _id wins, as in transcribe.py
    vad = load_silero_vad()
    with open(stats_path, "a", encoding="utf-8") as stats:
        for n, d in enumerate(docs.values(), 1):
            if d["_id"] in done:
                continue
            rec = {"_id": d["_id"], "source_file": d["file"]}
            t0 = time.time()
            try:
                x = decode(src / d["file"], SR, 1)[0]
            except subprocess.CalledProcessError:
                x = np.zeros(0, np.float32)
            if len(x) == 0:
                rec["status"] = "no_audio"
            else:
                spans = get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=SR, return_seconds=True, **VAD)
                mask = np.zeros_like(x)
                for s in spans:
                    mask[int(s["start"] * SR):int(s["end"] * SR)] = 1
                rec["vad_seconds"] = round(time.time() - t0, 1)
                rec["audio_seconds"] = round(len(x) / SR, 1)
                rec["vad_speech_seconds"] = round(float(mask.sum() / SR), 1)
                rec["vad_spans"] = [[round(s["start"], 2), round(s["end"], 2)] for s in spans]
                if out:
                    write_wav(out / (pathlib.Path(d["file"]).stem + ".wav"), x * mask)
                rec["status"] = "ok"
            stats.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.flush()
            print(f"[{n:2}/{len(docs)}] {d['_id'][:24]:24} {rec['status']:8} audio={rec.get('audio_seconds', 0):6}s "
                  f"vad={rec.get('vad_seconds', 0):5}s speech={rec.get('vad_speech_seconds', 0)}s", flush=True)
    if not out:
        print(f"-> {stats_path}")
        return
    recs = {r["_id"]: r for r in (json.loads(l) for l in stats_path.read_text(encoding="utf-8").splitlines() if l)}
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as m:
        for doc in docs.values():
            ok = recs.get(doc["_id"], {}).get("status") == "ok"
            m.write(json.dumps(doc | {"file": pathlib.Path(doc["file"]).stem + ".wav" if ok else None,
                                      "source_file": doc["file"]}, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
