"""Check whether a known speaker (e.g. Prabowo) is the one talking in each video.

Uses SpeechBrain ECAPA-TDNN speaker embeddings. Audio is cut into 3 s windows
(1.5 s hop) inside the speech segments Whisper found, and every window is
compared by cosine similarity to a reference voiceprint.

Runs inside the speaker-id:local container (see Dockerfile). Subcommands:
  enroll --audio /data/ref.mp3 --name prabowo   build a voiceprint from a clean reference clip
  test   --audio /data/clip.mp3 --name prabowo  score one clip
  score  --dir /data/<batch> --name prabowo     score transcribed videos, write speakers.jsonl
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np
import requests
import torch
from speechbrain.inference.speaker import EncoderClassifier

SR = 16000
WIN, HOP, MIN_LEN = 3.0, 1.5, 1.0  # seconds
REFS = pathlib.Path("/app/refs")
SPEACHES_URL = "http://host.docker.internal:8010/v1/audio/transcriptions"
WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Phrases Whisper emits on music or silence; such segments are not speech (same list as transcribe.py).
HALLUCINATIONS = {"terima kasih", "terima kasih telah menonton", "sampai jumpa", "thank you",
                  "thanks for watching", "you"}

_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        _encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                  savedir="/cache/models/ecapa", run_opts={"device": "cpu"})
    return _encoder


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def whisper_segments(path):
    """Speech segments for a clip that has not been through transcribe.py."""
    with open(path, "rb") as f:
        r = requests.post(SPEACHES_URL, files={"file": (path.name, f)}, timeout=1800,
                          data={"model": WHISPER_MODEL, "response_format": "verbose_json", "temperature": "0"})
    r.raise_for_status()
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
             "no_speech_prob": s.get("no_speech_prob", 0)} for s in r.json()["segments"]]


def windows(segments, audio_seconds):
    """(segment_index, start, end) windows inside speech segments."""
    out = []
    for i, s in enumerate(segments):
        start, end = s["start"], min(s["end"], audio_seconds)
        if s.get("no_speech_prob", 0) >= 0.6 or end - start < MIN_LEN:
            continue
        if s.get("hallucination") or s["text"].lower().strip(" .!?") in HALLUCINATIONS:
            continue  # flagged by transcribe.py, or a stock phrase in a clip it has not seen
        t = start
        while True:
            out.append((i, t, min(t + WIN, end)))
            if t + WIN >= end:
                break
            t += HOP
    return out


def embed(audio, wins, batch=32):
    """L2-normalised embedding per window, shape (n, 192)."""
    embs = []
    for b in range(0, len(wins), batch):
        chunk = [audio[int(s * SR):int(e * SR)] for _, s, e in wins[b:b + batch]]
        longest = max(len(c) for c in chunk)
        wavs = torch.zeros(len(chunk), longest)
        for j, c in enumerate(chunk):
            wavs[j, :len(c)] = torch.from_numpy(c.copy())
        lens = torch.tensor([len(c) / longest for c in chunk])
        with torch.no_grad():
            embs.append(encoder().encode_batch(wavs, lens).squeeze(1).numpy())
    e = np.concatenate(embs)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def dominant_cluster(E, thr):
    """Centroid of the largest group of mutually similar windows (the main speaker)."""
    S = E @ E.T
    seed = int((S > thr).sum(1).argmax())
    c = E[S[seed] > thr].mean(0)
    for _ in range(20):
        c = c / np.linalg.norm(c)
        members = E @ c >= thr
        new = E[members].mean(0)
        new = new / np.linalg.norm(new)
        if np.allclose(new, c, atol=1e-5):
            break
        c = new
    members = E @ c >= thr
    if not members.any():
        members[seed] = True
    return c, members


def speaker_clusters(E, thr):
    """Greedy diarization: peel off the dominant voice until every window has a cluster id,
    then merge clusters whose centroids are still similar (peeling can split one voice in two)."""
    labels = np.full(len(E), -1)
    k = 0
    while (labels == -1).any():
        idx = np.where(labels == -1)[0]
        _, members = dominant_cluster(E[idx], thr)
        labels[idx[members]] = k
        k += 1
    while True:
        ids = np.unique(labels)
        if len(ids) < 2:
            break
        C = np.stack([E[labels == i].mean(0) for i in ids])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ C.T
        np.fill_diagonal(S, -1)
        a, b = np.unravel_index(S.argmax(), S.shape)
        if S[a, b] < thr:
            break
        labels[labels == ids[b]] = ids[a]
    return np.unique(labels, return_inverse=True)[1]


def score_clip(audio, segments, ref, threshold, cluster_thr=0.45):
    """Cluster the clip's windows into voices, then compare each voice's centroid to the reference.

    A centroid averages many windows, so it is less affected by noise and music than
    single windows; one voice gets one verdict for the whole clip.
    """
    wins = windows(segments, len(audio) / SR)
    if not wins:
        return {"verdict": "tanpa_ucapan", "target_seconds": 0.0, "speech_seconds": 0.0, "segments": []}
    E = embed(audio, wins)
    sims = E @ ref
    labels = speaker_clusters(E, cluster_thr)
    voices = []
    for k in range(labels.max() + 1):
        c = E[labels == k].mean(0)
        sim = float(c @ ref / np.linalg.norm(c))
        voices.append({"voice": k, "windows": int((labels == k).sum()), "sim": round(sim, 3),
                       "label": "target" if sim >= threshold else "lain"})
    segs = []
    for i, s in enumerate(segments):
        w = [j for j, (k, _, _) in enumerate(wins) if k == i]
        if not w:
            continue
        voice = int(np.bincount(labels[w]).argmax())
        segs.append({"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"],
                     "sim": round(float(sims[w].mean()), 3), "voice": voice, "label": voices[voice]["label"]})
    dur = lambda lab: round(sum(s["end"] - s["start"] for s in segs if lab is None or s["label"] == lab), 1)
    target, speech = dur("target"), dur(None)
    verdict = "ya" if target >= 3.0 else "tidak"
    for v in voices:
        v["seconds"] = round(sum(s["end"] - s["start"] for s in segs if s["voice"] == v["voice"]), 1)
    return {"verdict": verdict, "target_seconds": target, "speech_seconds": speech,
            "target_share": round(target / speech, 3) if speech else 0.0,
            "max_sim": max(v["sim"] for v in voices), "n_voices": sum(v["seconds"] >= 3.0 for v in voices),
            "voices": voices, "segments": segs}


def cmd_enroll(a):
    path = pathlib.Path(a.audio)
    audio, segments = load_audio(path), whisper_segments(path)
    wins = windows(segments, len(audio) / SR)
    E = embed(audio, wins)
    c, members = dominant_cluster(E, a.cluster_thr)
    sims = E @ c
    per_seg = []
    for i, s in enumerate(segments):
        w = [float(x) for (k, _, _), x in zip(wins, sims) if k == i]
        if w:
            per_seg.append({"start": round(s["start"], 1), "end": round(s["end"], 1),
                            "sim": round(float(np.mean(w)), 3), "text": s["text"]})
    REFS.mkdir(exist_ok=True)
    np.save(REFS / f"{a.name}.npy", c)
    meta = {"name": a.name, "source": path.name, "audio_seconds": round(len(audio) / SR, 1),
            "windows_total": len(wins), "windows_used": int(members.sum()), "cluster_thr": a.cluster_thr,
            "member_sim_min": round(float(sims[members].min()), 3),
            "member_sim_median": round(float(np.median(sims[members])), 3),
            "outlier_windows": [[round(s, 1), round(e, 1), round(float(x), 3)]
                                for (_, s, e), x, m in zip(wins, sims, members) if not m],
            "segments": per_seg}
    (REFS / f"{a.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "segments"}, ensure_ascii=False))
    for s in per_seg:
        print(f"  {s['start']:6.1f}-{s['end']:6.1f}  sim={s['sim']:.3f}  {s['text'][:80]}")


def cmd_test(a):
    path = pathlib.Path(a.audio)
    res = score_clip(load_audio(path), whisper_segments(path), np.load(REFS / f"{a.name}.npy"), a.threshold)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_score(a):
    folder = pathlib.Path(a.dir)
    ref = np.load(REFS / f"{a.name}.npy")
    files = {json.loads(l)["_id"]: json.loads(l).get("file")
             for l in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l}
    recs = [json.loads(l) for l in (folder / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l]
    out_path = folder / "speakers.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for r in recs:
            if r["transcript_status"] not in a.status or not r.get("segments"):
                continue
            res = score_clip(load_audio(folder / files[r["_id"]]), r["segments"], ref, a.threshold)
            res = {"_id": r["_id"], "speaker": a.name, "threshold": a.threshold} | res
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"{r['_id']:22} {res['verdict']:12} target={res['target_seconds']:6.1f}s "
                  f"speech={res['speech_seconds']:6.1f}s voices=" +
                  " ".join(f"{v['sim']:.2f}/{v['seconds']:.0f}s" for v in res.get("voices", []) if v["seconds"] >= 3) +
                  f"  {(r.get('transcript') or '')[:40]!r}")
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("enroll", "test", "score"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        if name == "score":
            p.add_argument("--dir", required=True)
            p.add_argument("--status", nargs="+", default=["done"])
        else:
            p.add_argument("--audio", required=True)
        if name == "enroll":
            p.add_argument("--cluster-thr", type=float, default=0.5)
        else:
            # One cut-off, no grey zone. Set so that misattributing a quote costs more than missing one.
            p.add_argument("--threshold", type=float, default=0.45, help="voice centroid sim at or above: target speaker")
    a = ap.parse_args()
    {"enroll": cmd_enroll, "test": cmd_test, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
