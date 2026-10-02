"""Check whether a known speaker (e.g. Prabowo) is the one talking in each video.

Uses SpeechBrain ECAPA-TDNN speaker embeddings. Audio is cut into 3 s windows
(1.5 s hop) inside the speech segments Whisper found, and every window is
compared by cosine similarity to a reference voiceprint.

--model ecapa2 swaps in ECAPA2 (Jenthe/ECAPA2, TorchScript, 192-dim) for experiments. Its weights are
CC-BY-NC-4.0 (non-commercial only), it is about 35x slower on CPU (~2.9 s per 3 s window), and the
thresholds below were calibrated on ECAPA, not ECAPA2. Voiceprints are kept per model:
refs/<name>.npy for ECAPA, refs/<name>_ecapa2.npy for ECAPA2; the two are not interchangeable.

Runs inside the speaker-id:local container (see Dockerfile). Subcommands:
  enroll --audio /data/ref.mp3 --name prabowo   build a voiceprint from a clean reference clip
  test   --audio /data/clip.mp3 --name prabowo  score one clip
  score  --dir /data/<batch> --name prabowo     score transcribed videos, write speakers.jsonl
All three take --model ecapa|ecapa2 (default ecapa).
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np
import requests
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from speechbrain.inference.speaker import EncoderClassifier

SR = 16000
WIN, HOP, MIN_LEN = 3.0, 1.5, 1.0  # seconds
REFS = pathlib.Path("/app/refs")
SPEACHES_URL = "http://host.docker.internal:8010/v1/audio/transcriptions"
WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Phrases Whisper emits on music or silence; such segments are not speech (same list as transcribe.py).
HALLUCINATIONS = {"terima kasih", "terima kasih telah menonton", "sampai jumpa", "thank you",
                  "thanks for watching", "you"}
# Segments with no main voice (under 1 s, or in a voice with under 3 s in total) are matched to the
# clip's main voices instead: picking one of 2-3 known voices works on 0.5-1.5 s of audio, clustering
# it from scratch does not. Simulated on batch 20260929 with pieces cut from known voices: every match
# went to the right voice, and a voice from another clip was accepted for 0% / 3% / 4% of
# 0.5 / 1.0 / 1.5 s pieces.
MATCH_THR, MATCH_MARGIN, MATCH_MIN = 0.45, 0.10, 0.3  # cosine, lead over the runner-up, seconds

ECAPA2_REPO, ECAPA2_REVISION = "Jenthe/ECAPA2", "207cb6d137c671a12ba820ebec3b719549b06c0f"
MODEL = "ecapa"  # set from --model
_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        if MODEL == "ecapa2":
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(ECAPA2_REPO, "ecapa2.pt", revision=ECAPA2_REVISION)
            _encoder = torch.jit.load(path, map_location="cpu")
        else:
            _encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                      savedir="/cache/models/ecapa", run_opts={"device": "cpu"})
    return _encoder


def ref_path(name):
    return REFS / (f"{name}.npy" if MODEL == "ecapa" else f"{name}_{MODEL}.npy")


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
    if MODEL == "ecapa2":
        return embed_ecapa2(audio, wins)
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


def embed_ecapa2(audio, wins, batch=8):
    """ECAPA2 takes no lengths, and zero padding shifts its embedding (cosine 0.81 to the unpadded
    one in a test on noise), so windows are batched only with windows of the same length."""
    chunks = [audio[int(s * SR):int(e * SR)] for _, s, e in wins]
    by_len = {}
    for j, c in enumerate(chunks):
        by_len.setdefault(len(c), []).append(j)
    e = np.zeros((len(chunks), 192), dtype=np.float32)
    with torch.no_grad():
        for idx in by_len.values():
            for b in range(0, len(idx), batch):
                part = idx[b:b + batch]
                wavs = torch.from_numpy(np.stack([chunks[j] for j in part]).copy())
                e[part] = encoder()(wavs).numpy()
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
    """Average-linkage diarization: two groups of windows are one voice while the mean cosine
    over all window pairs between them is at least thr. Voice 0 is the one with most windows.

    Cosine to a group centroid is not used as the test: the centroid of two mixed voices stays
    close to both (0.67 against a pairwise 0.25 in a moderator + candidate debate clip), so it
    let a second speaker into the first speaker's voice."""
    if len(E) < 2:
        return np.zeros(len(E), dtype=int)
    labels = fcluster(linkage(E, method="average", metric="cosine"), t=1 - thr, criterion="distance")
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return np.argsort(np.argsort(-counts, kind="stable"))[inv]


def match_short(audio, segments, segs, E, labels, voices, ref):
    """Add match_voice / match_sim to segments without a main voice when one main voice is clearly
    closest. voice, label and every duration stay as clustered, so a matched segment never counts
    toward the target speaker's time, the quotes, or the number of voices."""
    main = [v["voice"] for v in voices if v["seconds"] >= 3.0]
    if not main:
        return segs
    cents = {k: E[labels == k].mean(0) for k in main}
    cents = {k: c / np.linalg.norm(c) for k, c in cents.items()}
    audio_seconds = len(audio) / SR
    known = {(s["start"], s["end"], s["text"]): s for s in segs}
    todo = []
    for s in segments:
        if s.get("hallucination") or s["text"].lower().strip(" .!?") in HALLUCINATIONS:
            continue
        key = (round(s["start"], 1), round(s["end"], 1), s["text"])
        seg = known.get(key)
        start, end = s["start"], min(s["end"], audio_seconds)
        if (seg and seg["voice"] in main) or end - start < MATCH_MIN:
            continue
        todo.append((seg or {"start": key[0], "end": key[1], "text": s["text"], "voice": None, "label": None},
                     start, end))
    if not todo:
        return segs
    for (seg, _, _), e in zip(todo, embed(audio, [(0, a, b) for _, a, b in todo])):
        sims = sorted(((float(e @ c), k) for k, c in cents.items()), reverse=True)
        best, k = sims[0]
        if best >= MATCH_THR and best - (sims[1][0] if len(sims) > 1 else -1.0) >= MATCH_MARGIN:
            seg["match_voice"], seg["match_sim"] = k, round(best, 3)
            if seg["voice"] is None:
                seg["sim"] = round(float(e @ ref), 3)
                segs.append(seg)
    return sorted(segs, key=lambda s: s["start"])


def score_clip(audio, segments, ref, threshold, cluster_thr=0.35):
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
    segs = match_short(audio, segments, segs, E, labels, voices, ref)
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
    np.save(ref_path(a.name), c)
    meta = {"name": a.name, "model": MODEL, "source": path.name, "audio_seconds": round(len(audio) / SR, 1),
            "windows_total": len(wins), "windows_used": int(members.sum()), "cluster_thr": a.cluster_thr,
            "member_sim_min": round(float(sims[members].min()), 3),
            "member_sim_median": round(float(np.median(sims[members])), 3),
            "outlier_windows": [[round(s, 1), round(e, 1), round(float(x), 3)]
                                for (_, s, e), x, m in zip(wins, sims, members) if not m],
            "segments": per_seg}
    ref_path(a.name).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "segments"}, ensure_ascii=False))
    for s in per_seg:
        print(f"  {s['start']:6.1f}-{s['end']:6.1f}  sim={s['sim']:.3f}  {s['text'][:80]}")


def cmd_test(a):
    path = pathlib.Path(a.audio)
    res = score_clip(load_audio(path), whisper_segments(path), np.load(ref_path(a.name)), a.threshold)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_score(a):
    folder = pathlib.Path(a.dir)
    ref = np.load(ref_path(a.name))
    files = {json.loads(l)["_id"]: json.loads(l).get("file")
             for l in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l}
    recs = [json.loads(l) for l in (folder / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l]
    out_path = folder / "speakers.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for r in recs:
            if r["transcript_status"] not in a.status or not r.get("segments"):
                continue
            res = score_clip(load_audio(folder / files[r["_id"]]), r["segments"], ref, a.threshold)
            res = {"_id": r["_id"], "speaker": a.name, "model": MODEL, "threshold": a.threshold} | res
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
        p.add_argument("--model", choices=["ecapa", "ecapa2"], default="ecapa")
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
    global MODEL
    MODEL = a.model
    {"enroll": cmd_enroll, "test": cmd_test, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
