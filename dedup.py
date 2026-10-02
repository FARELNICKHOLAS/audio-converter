"""Find videos in a batch that carry the same audio (re-uploads, reposts, trimmed copies).

Audio is compared with Chromaprint fingerprints (the AcoustID fingerprint, via ffmpeg's
chromaprint muxer): one 32-bit value per ~0.12 s, so two copies of the same recording match
bit for bit even when re-encoded, while unrelated audio differs in about half the bits.
Every pair is aligned at its best time offset, so a copy with a trimmed or added intro still
matches. Transcript similarity is reported next to it as a second signal.

Writes duplicates.json in the batch folder: groups of _ids sharing audio, the earliest post
first ("keep"). Needs ffmpeg built with chromaprint, e.g. the speaker-id:local image:
  docker run --rm -v "C:\\Users\\wayan\\Downloads:/data" -v "D:\\audio-converter:/src:ro" speaker-id:local
         python /src/dedup.py --dir /data/socmed-prabowo-mbg-20260929
"""
import argparse
import difflib
import json
import pathlib
import re
import subprocess

import numpy as np

ITEM = 4096 / 3 / 11025  # seconds per fingerprint value (chromaprint default algorithm)
MAX_BER = 0.20           # bit error rate at or below: same audio (unrelated audio is ~0.5)
MIN_OVERLAP = 5.0        # seconds of aligned audio needed before a match counts


def fingerprint(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-f", "chromaprint", "-fp_format", "raw", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<u4")


def bits(fp):
    """(32, n) array of +1/-1, one row per fingerprint bit."""
    return np.unpackbits(fp.view(np.uint8).reshape(-1, 4), axis=1, bitorder="little").T.astype(np.float32) * 2 - 1


def best_alignment(a, b):
    """(bit error rate, offset of b in a in seconds, overlap seconds) at the offset with the
    lowest bit error rate among offsets that overlap at least MIN_OVERLAP."""
    A, B = bits(a), bits(b)
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    # sum over bits of the cross-correlation of the +1/-1 sequences = agreeing bits - differing bits
    corr = np.fft.irfft((np.fft.rfft(A, size) * np.conj(np.fft.rfft(B, size))).sum(0), size)
    lags = np.arange(-(len(b) - 1), len(a))
    corr = np.concatenate([corr[size - (len(b) - 1):], corr[:len(a)]]) if len(b) > 1 else corr[:len(a)]
    overlap = np.minimum(len(a), lags + len(b)) - np.maximum(0, lags)
    ok = overlap * ITEM >= MIN_OVERLAP
    if not ok.any():
        return None
    ber = np.where(ok, (1 - corr / (32 * np.maximum(overlap, 1))) / 2, 1.0)
    i = int(ber.argmin())
    return float(ber[i]), lags[i] * ITEM, overlap[i] * ITEM


def words(text):
    return re.findall(r"\w+", (text or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    folder = pathlib.Path(a.dir)
    docs = {}
    for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line and (d := json.loads(line)).get("file"):
            docs[d["_id"]] = d
    texts = {}
    if (folder / "transcripts.jsonl").exists():
        for line in (folder / "transcripts.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                texts[r["_id"]] = r.get("transcript") or ""
    fps = {}
    for i, d in docs.items():
        try:
            fp = fingerprint(folder / d["file"])
        except subprocess.CalledProcessError:
            continue  # no audio stream
        if len(np.unique(fp)) > 10:  # silence or a constant tone gives a flat fingerprint that matches anything flat
            fps[i] = fp
    ids = sorted(fps, key=lambda i: docs[i].get("created_at") or "")
    pairs = []
    for x, i in enumerate(ids):
        for j in ids[x + 1:]:
            m = best_alignment(fps[i], fps[j])
            if m and m[0] <= MAX_BER:
                ratio = difflib.SequenceMatcher(None, words(texts.get(i)), words(texts.get(j))).ratio()
                pairs.append({"a": i, "b": j, "ber": round(m[0], 3), "offset_seconds": round(m[1], 1),
                              "overlap_seconds": round(m[2], 1), "transcript_ratio": round(ratio, 3)})
    group = {i: i for i in ids}  # union-find over matching pairs
    def root(i):
        while group[i] != i:
            i = group[i]
        return i
    for p in pairs:
        ra, rb = root(p["a"]), root(p["b"])
        if ra != rb:
            group[max(ra, rb, key=ids.index)] = min(ra, rb, key=ids.index)
    groups = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)
    out = {"max_ber": MAX_BER, "min_overlap_seconds": MIN_OVERLAP, "fingerprinted": len(fps),
           "groups": [{"keep": g[0], "duplicates": g[1:]} for g in groups.values() if len(g) > 1],
           "pairs": pairs}
    (folder / "duplicates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(fps)} fingerprinted, {len(pairs)} matching pairs, {len(out['groups'])} groups")
    for p in pairs:
        print(f"  {p['a']:24} {p['b']:24} ber={p['ber']:.3f} offset={p['offset_seconds']:+.1f}s "
              f"overlap={p['overlap_seconds']:.1f}s transcript={p['transcript_ratio']:.2f}")
    print(f"-> {folder / 'duplicates.json'}")


if __name__ == "__main__":
    main()
