"""Transcribe downloaded videos with the local Speaches (faster-whisper) server.

Reads manifest.jsonl written by fetch_videos.py, sends each video file to
Speaches, and writes transcripts.jsonl (one line per _id) in the same folder.
Already-transcribed _ids are skipped, so it is safe to re-run.

Segments that look like Whisper hallucinations (stock phrases, repetition loops,
copies of the prompt, timestamps past the end of the audio) are kept in "segments" with a "hallucination" reason but are
left out of "transcript"; the unfiltered text is kept in "transcript_raw".

The spelling prompt fixes words like "MBG" (without it Whisper writes "MBT", "bergiji"),
but it can also leak into the output. When a prompted result has a repetition loop or
a prompt copy, the video is transcribed again without the prompt and that result is kept
("prompt_fallback": true).

Usage:
  python transcribe.py --dir "C:/Users/wayan/Downloads/socmed-prabowo-mbg-20260929"
"""
import argparse
import json
import pathlib
import re
import time

import requests

SPEACHES_URL = "http://localhost:8010/v1/audio/transcriptions"
MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Spelling hints. On batch 20260929 this prompt made Whisper replace 19 s of real conversation
# with "Badan Gizi" repeated 12 times, hence the no-prompt fallback (see LEAKS).
PROMPT = "Prabowo Subianto, MBG, Makan Bergizi Gratis, Badan Gizi Nasional (BGN), SPPG."
PROMPT_WORDS = set(re.findall(r"\w+", PROMPT.lower()))
# Phrases Whisper emits on music or silence instead of real speech.
HALLUCINATIONS = {"terima kasih", "terima kasih.", "terima kasih telah menonton",
                  "sampai jumpa", "thank you.", "thanks for watching!", "you"}
# Hallucination reasons that mean the prompt leaked into the output.
LEAKS = {"berulang", "salinan_prompt"}


def load_manifest(folder):
    docs = {}
    for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("file"):
            docs[rec["_id"]] = rec  # last record per _id wins
    return docs


def has_audio_track(path):
    # MP4 track handler box: 'hdlr' + version/flags + pre_defined + handler type 'soun'.
    return re.search(rb"hdlr.{8}soun", path.read_bytes(), re.DOTALL) is not None


def hallucination_reason(seg, duration=None):
    """Why a segment looks like a Whisper hallucination rather than speech, or None.

    duration is the audio length in seconds; segments starting at or after it have no audio under them."""
    text = seg["text"].lower()
    words = re.findall(r"\w+", text)
    if not words:
        return "kosong"
    if duration and seg["start"] >= duration:
        return "di_luar_audio"  # e.g. a phrase looped for 11 s past the end of a 59.4 s clip
    if (text.strip(" .!?") in {h.strip(" .!") for h in HALLUCINATIONS}
            or re.fullmatch(r"(pembicara|speaker) \d+", text.strip(" .!?"))):
        return "frasa_umum"  # stock phrase, or a subtitle-style speaker label over music
    if len(words) >= 8 and len(set(words)) / len(words) < 0.4:
        return "berulang"
    if len(words) >= 5 and sum(w in PROMPT_WORDS for w in words) / len(words) >= 0.8:
        return "salinan_prompt"
    return None


def clean(rec):
    """Flag hallucinated segments and rebuild transcript from the rest."""
    for s in rec["segments"]:
        s.pop("hallucination", None)
        if reason := hallucination_reason(s, rec.get("transcript_duration")):
            s["hallucination"] = reason
    rec.setdefault("transcript_raw", rec.get("transcript", ""))
    rec["transcript"] = " ".join(s["text"] for s in rec["segments"] if "hallucination" not in s).strip()
    return rec


def classify(text, segments):
    """text is the cleaned transcript (see clean)."""
    voiced = [s for s in segments if s.get("no_speech_prob", 0) < 0.6]
    if (text or "").strip() and any("hallucination" not in s for s in voiced):
        return "done"
    if any(s.get("hallucination") not in (None, "kosong") for s in voiced):
        return "suspect_hallucination"  # Whisper produced text, but all of it was flagged
    return "no_speech"


def reclassify(out_path):
    recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    backup = out_path.with_suffix(".jsonl.bak")
    if not backup.exists():
        backup.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    for r in recs:
        if "segments" in r:
            clean(r)
            r["transcript_status"] = classify(r.get("transcript"), r["segments"])
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    return recs


def transcribe(path, lang, prompt):
    data = {"model": MODEL, "response_format": "verbose_json", "temperature": "0"}
    if prompt:
        data["prompt"] = prompt
    if lang in ("id", "en"):
        data["language"] = lang  # caption language as hint; otherwise auto-detect
    with open(path, "rb") as f:
        r = requests.post(SPEACHES_URL, files={"file": (path.name, f, "video/mp4")},
                          data=data, timeout=1800)
    r.raise_for_status()
    return r.json()


def transcript_fields(path, lang, prompt):
    res = transcribe(path, lang, prompt)
    segments = [{"start": round(s["start"], 1), "end": round(s["end"], 1),
                 "text": s["text"].strip(),
                 "no_speech_prob": round(s.get("no_speech_prob", 0), 2),
                 "compression_ratio": round(s.get("compression_ratio", 0), 2),
                 "avg_logprob": round(s.get("avg_logprob", 0), 2)}
                for s in res.get("segments") or []]
    rec = {"transcript": res.get("text", "").strip(),
           "transcript_lang": res.get("language"),
           "transcript_duration": round(res.get("duration", 0), 1),
           "transcript_prompt": prompt,
           "segments": segments}
    clean(rec)
    rec["transcript_status"] = classify(rec["transcript"], segments)
    return rec


def prompt_leaked(rec):
    return bool(rec.get("transcript_prompt")) and any(s.get("hallucination") in LEAKS
                                                      for s in rec.get("segments", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--retry-failed", action="store_true", help="re-run records with status failed")
    ap.add_argument("--reclassify", action="store_true", help="re-clean and recompute status from saved segments, no transcription")
    ap.add_argument("--prompt", default=PROMPT, help='spelling-hint prompt; "" = none')
    ap.add_argument("--out-name", default="transcripts.jsonl")
    a = ap.parse_args()
    folder = pathlib.Path(a.dir)
    out_path = folder / a.out_name
    if a.reclassify:
        recs = reclassify(out_path)
        print({s: sum(r["transcript_status"] == s for r in recs) for s in {r["transcript_status"] for r in recs}})
        return
    done = set()
    if out_path.exists():
        recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
        if a.retry_failed:
            recs = [r for r in recs if r["transcript_status"] != "failed"]
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        done = {r["_id"] for r in recs}

    docs = [d for d in load_manifest(folder).values() if d["_id"] not in done]
    print(f"{len(docs)} to transcribe ({len(done)} already done)")
    t_all = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for i, d in enumerate(docs, 1):
            t0 = time.time()
            rec = {k: d.get(k) for k in ("_id", "_index", "platform", "link", "created_at", "lang", "content")}
            try:
                if not has_audio_track(folder / d["file"]):
                    rec["transcript_status"] = "no_audio"
                else:
                    rec |= transcript_fields(folder / d["file"], d.get("lang"), a.prompt)
                    if prompt_leaked(rec):
                        rec |= transcript_fields(folder / d["file"], d.get("lang"), "")
                        rec["prompt_fallback"] = True
            except Exception as e:
                rec |= {"transcript_status": "failed", "error": str(e)[:300]}
            rec["transcript_model"] = MODEL
            rec["process_seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i:2}/{len(docs)}] {rec['transcript_status']:21} {rec['platform']:9} "
                  f"audio={rec.get('transcript_duration', 0):>6}s proc={rec['process_seconds']:>6}s "
                  f"{(rec.get('transcript') or rec.get('error', ''))[:70]!r}")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
