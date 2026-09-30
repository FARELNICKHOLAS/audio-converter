"""Export transcripts.jsonl to a CSV that opens cleanly in Excel.

Uses ';' as delimiter, ',' as decimal separator and a UTF-8 BOM, matching
Excel on Indonesian-locale Windows.

Usage:
  python export_csv.py --dir "C:/Users/wayan/Downloads/socmed-prabowo-mbg-20260929" \
      --out "C:/Users/wayan/Downloads/transkrip-prabowo-mbg-20260929.csv"
"""
import argparse
import csv
import json
import pathlib
import re

STATUS_ORDER = {"done": 0, "suspect_hallucination": 1, "no_speech": 2, "no_audio": 3, "failed": 4}
COLUMNS = ["no", "status", "catatan", "platform", "created_at", "durasi_audio_detik", "waktu_proses_detik",
           "bahasa_transkrip", "jumlah_kata", "sebut_mbg_lisan", "sebut_prabowo_lisan",
           "caption", "transkrip", "link", "file", "_id", "_index"]


def dec(x):
    return f"{x:.1f}".replace(".", ",") if isinstance(x, (int, float)) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folder = pathlib.Path(a.dir)
    files = {json.loads(l)["_id"]: json.loads(l).get("file")
             for l in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l}
    recs = [json.loads(l) for l in (folder / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l]
    recs.sort(key=lambda r: (STATUS_ORDER.get(r["transcript_status"], 9), r.get("created_at") or ""))

    first_seen = {}
    rows = []
    for i, r in enumerate(recs, 1):
        text = r.get("transcript") or ""
        words = len(text.split())
        notes = []
        if r["transcript_status"] == "done":
            if words <= 3:
                notes.append("transkrip sangat pendek, cek manual")
            if r.get("transcript_lang") != r.get("lang"):
                notes.append(f"bahasa beda dari caption ({r.get('lang')})")
            key = text[:120]
            if key in first_seen:
                notes.append(f"duplikat isi dengan _id {first_seen[key]}")
            else:
                first_seen[key] = r["_id"]
        rows.append({
            "no": i, "status": r["transcript_status"], "catatan": "; ".join(notes),
            "platform": r.get("platform"), "created_at": r.get("created_at"),
            "durasi_audio_detik": dec(r.get("transcript_duration")),
            "waktu_proses_detik": dec(r.get("process_seconds")),
            "bahasa_transkrip": r.get("transcript_lang") or "", "jumlah_kata": words,
            "sebut_mbg_lisan": "ya" if re.search(r"\bMBG\b|makan bergizi", text, re.I) else "tidak",
            "sebut_prabowo_lisan": "ya" if re.search(r"prabowo|presiden", text, re.I) else "tidak",
            "caption": (r.get("content") or "").replace("\r", " "), "transkrip": text,
            "link": r.get("link"), "file": files.get(r["_id"]), "_id": r["_id"], "_index": r.get("_index"),
        })

    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
