"""Download socmed videos found by an OpenSearch query.

Read-only against OpenSearch: runs one _search, then downloads each hit's
media_url. Falls back to yt-dlp on the post link when media_url is missing or
fails (e.g. expired CDN URL, YouTube). Writes a manifest.jsonl next to the files.

Usage:
  python fetch_videos.py --query queries/query_prabowo.json --size 30 --out downloads \
      --netrc-file path/to/os.netrc
Credentials: --netrc-file, or env OS_USER / OS_PASS.
"""
import argparse
import concurrent.futures as cf
import json
import netrc
import os
import pathlib
import time
from urllib.parse import urlparse

import requests

DEFAULT_URL = "https://osearch.prod.int.edwi.co.id"
SOURCE_FIELDS = ["id", "platform", "media_type", "media_url", "link", "content", "created_at", "lang"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
REFERERS = {"tiktok": "https://www.tiktok.com/", "instagram": "https://www.instagram.com/",
            "threads": "https://www.threads.net/", "facebook": "https://www.facebook.com/",
            "twitter": "https://x.com/"}


def credentials(os_url, netrc_file):
    user, pwd = os.environ.get("OS_USER"), os.environ.get("OS_PASS")
    if netrc_file:
        auth = netrc.netrc(netrc_file).authenticators(urlparse(os_url).hostname)
        if auth:
            user, pwd = auth[0], auth[2]
    if not (user and pwd):
        raise SystemExit("No OpenSearch credentials: use --netrc-file or OS_USER/OS_PASS")
    return user, pwd


def search(os_url, index, body, size, auth):
    body = {**body, "size": size, "_source": SOURCE_FIELDS}
    r = requests.post(f"{os_url}/{index}/_search", json=body, auth=auth, timeout=60)
    r.raise_for_status()
    return [h["_source"] | {"_id": h["_id"], "_index": h["_index"]} for h in r.json()["hits"]["hits"]]


def pick_video_url(urls):
    urls = list(dict.fromkeys(urls or []))  # dedupe, keep order
    return next((u for u in urls if ".mp4" in u), urls[0] if urls else None)


def download_direct(url, dest, platform):
    headers = {"User-Agent": UA, "Referer": REFERERS.get(platform, "")}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 120)) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if not ctype.startswith(("video/", "application/octet-stream", "binary/")):
            raise ValueError(f"not a video (Content-Type: {ctype})")
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        tmp.replace(dest)


def download_ytdlp(link, out_dir, stem):
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed (pip install yt-dlp)")
    # Audio-only m4a as last resort: some YouTube Shorts offer no combined audio+video format,
    # and merging separate streams needs ffmpeg on the host. The pipeline only needs the audio.
    opts = {"format": "b[height<=720]/b/ba[ext=m4a]/ba", "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "js_runtimes": {"node": {}}}  # YouTube extraction needs a JS runtime; Node is installed
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(link, download=True)
        return pathlib.Path(ydl.prepare_filename(info))


def fetch(doc, out_dir):
    platform = doc.get("platform", "unknown")
    stem = f"{platform}_{doc['_id']}"
    rec = {"_id": doc["_id"], "_index": doc["_index"], "platform": platform,
           "link": doc.get("link"), "created_at": doc.get("created_at"), "lang": doc.get("lang"),
           "content": (doc.get("content") or "")[:300]}
    existing = next(out_dir.glob(f"{stem}.*"), None)
    if existing and existing.suffix != ".part":
        return rec | {"status": "exists", "file": existing.name, "bytes": existing.stat().st_size}

    t0, errors = time.time(), []
    url = pick_video_url(doc.get("media_url"))
    if url:
        dest = out_dir / f"{stem}.mp4"
        try:
            download_direct(url, dest, platform)
            return rec | {"status": "ok", "method": "media_url", "file": dest.name,
                          "bytes": dest.stat().st_size, "seconds": round(time.time() - t0, 1)}
        except Exception as e:
            errors.append(f"media_url: {e}")
    if doc.get("link"):
        try:
            path = download_ytdlp(doc["link"], out_dir, stem)
            return rec | {"status": "ok", "method": "yt-dlp", "file": path.name,
                          "bytes": path.stat().st_size, "seconds": round(time.time() - t0, 1)}
        except Exception as e:
            errors.append(f"yt-dlp: {e}")
    return rec | {"status": "failed", "error": " | ".join(errors) or "no media_url and no link"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="JSON file with the search body (query part)")
    ap.add_argument("--index", default="socmed-content*")
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--out", default="downloads")
    ap.add_argument("--os-url", default=os.environ.get("OS_URL", DEFAULT_URL))
    ap.add_argument("--netrc-file")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    auth = credentials(a.os_url, a.netrc_file)
    body = json.loads(pathlib.Path(a.query).read_text(encoding="utf-8"))
    docs = search(a.os_url, a.index, body, a.size, auth)
    print(f"{len(docs)} hits")

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as pool, \
            open(out_dir / "manifest.jsonl", "a", encoding="utf-8") as manifest:
        for rec in pool.map(lambda d: fetch(d, out_dir), docs):
            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            size = f"{rec.get('bytes', 0) / 1e6:.1f}MB" if rec.get("bytes") else ""
            print(f"[{rec['status']:6}] {rec['platform']:9} {rec['_id']:<24} {size:>8} {rec.get('error', '')[:120]}")

    ok = sum(1 for _ in out_dir.glob("*.mp4"))
    print(f"done in {time.time() - t0:.1f}s, {ok} mp4 files in {out_dir}")


if __name__ == "__main__":
    main()
