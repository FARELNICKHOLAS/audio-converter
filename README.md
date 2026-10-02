# audio-converter

Pipeline untuk mengubah video socmed dari OpenSearch menjadi transkrip, mengecek apakah
Prabowo sendiri yang berbicara, lalu mengekspor hasilnya ke Excel.

```
fetch_videos.py   OpenSearch (read-only) -> unduh video -> manifest.jsonl
prep/vad.py       video -> Silero VAD -> vad.jsonl                           (di Docker)
transcribe.py     video -> Speaches (Whisper) -> transcripts.jsonl          (lewati video tanpa ucapan)
speaker_id.py     video + segmen -> voiceprint ECAPA -> speakers.jsonl   (di Docker)
export_xlsx.py    semua .jsonl -> Excel (sheet Transkrip + Ringkasan)
```

## Isi folder

| Path | Fungsi |
|---|---|
| `fetch_videos.py` | Menjalankan satu `_search` ke OpenSearch, mengunduh `media_url` tiap hasil (fallback yt-dlp), menulis `manifest.jsonl` |
| `transcribe.py` | Transkripsi lewat Speaches, gerbang Silero VAD, filter halusinasi Whisper, fallback tanpa prompt bila prompt bocor |
| `export_xlsx.py` | Ekspor ke Excel, termasuk kolom voiceprint bila ada `speakers.jsonl` |
| `export_csv.py` | Ekspor CSV lama (tanpa kolom voiceprint) |
| `dedup.py` | Opsional: cari video dengan audio sama (fingerprint Chromaprint), menulis `duplicates.json`. Hanya informasi, tidak dipakai export |
| `queries/` | Body query OpenSearch (`query_prabowo.json`, `query_prabowo_mbg.json`) |
| `prep/` | Dockerfile `audio-prep:local` + `vad.py` (gerbang Silero VAD) + `prep.py` (eksperimen Demucs) |
| `speaker-id/` | Dockerfile + `speaker_id.py` (enroll / test / score) |
| `speaker-id/refs/` | Voiceprint referensi (`prabowo.npy`) dan metadatanya |
| `speaches/docker-compose.yml` | Service STT Speaches di port 8010 |

## Setup (sekali)

1. Docker Desktop berjalan.
2. Paket Python di host:
   ```powershell
   pip install -r D:\audio-converter\requirements.txt
   ```
3. Service STT (`restart: unless-stopped`):
   ```powershell
   docker compose -f D:\audio-converter\speaches\docker-compose.yml up -d
   ```
   Bila container pernah dihentikan manual, nyalakan lagi dengan `docker start speaches`.
   Cek dengan `docker ps --filter name=speaches`.
4. Image speaker-id (sudah ada sebagai `speaker-id:local`; build ulang hanya bila Dockerfile berubah):
   ```powershell
   docker build -t speaker-id:local D:\audio-converter\speaker-id
   ```
   Model ECAPA diunduh sekali ke volume `spk-cache` saat pertama dipakai.
5. Image audio-prep (Silero VAD, dibangun di atas `speaker-id:local`; sudah ada sebagai `audio-prep:local`):
   ```powershell
   docker build -t audio-prep:local D:\audio-converter\prep
   ```
6. Kredensial OpenSearch: file netrc di luar folder ini, isinya
   `machine osearch.prod.int.edwi.co.id login <user> password <password>`.
   Alternatif: env `OS_USER` dan `OS_PASS`. Jangan simpan kredensial di folder ini.

## Menjalankan satu batch

Contoh batch `socmed-prabowo-mbg-20260929` di folder Downloads. Jalankan dari PowerShell
(di Git Bash path Docker bisa rusak).

```powershell
cd D:\audio-converter
$B = "C:\Users\wayan\Downloads\socmed-prabowo-mbg-20260929"
$env:PYTHONIOENCODING = "utf-8"

# 1. Ambil dan unduh video (ubah tanggal/kata kunci di file query dulu)
python fetch_videos.py --query queries\query_prabowo_mbg.json --size 30 --out $B --netrc-file C:\path\ke\os.netrc

# 2. Silero VAD: tulis vad.jsonl di folder batch (sekitar 20 detik untuk 37 video)
docker run --rm -v "C:\Users\wayan\Downloads:/data" -v "D:\audio-converter\prep:/prep:ro" -v spk-cache:/cache audio-prep:local python /prep/vad.py --src /data/socmed-prabowo-mbg-20260929

# 3. Transkripsi (sekitar 2x real-time di CPU); video dengan ucapan Silero < 1 detik tidak dikirim ke Whisper
python transcribe.py --dir $B

# 4. Cek suara Prabowo
docker run --rm -e PYTHONIOENCODING=utf-8 -v "C:\Users\wayan\Downloads:/data" -v "D:\audio-converter\speaker-id:/app" -v spk-cache:/cache speaker-id:local python speaker_id.py score --name prabowo --dir /data/socmed-prabowo-mbg-20260929

# 5. Ekspor Excel. --netrc-file hanya perlu untuk caption lengkap yang belum ada di captions.json
python export_xlsx.py --dir $B --out "C:\Users\wayan\Downloads\transkrip-prabowo-mbg-20260929.xlsx" --netrc-file C:\path\ke\os.netrc --ref-source "PRESIDEN PRABOWO Kadang-kadang Saya Kalau Bicara di Audiens Suka Dipelintir.mp3"
```

Caption lengkap disimpan di `captions.json` saat export pertama. Export ulang batch yang sama
cukup tanpa `--netrc-file`, jadi tidak menghubungi OpenSearch lagi. Tanpa `--netrc-file` dan
tanpa `captions.json`, Excel memakai caption 300 karakter dari `manifest.jsonl`.

Opsional, cek video dengan audio sama (setelah langkah 3, karena kemiripan transkrip ikut dihitung):

```powershell
docker run --rm -v "C:\Users\wayan\Downloads:/data" -v "D:\audio-converter:/src:ro" speaker-id:local python /src/dedup.py --dir /data/socmed-prabowo-mbg-20260929
```

`duplicates.json` tidak dibaca `export_xlsx.py`. Konten sama dari akun berbeda tetap dihitung
sebagai video sendiri-sendiri.

Rumus di Excel belum punya nilai tersimpan sampai file dibuka di Excel. Buka sekali lalu
simpan, atau tekan Ctrl+Alt+F9.

Opsi yang sering dipakai:

- `transcribe.py --retry-failed`: ulangi video yang gagal.
- `transcribe.py --reclassify`: hitung ulang filter halusinasi dan status dari segmen tersimpan,
  tanpa transkripsi ulang (membuat `transcripts.jsonl.bak` dulu).
- `transcribe.py --prompt ""`: transkripsi tanpa prompt ejaan.
- `transcribe.py --no-vad-gate`: abaikan `vad.jsonl`, semua video beraudio dikirim ke Whisper.
- `export_xlsx.py --note "<_id>=teks"` dan `--summary-note "teks"`: catatan manual di Excel.

## Menambah voiceprint tokoh lain

Taruh klip bersih (hanya suara tokoh itu, 1 menit atau lebih) di folder yang di-mount ke `/data`:

```powershell
docker run --rm -v "C:\Users\wayan\Downloads:/data" -v "D:\audio-converter\speaker-id:/app" -v spk-cache:/cache speaker-id:local python speaker_id.py enroll --audio /data/klip.mp3 --name nama_tokoh
```

Cek klip lain dengan `speaker_id.py test --audio /data/klip2.mp4 --name nama_tokoh`.

Eksperimen: `--model ecapa2` (enroll / test / score) memakai ECAPA2 (`Jenthe/ECAPA2`, lisensi CC-BY-NC-4.0, hanya non-komersial). Voiceprint disimpan terpisah di `refs/<nama>_ecapa2.npy`. Ambang sama dengan ECAPA tetapi belum dikalibrasi untuk ECAPA2. Di CPU sekitar 2,5 detik per jendela 3 detik (ECAPA 0,08 detik).

## Hasil per batch

Di folder batch:

- `manifest.jsonl`: hasil query dan status unduhan per `_id`.
- `vad.jsonl`: hasil Silero VAD per video (`vad_speech_seconds`, `vad_spans`).
- `transcripts.jsonl`: transkrip, segmen bertimestamp, dan status.
  Status: `done` / `no_speech` / `suspect_hallucination` / `no_audio` / `failed`.
  `no_speech` dengan `vad_gate: true` berarti Whisper tidak dipanggil. `vad_speech_seconds`
  ada di setiap video yang tercatat di `vad.jsonl`.
- `speakers.jsonl`: per video, berisi `verdict` ya/tidak, durasi suara Prabowo, suara yang
  terdeteksi, dan segmen yang diatribusikan.
- `captions.json`: cache caption lengkap dari OpenSearch (ditulis oleh `export_xlsx.py`).
- `translations.json`: opsional, terjemahan manual `{_id: teks}`; bila ada, Excel mendapat kolom
  terjemahan.
- `duplicates.json`: opsional, hasil `dedup.py`.

## Catatan penting

- **OpenSearch prod hanya dibaca.** Skrip hanya memakai `_search` dan `_mget`, tidak ada tulis
  atau hapus.
- **Ambang voiceprint 0,45, biner.** Suara dengan skor di atas 0,45 dianggap Prabowo, di
  bawahnya bukan. Tidak ada zona abu-abu.
  - Kalibrasi batch 20260929: suara lain 0,07–0,29, Prabowo 0,55–0,82.
  - Sampel kecil, jadi ambang perlu divalidasi dengan data berlabel.
- **Voiceprint tidak bisa mendeteksi suara tiruan atau AI.** Klip parodi mendapat skor 0,57.
  Jadi `pembicara` berarti suaranya mirip Prabowo, bukan jaminan rekamannya asli.
- **Voiceprint adalah data biometrik (UU PDP).** Simpan `speaker-id/refs/` dengan hati-hati.
- **Prompt ejaan** (MBG, BGN, SPPG) memperbaiki ejaan istilah, tetapi bisa bocor ke hasil.
  Bila ada pengulangan atau salinan prompt, `transcribe.py` otomatis mentranskrip ulang video
  itu tanpa prompt (`prompt_fallback: true`).
- **Gerbang Silero VAD.** Video dengan ucapan Silero < 1 detik langsung `no_speech`. Audio yang
  dikirim ke Whisper tidak diubah (tanpa Demucs, tanpa pembisuan).
  - Batch 20260929: 10 dari 33 video beraudio dilewati (ucapan 0,0–0,8 detik). Tanpa gerbang,
    6 di antaranya `no_speech` dan 4 `suspect_hallucination` berisi "Terima kasih.". 23 video
    lain teksnya identik. Video terendah berikutnya punya 2,0 detik ucapan.
  - Varian lain yang diuji (Demucs, Demucs + Silero, Silero membisukan audio) mengubah teks
    dan bisa meloloskan halusinasi baru, jadi tidak dipakai. Lihat bagian Eksperimen di bawah.
  - Tanpa `vad.jsonl` (langkah 2 dilewati), `transcribe.py` berjalan seperti sebelumnya.
- **Jumlah suara adalah perkiraan.** Jendela 3 detik dikelompokkan dengan average linkage:
  dua kelompok jadi satu suara bila rata-rata cosine semua pasangan jendelanya >= 0,35.
  - Uji di klip berlabel: pasangan jendela dari orang berbeda 0,15–0,32, potongan suara
    Prabowo di video berisik 0,34–0,39. Ambang 0,40 sudah memecah suara Prabowo.
  - Dulu dipakai cosine ke centroid kelompok. Centroid dua suara campuran tetap dekat ke
    keduanya (0,67, padahal antar-pasangan 0,25), jadi orang kedua ikut tergabung.
  - Masih bisa kurang: dua suara yang sangat mirip tetap tergabung, dan satu segmen Whisper
    berisi dua orang hanya mendapat satu suara.
  - Suara yang bicara kurang dari 3 detik tidak dihitung. Kolom Catatan di Excel menandai
    video yang sebagian besar ucapannya berupa giliran pendek.
- **Selaan pendek dicocokkan ke pembicara utama.** Segmen < 1 detik, atau dari suara < 3 detik,
  dibandingkan utuh dengan rata-rata suara tiap pembicara utama (>= 3 detik) di video itu.
  Segmen diberi `match_voice` bila cosine >= 0,45 dan unggul >= 0,10 dari pembicara kedua.
  Di Excel tampil sebagai `Pembicara N*`.
  - Simulasi dengan potongan 0,5 / 1,0 / 1,5 detik dari pembicara yang sudah diketahui:
    yang dicocokkan 31% / 69% / 83%, dan semuanya ke pembicara yang benar. Suara dari video
    lain diterima 0% / 3% / 4%. "Benar" di sini artinya sama dengan label pipeline sendiri,
    bukan label manusia.
  - `voice`, `label`, durasi, jumlah suara, dan kutipan Prabowo tidak berubah: segmen yang
    dicocokkan tidak pernah dihitung sebagai suara Prabowo.

## Eksperimen pra-proses audio (2026-10-01)

Batch 20260929 (37 video, 33 beraudio) ditranskrip ulang dengan lima varian. Model Whisper,
prompt, dan filter halusinasi sama di semua varian.

| Varian | Audio ke Whisper | done / suspect / no_speech | Halu dibuang filter | Halu lolos filter | Whisper | Pra-proses |
|---|---|---|---|---|---|---|
| A | Audio asli (pipeline lama) | 21 / 6 / 6 | 25 | 3 (1 video) | 646 s | 0 s |
| B | Vokal Demucs | 22 / 4 / 7 | 56 | 5 (4 video) | 659 s | 800 s |
| C | Vokal Demucs, dibisukan di luar ucapan Silero | 24 / 1 / 8 | 13 | 5 (5 video) | 585 s | 880 s |
| D | Audio asli, dibisukan di luar ucapan Silero | 19 / 5 / 9 | 29 | 7 (2 video) | 558 s | 19 s |
| **E** | Audio asli; ucapan Silero < 1 detik tidak dikirim | 21 / 2 / 10 | 21 | 3 (1 video) | 584 s | 19 s |

- "Halu dibuang filter" dan "Halu lolos filter" dihitung per segmen. "Lolos" dihitung oleh
  skrip perbandingan dengan tiga aturan tambahan yang tidak ada di `transcribe.py`:
  - segmen melewati akhir audio lebih dari 1 detik;
  - teks sama di 3 segmen berturut-turut atau lebih;
  - segmen 1–4 kata yang semuanya kata prompt.
- Di semua varian: 4 video `no_audio` dan 4 video Prabowo `ya`.
- E dipakai di pipeline. Teks 23 video yang tetap dikirim ke Whisper identik dengan A, dan
  halusinasi "Terima kasih." di 4 video hilang. Silero butuh 19 detik, dan Whisper selesai
  62 detik lebih cepat.
- C terlihat paling bersih (suspect 1), tetapi 5 halusinasi lolos ke transkrip, termasuk
  "selamat menikmati" di video Prabowo `ya`. Demucs juga menambah 13–15 menit per batch.
- D membuang sebagian ucapan asli: jumlah kata 3246 menjadi 3203, suara Prabowo 184,9 menjadi
  173,0 detik.

Untuk mengulang, tulis WAV per varian ke folder eksperimen (di luar folder batch):

```powershell
# B dan C: <eksperimen>/demucs dan <eksperimen>/demucs_silero
docker run --rm -v "C:\Users\wayan\Downloads:/data" -v "D:\audio-converter\prep:/prep:ro" -v spk-cache:/cache audio-prep:local python /prep/prep.py --src /data/<batch> --out /data/<eksperimen>
# D: <eksperimen>/silero
docker run --rm -v "C:\Users\wayan\Downloads:/data" -v "D:\audio-converter\prep:/prep:ro" -v spk-cache:/cache audio-prep:local python /prep/vad.py --src /data/<batch> --out /data/<eksperimen>
```

`transcribe.py` belum bisa langsung membaca folder WAV ini, karena cek track audionya hanya
mengenali kontainer MP4 (mp4/m4a). Eksperimen di atas ditranskrip dengan skrip terpisah yang
memanggil `transcript_fields` dari `transcribe.py`.
