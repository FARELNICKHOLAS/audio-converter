# audio-converter

Pipeline untuk mengubah video socmed dari OpenSearch menjadi transkrip, mengecek apakah
Prabowo sendiri yang berbicara, lalu mengekspor hasilnya ke Excel.

```
fetch_videos.py   OpenSearch (read-only) -> unduh video -> manifest.jsonl
transcribe.py     video -> Speaches (Whisper) -> transcripts.jsonl
speaker_id.py     video + segmen -> voiceprint ECAPA -> speakers.jsonl   (di Docker)
export_xlsx.py    semua .jsonl -> Excel (sheet Transkrip + Ringkasan)
```

## Isi folder

| Path | Fungsi |
|---|---|
| `fetch_videos.py` | Menjalankan satu `_search` ke OpenSearch, mengunduh `media_url` tiap hasil (fallback yt-dlp), menulis `manifest.jsonl` |
| `transcribe.py` | Transkripsi lewat Speaches, filter halusinasi Whisper, fallback tanpa prompt bila prompt bocor |
| `export_xlsx.py` | Ekspor ke Excel, termasuk kolom voiceprint bila ada `speakers.jsonl` |
| `export_csv.py` | Ekspor CSV lama (tanpa kolom voiceprint) |
| `queries/` | Body query OpenSearch (`query_prabowo.json`, `query_prabowo_mbg.json`) |
| `speaker-id/` | Dockerfile + `speaker_id.py` (enroll / test / score) |
| `speaker-id/refs/` | Voiceprint referensi (`prabowo.npy`) dan metadatanya |
| `speaches/docker-compose.yml` | Service STT Speaches di port 8010 |

## Setup (sekali)

1. Docker Desktop berjalan.
2. Paket Python di host:
   ```powershell
   pip install -r D:\audio-converter\requirements.txt
   ```
3. Service STT (sudah berjalan, `restart: unless-stopped`):
   ```powershell
   docker compose -f D:\audio-converter\speaches\docker-compose.yml up -d
   ```
4. Image speaker-id (sudah ada sebagai `speaker-id:local`; build ulang hanya bila Dockerfile berubah):
   ```powershell
   docker build -t speaker-id:local D:\audio-converter\speaker-id
   ```
   Model ECAPA diunduh sekali ke volume `spk-cache` saat pertama dipakai.
5. Kredensial OpenSearch: file netrc di luar folder ini, isinya
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

# 2. Transkripsi (sekitar 2x real-time di CPU)
python transcribe.py --dir $B

# 3. Cek suara Prabowo
docker run --rm -e PYTHONIOENCODING=utf-8 -v "C:\Users\wayan\Downloads:/data" -v "D:\audio-converter\speaker-id:/app" -v spk-cache:/cache speaker-id:local python speaker_id.py score --name prabowo --dir /data/socmed-prabowo-mbg-20260929

# 4. Ekspor Excel
python export_xlsx.py --dir $B --out "C:\Users\wayan\Downloads\transkrip-prabowo-mbg-20260929.xlsx" --netrc-file C:\path\ke\os.netrc --ref-source "PRESIDEN PRABOWO Kadang-kadang Saya Kalau Bicara di Audiens Suka Dipelintir.mp3"
```

Rumus di Excel belum punya nilai tersimpan sampai file dibuka di Excel. Buka sekali lalu
simpan, atau tekan Ctrl+Alt+F9.

Opsi yang sering dipakai:

- `transcribe.py --retry-failed`: ulangi video yang gagal.
- `transcribe.py --reclassify`: hitung ulang filter halusinasi dan status dari segmen tersimpan,
  tanpa transkripsi ulang (membuat `transcripts.jsonl.bak` dulu).
- `transcribe.py --prompt ""`: transkripsi tanpa prompt ejaan.
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
- `transcripts.jsonl`: transkrip, segmen bertimestamp, dan status.
  Status: `done` / `no_speech` / `suspect_hallucination` / `no_audio` / `failed`.
- `speakers.jsonl`: per video, berisi `verdict` ya/tidak, durasi suara Prabowo, suara yang
  terdeteksi, dan segmen yang diatribusikan.

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
