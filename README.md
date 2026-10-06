# 🎬 AI Auto-Clipper & Video Editor Engine

Sistem pemrosesan video otomatis berbasis AI yang mengekstrak momen viral dari podcast/webinar panjang menjadi klip vertikal 9:16 siap upload ke TikTok, YouTube Shorts, dan Instagram Reels.

---

## 🏗 Arsitektur Pipeline

```
Input Video (MP4/MOV)
      ↓
[1] transcriber.py     → Deepgram Nova-2 API
                          Word-level timestamps + Speaker Diarization
      ↓
[2] multimodal_analyzer.py → Gemini 1.5 Flash API
                              Analisis emosi visual + audio
      ↓
[3] clip_selector.py   → Claude Haiku / GPT-4o-mini
                          JSON instruksi potong klip viral
      ↓
[4] video_editor.py    → FFmpeg + OpenCV + MediaPipe
                          Crop 9:16 + ASS Karaoke + Render MP4
      ↓
Output Klip MP4 (1080×1920)
```

---

## ⚙️ Instalasi

### 1. Prasyarat Sistem
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows: Download dari https://ffmpeg.org/download.html
# dan tambahkan ke PATH
```

### 2. Clone & Setup Virtual Environment
```bash
git clone <repo-url>
cd ai-auto-clipper

python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies Python
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi API Keys
```bash
cp .env.example .env
# Edit .env dan isi semua API key yang dibutuhkan
```

---

## 🚀 Penggunaan

### Penggunaan Dasar
```bash
python main.py --input podcast.mp4
```

### Opsi Lengkap
```bash
python main.py \
  --input podcast.mp4 \
  --output hasil_klip/ \
  --llm claude \           # atau: openai
  --log-level INFO \
  --log-file run.log
```

### Lewati Gemini (Hemat Kuota / Fallback Teks)
```bash
python main.py --input podcast.mp4 --skip-gemini
```

### Reset Cache & Jalankan Ulang
```bash
python main.py --input podcast.mp4 --clear-cache
```

---

## 📁 Struktur Output

```
output/
├── clip_001.mp4         ← Klip viral #1 (9:16, 1080×1920)
├── clip_002.mp4         ← Klip viral #2
├── clip_003.mp4         ← Klip viral #3
└── .cache/
    ├── transcript.json       ← Cache hasil Deepgram
    ├── emotions.json         ← Cache hasil Gemini
    └── clips_selected.json   ← Cache keputusan LLM
```

> **Tip:** File cache `.cache/` memungkinkan Anda mengulang tahap tertentu
> (misalnya hanya re-render) tanpa menghabiskan kuota API.

---

## 🔧 Penggunaan Modul Secara Terpisah

### Hanya Transkripsi
```bash
python transcriber.py podcast.mp4
# → podcast_transcript.json
```

### Hanya Analisis Gemini
```bash
python multimodal_analyzer.py podcast.mp4 3600
# → podcast_emotions.json
```

### Hanya Seleksi Klip (dari cache)
```bash
python clip_selector.py transcript.json emotions.json claude
# → clips_selected.json
```

---

## 📊 Format JSON Output LLM

```json
{
  "total_clips_found": 2,
  "clips": [
    {
      "clip_id": "clip_001",
      "headline": "RAHASIA SUKSES YANG JARANG DIKETAHUI",
      "start_time": 142.8,
      "end_time": 198.5,
      "duration": 55.7,
      "virality_score": 0.92,
      "hook_statement": "Ternyata 90% orang salah paham soal ini...",
      "reason": "Opening dengan fakta mengejutkan + puncak emosi di detik ke-30"
    }
  ]
}
```

---

## 🛡 Penanganan Error & Fallback

| Skenario | Penanganan |
|---|---|
| Gemini rate-limited | Otomatis switch ke Gemini Pro, lalu fallback ke analisis teks |
| Gemini total down | Fallback ke analisis heuristik berbasis kata kunci transkrip |
| Wajah tidak terdeteksi | Center-crop otomatis berdasarkan lebar frame |
| LLM JSON tidak valid | Retry hingga 3x dengan delay |
| FFmpeg error | Log error detail, lanjut ke klip berikutnya |

---

## 📋 Matriks Penilaian Virality (dari PRD)

| Kriteria | Bobot | Indikator |
|---|---|---|
| Hook Statement (0-3 detik) | 35% | Pertanyaan kontroversial / fakta mengejutkan |
| Emotional Peak | 30% | Tawa, nada tinggi, ekspresi dramatis |
| Completeness Narrative | 20% | Pesan utuh, tidak terpotong tengah kalimat |
| Information Density | 15% | Tanpa dead air > 1.5 detik |

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'mediapipe'`**
```bash
pip install mediapipe --upgrade
```

**`FFmpeg not found`**
```bash
# Pastikan ffmpeg ada di PATH:
ffmpeg -version
```

**`Gemini: File processing failed`**
```bash
# Gunakan flag --skip-gemini untuk bypass:
python main.py --input video.mp4 --skip-gemini
```

**Subtitle tidak sinkron dengan audio**
- Pastikan Deepgram menggunakan model `nova-2` dengan `words=True`
- Periksa bahwa `language` sesuai dengan bahasa video

---

## 📜 Lisensi & API Keys

| Service | Free Tier | Link |
|---|---|---|
| Deepgram | 200 jam/bulan gratis | console.deepgram.com |
| Google Gemini | 1500 req/hari gratis | aistudio.google.com |
| Anthropic Claude | $5 kredit awal | console.anthropic.com |
| OpenAI | $5 kredit awal | platform.openai.com |
