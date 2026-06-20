# TinyGPT — Corpus, Training, & Perbandingan Mode Tokenisasi

Proyek Data Mining: melatih sebuah model bahasa mini bergaya GPT (TinyGPT) di atas
corpus Bahasa Indonesia, lalu **membandingkan empat pendekatan mode tokenisasi**
(`char`, `bpe`, `unigram`, `word`) dan menganalisis performanya.

## Tujuan Tugas

1. Membuat corpus dengan total **≥ 2000 kata** (bebas domain/topik).
2. Melatih model dengan corpus tersebut.
3. Mencoba beberapa **mode tokenisasi**.
4. Menampilkan hasil dan melakukan **analisis performa** model.

## Struktur Proyek

```
tinyGPT/
├── corpus.txt              # Corpus Bahasa Indonesia (tema Data Mining/ML), 2101 kata
├── transformer_blocks.py   # Blok transformer: self-attention, multi-head, feed-forward
├── main.py                 # Orchestrator: tokenizer → training → metrik → plot → laporan
├── tokenizers/             # Artefak SentencePiece per mode (dibuat otomatis)
│   ├── tok_char.model / .vocab
│   ├── tok_bpe.model / .vocab
│   ├── tok_unigram.model / .vocab
│   └── tok_word.model / .vocab
└── results/                # Output (dibuat otomatis)
    ├── loss_curves.png         # Kurva loss ke-4 mode
    ├── metrics_comparison.png  # Bar chart vocab / panjang sekuens / BPC
    └── laporan.md              # Laporan analisis lengkap
```

## Cara Menjalankan

Proyek memakai virtual environment `.venv` (sudah berisi semua dependensi).

```bash
# dari root proyek
.venv/bin/python main.py
```

Jika perlu menyiapkan environment dari nol:

```bash
python3 -m venv .venv
.venv/bin/pip install torch sentencepiece numpy matplotlib
.venv/bin/python main.py
```

Skrip berjalan **deterministik** (seed = 1337) dan memakan waktu ~2 menit di CPU
(tanpa GPU). Setelah selesai, cek folder `results/`.

> Catatan VS Code: bila muncul warning *"Import could not be resolved"*, pilih
> interpreter `.venv/bin/python` lewat Command Palette → **Python: Select Interpreter**.
> Paket sudah terpasang di `.venv`, jadi ini hanya soal konfigurasi editor.

## Arsitektur Model (konstan untuk semua mode)

Agar perbandingan **adil**, arsitektur dibuat sama di semua mode — yang divariasikan
hanya tokenizer-nya.

| Hyperparameter | Nilai |
|----------------|------:|
| block_size (panjang konteks) | 32 |
| embedding_dim | 64 |
| jumlah head (n_heads) | 4 |
| jumlah layer (n_layers) | 3 |
| learning rate | 3e-3 |
| epochs | 2500 |
| batch_size | 32 |

Alur model: `token + positional embedding → 3× Transformer Block (causal
self-attention + FFN) → LayerNorm → linear head → softmax`.

## Mode Tokenisasi yang Dibandingkan

Keempatnya dilatih dengan **SentencePiece** (`model_type` berbeda):

| Mode | Granularitas | Karakteristik |
|------|--------------|---------------|
| `char` | Karakter | Vocab kecil, **tanpa OOV**, tapi sekuens sangat panjang |
| `bpe` | Subkata (Byte Pair Encoding) | Kompromi vocab vs panjang sekuens |
| `unigram` | Subkata (probabilistik) | Mirip BPE, pemilihan piece berbasis peluang |
| `word` | Kata | Sekuens terpendek, vocab terbesar, rawan **OOV/`<unk>`** |

## Ringkasan Hasil

| Mode | Vocab | Token | Token/Kata | Perplexity | **BPC** |
|------|------:|------:|-----------:|-----------:|--------:|
| `char` | 52 | 15.553 | 7.40 | 1.52 | 0.6062 |
| `bpe` | 512 | 4.655 | 2.22 | 1.17 | 0.0662 |
| `unigram` | 512 | 4.794 | 2.28 | 1.16 | 0.0671 |
| `word` | 868 | 2.101 | 1.00 | 1.07 | **0.0124** |

**BPC = bits-per-character**, metrik yang dinormalisasi terhadap jumlah karakter
corpus. Dipakai karena loss/perplexity per-token **tidak sebanding** antar mode
(1 token bisa berarti 1 karakter atau 1 kata utuh).

## Analisis Singkat

- **Trade-off granularitas terlihat jelas**: dari `char` → `word`, ukuran vocab
  membesar, jumlah token mengecil, dan BPC mengecil — sesuai teori.
- **Temuan penting**: `word` punya BPC terendah, **tetapi karena memorisasi/overfitting**
  pada corpus kecil — teks generasinya nyaris identik kata-per-kata dengan corpus
  (perplexity ≈ 1.07). Jadi **loss terendah ≠ model terbaik**.
- **Paling seimbang untuk pemakaian nyata**: `bpe`/`unigram` — koheren tetapi masih
  bervariasi dan bebas OOV (inilah sebabnya tokenisasi subkata menjadi standar pada
  model bahasa modern).
- **`char`** paling sulit: banyak kata "mengada-ada" karena harus belajar mengeja dari
  nol dengan konteks efektif paling pendek.

Analisis lengkap beserta contoh teks hasil generasi ada di
[results/laporan.md](results/laporan.md).

## Referensi

Diadaptasi dari implementasi TinyGPT minimal pada `.claude/reference/tinyGPT`
(tokenizer SentencePiece + transformer 2-layer), diperluas menjadi pipeline
perbandingan multi-mode tokenisasi dengan metrik dan laporan otomatis.
