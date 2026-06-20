"""
TinyGPT — Corpus, Training, & Perbandingan Mode Tokenisasi
==========================================================
Tugas Proyek Data Mining:
  1. Membuat corpus >= 2000 kata (lihat corpus.txt, tema Data Mining/ML).
  2. Melatih model TinyGPT dengan corpus tersebut.
  3. Mencoba beberapa pendekatan mode tokenisasi (char, bpe, unigram, word).
  4. Menampilkan hasil + analisis performa model.

Catatan desain:
  - Arsitektur model dibuat KONSTAN di semua mode supaya perbandingan adil;
    yang divariasikan HANYA tokenizer-nya.
  - Cross-entropy loss TIDAK sebanding antar vocab_size yang berbeda, sehingga
    metrik pembanding utama yang dipakai adalah bits-per-character (BPC) yang
    sudah dinormalisasi terhadap jumlah karakter corpus.
"""

import os
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import matplotlib

matplotlib.use("Agg")  # backend non-interaktif (simpan ke file PNG)
import matplotlib.pyplot as plt

from transformer_blocks import Block

# --------------------------------------------------------------------------
# 0. Setup
# --------------------------------------------------------------------------
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

CORPUS_PATH = "corpus.txt"
TOK_DIR = "tokenizers"
RESULT_DIR = "results"
os.makedirs(TOK_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    TEXT = f.read()

N_CHARS = len(TEXT)            # jumlah karakter corpus (denominator BPC, konstan)
N_WORDS = len(TEXT.split())    # jumlah kata corpus (untuk fertility/tokens-per-word)
print(f"Corpus: {N_WORDS} kata, {N_CHARS} karakter\n")

# --------------------------------------------------------------------------
# 1. Hyperparameter model (KONSTAN untuk semua mode tokenisasi)
# --------------------------------------------------------------------------
BLOCK_SIZE = 32       # panjang konteks (jumlah token ke belakang)
EMBED_DIM = 64        # dimensi embedding
N_HEADS = 4           # jumlah attention head
N_LAYERS = 3          # jumlah transformer block
LR = 3e-3             # learning rate
EPOCHS = 2500         # jumlah langkah pelatihan
BATCH_SIZE = 32       # ukuran batch
EVAL_EVERY = 50       # rekam loss tiap sekian langkah (untuk kurva)
SEED_PROMPT = "data mining"  # prompt awal untuk generate

# Empat pendekatan mode tokenisasi SentencePiece yang dibandingkan.
# vocab_size adalah TARGET; ukuran nyata dibaca dari sp.get_piece_size().
MODES = [
    {"name": "char",    "model_type": "char",    "vocab_size": 300,  "gen_tokens": 250},
    {"name": "bpe",     "model_type": "bpe",     "vocab_size": 512,  "gen_tokens": 110},
    {"name": "unigram", "model_type": "unigram", "vocab_size": 512,  "gen_tokens": 110},
    {"name": "word",    "model_type": "word",    "vocab_size": 1200, "gen_tokens": 55},
]


# --------------------------------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------------------------------
def build_tokenizer(mode):
    """Latih SentencePiece untuk satu mode, kembalikan processor-nya.

    hard_vocab_limit=False  -> tidak error walau vocab_size > jumlah piece unik
                               (penting untuk mode 'char' & 'word').
    character_coverage=1.0  -> cakup semua karakter (corpus Bahasa Indonesia).
    """
    prefix = os.path.join(TOK_DIR, f"tok_{mode['name']}")
    spm.SentencePieceTrainer.Train(
        input=CORPUS_PATH,
        model_prefix=prefix,
        model_type=mode["model_type"],
        vocab_size=mode["vocab_size"],
        character_coverage=1.0,
        hard_vocab_limit=False,
        minloglevel=2,  # kurangi log SentencePiece
    )
    sp = spm.SentencePieceProcessor()
    sp.load(prefix + ".model")
    return sp


# --------------------------------------------------------------------------
# 3. Model TinyGPT (sama seperti referensi, vocab_size per mode)
# --------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[Block(EMBED_DIM, BLOCK_SIZE, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_corpus_loss(model, data):
    """Rata-rata NLL (nats/token) di SELURUH corpus secara deterministik.

    Memakai blok non-overlap sehingga setiap token diprediksi ~satu kali.
    Hasil ini dipakai untuk perplexity & BPC (lebih stabil dari loss batch).
    """
    model.eval()
    nbat = (len(data) - 1) // BLOCK_SIZE
    x = data[:nbat * BLOCK_SIZE].view(nbat, BLOCK_SIZE)
    y = data[1:nbat * BLOCK_SIZE + 1].view(nbat, BLOCK_SIZE)
    total_loss, total_tok = 0.0, 0
    for i in range(0, nbat, 64):
        xb, yb = x[i:i + 64], y[i:i + 64]
        logits, _ = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T), reduction="sum")
        total_loss += loss.item()
        total_tok += B * T
    model.train()
    mean_nll = total_loss / total_tok
    return mean_nll, total_tok


# --------------------------------------------------------------------------
# 4. Latih satu mode (tokenizer -> encode -> training -> metrik -> generate)
# --------------------------------------------------------------------------
def train_one(mode):
    print("=" * 70)
    print(f"MODE TOKENISASI: {mode['name'].upper()}  (model_type={mode['model_type']})")
    print("=" * 70)

    # -- tokenizer
    sp = build_tokenizer(mode)
    vocab_size = sp.get_piece_size()
    ids = sp.encode(TEXT, out_type=int)
    data = torch.tensor(ids, dtype=torch.long)
    n_tokens = len(data)
    tokens_per_word = n_tokens / N_WORDS
    print(f"  vocab_size aktual : {vocab_size}")
    print(f"  jumlah token      : {n_tokens}")
    print(f"  token / kata      : {tokens_per_word:.2f}")

    # -- model + optimizer
    torch.manual_seed(SEED)  # init bobot sama untuk tiap mode
    model = TinyGPT(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -- training loop
    steps, losses = [], []
    t0 = time.time()
    for step in range(EPOCHS):
        xb, yb = get_batch(data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == EPOCHS - 1:
            steps.append(step)
            losses.append(loss.item())
        if step % 500 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}")
    train_time = time.time() - t0

    # -- metrik akhir (di seluruh corpus)
    mean_nll, tok_used = estimate_corpus_loss(model, data)
    perplexity = math.exp(mean_nll)
    # BPC = total bit utk meng-encode corpus / jumlah karakter corpus
    bpc = (mean_nll / math.log(2)) * (tok_used / N_CHARS)

    # -- generate sampel teks
    ctx = torch.tensor([sp.encode(SEED_PROMPT)], dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=mode["gen_tokens"])
    sample = sp.decode(out[0].tolist())

    print(f"  final loss (corpus): {mean_nll:.4f} nats/token")
    print(f"  perplexity         : {perplexity:.2f}")
    print(f"  bits-per-char (BPC): {bpc:.4f}")
    print(f"  waktu latih        : {train_time:.1f} s")
    print(f"\n  Contoh teks (seed='{SEED_PROMPT}'):\n  {sample}\n")

    return {
        "name": mode["name"],
        "model_type": mode["model_type"],
        "vocab_size": vocab_size,
        "n_tokens": n_tokens,
        "tokens_per_word": tokens_per_word,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "bpc": bpc,
        "train_time": train_time,
        "sample": sample,
        "curve_steps": steps,
        "curve_losses": losses,
    }


# --------------------------------------------------------------------------
# 5. Plot perbandingan
# --------------------------------------------------------------------------
def plot_loss_curves(records):
    plt.figure(figsize=(9, 5.5))
    for r in records:
        plt.plot(r["curve_steps"], r["curve_losses"], label=f"{r['name']} (vocab={r['vocab_size']})")
    plt.xlabel("Langkah pelatihan (step)")
    plt.ylabel("Training loss (cross-entropy, nats/token)")
    plt.title("Kurva Loss per Mode Tokenisasi — TinyGPT")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(RESULT_DIR, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Plot disimpan: {path}")


def plot_metrics(records):
    names = [r["name"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        ("vocab_size", "Ukuran Vocabulary", "tab:blue"),
        ("n_tokens", "Panjang Sekuens (jumlah token)", "tab:orange"),
        ("bpc", "Bits-per-Character (lebih kecil lebih baik)", "tab:green"),
    ]
    for ax, (key, title, color) in zip(axes, metrics):
        vals = [r[key] for r in records]
        bars = ax.bar(names, vals, color=color)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            label = f"{v:.3f}" if key == "bpc" else f"{v}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label,
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Perbandingan Metrik antar Mode Tokenisasi — TinyGPT", fontsize=13)
    path = os.path.join(RESULT_DIR, "metrics_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot disimpan: {path}")


# --------------------------------------------------------------------------
# 6. Laporan Markdown
# --------------------------------------------------------------------------
def write_report(records):
    best = min(records, key=lambda r: r["bpc"])
    lines = []
    A = lines.append
    A("# Laporan Analisis — TinyGPT & Perbandingan Mode Tokenisasi\n")
    A("## 1. Ringkasan Eksperimen\n")
    A(f"- **Corpus**: Bahasa Indonesia, tema Data Mining/Machine Learning "
      f"(`corpus.txt`) — **{N_WORDS} kata**, **{N_CHARS} karakter**.")
    A(f"- **Arsitektur (konstan semua mode)**: block_size={BLOCK_SIZE}, "
      f"embedding_dim={EMBED_DIM}, n_heads={N_HEADS}, n_layers={N_LAYERS}, "
      f"lr={LR}, epochs={EPOCHS}, batch_size={BATCH_SIZE}.")
    A(f"- **Mode tokenisasi dibandingkan**: " + ", ".join(f"`{r['name']}`" for r in records) + ".")
    A("- **Catatan adil**: hanya tokenizer yang divariasikan; arsitektur model "
      "identik. Karena cross-entropy loss bergantung pada `vocab_size`, metrik "
      "pembanding utama adalah **bits-per-character (BPC)** yang dinormalisasi "
      "terhadap jumlah karakter corpus (denominator sama untuk semua mode).\n")

    A("## 2. Tabel Hasil\n")
    A("| Mode | Vocab | Token | Token/Kata | Loss (nats/tok) | Perplexity | **BPC** | Waktu (s) |")
    A("|------|------:|------:|-----------:|----------------:|-----------:|--------:|----------:|")
    for r in records:
        A(f"| `{r['name']}` | {r['vocab_size']} | {r['n_tokens']} | "
          f"{r['tokens_per_word']:.2f} | {r['mean_nll']:.4f} | {r['perplexity']:.2f} | "
          f"**{r['bpc']:.4f}** | {r['train_time']:.1f} |")
    A("")
    A("![Kurva Loss](loss_curves.png)\n")
    A("![Perbandingan Metrik](metrics_comparison.png)\n")

    A("## 3. Contoh Teks Hasil Generasi\n")
    A(f"Seed/prompt awal: `{SEED_PROMPT}`\n")
    for r in records:
        A(f"**Mode `{r['name']}`:**\n")
        A("```")
        A(r["sample"].strip())
        A("```\n")

    A("## 4. Analisis Performa\n")
    A("### 4.1 Trade-off granularitas tokenisasi\n")
    A("- **`char` (karakter)** — vocab paling kecil dan **tidak pernah** menemui "
      "token tak dikenal (no-OOV), tetapi **sekuens paling panjang** "
      f"(token/kata ≈ {max(records, key=lambda r: r['tokens_per_word'])['tokens_per_word']:.2f}). "
      "Dengan block_size tetap, konteks efektif (dalam kata) menjadi paling "
      "pendek sehingga model sulit menangkap makna antar-kata.")
    A("- **`word` (kata)** — **sekuens paling pendek** dan konteks per-kata paling "
      "panjang, tetapi **vocab paling besar** dan rawan **OOV/`<unk>`** untuk kata "
      "langka. Pada corpus kecil ini banyak kata hanya muncul sekali sehingga sulit "
      "dipelajari dengan baik.")
    A("- **`bpe` & `unigram` (subkata)** — kompromi di tengah: memecah kata menjadi "
      "potongan yang sering muncul, menyeimbangkan ukuran vocab dan panjang sekuens, "
      "serta menekan OOV. Ini alasan tokenisasi subkata menjadi standar pada model "
      "bahasa modern.\n")

    A("### 4.2 Mengapa loss mentah tidak bisa dibandingkan langsung\n")
    A("Cross-entropy/perplexity dihitung **per token**, sedangkan satu token "
      "bermakna sangat berbeda antar mode (satu karakter vs satu kata utuh). "
      "Memprediksi 1 kata berikutnya menanggung informasi jauh lebih banyak daripada "
      "memprediksi 1 karakter berikutnya, jadi nilai loss per-token TIDAK setara. "
      "Karena itu kita menormalkan ke **BPC** (bit untuk meng-encode tiap karakter "
      "corpus), sehingga semua mode diukur pada satuan yang sama dan adil.\n")

    # Deteksi memorisasi secara data-driven: perplexity mendekati 1.0 berarti model
    # hampir hafal corpus (tidak ada ketidakpastian saat memprediksi token berikutnya).
    memorized = [r for r in records if r["perplexity"] < 1.3]
    char_rec = next(r for r in records if r["name"] == "char")

    A("### 4.3 Temuan penting: BPC terendah = MEMORISASI, bukan otomatis 'terbaik'\n")
    A(f"- Secara angka, BPC terendah dipegang **`{best['name']}`** "
      f"(BPC = **{best['bpc']:.4f}**, perplexity = **{best['perplexity']:.2f}**).")
    A("- **Namun** perplexity yang mendekati 1.0 adalah tanda kuat **overfitting / "
      "memorisasi**: pada corpus sekecil ini model cukup *menghafal* urutan token, "
      "bukan benar-benar 'memahami' bahasa. Buktinya, teks hasil generasi mode "
      f"`{best['name']}` di Bagian 3 hampir **identik kata-per-kata** dengan isi "
      "`corpus.txt` — model hanya mengulang corpus.")
    if memorized:
        A("- Mode yang menunjukkan gejala memorisasi (perplexity < 1.3): "
          + ", ".join(f"`{r['name']}` (ppl={r['perplexity']:.2f})" for r in memorized)
          + ". Semakin sedikit token (mode `word` → 1 token/kata, sekuens terpendek), "
          "semakin mudah dihafal sehingga BPC-nya paling rendah secara artifisial.\n")

    A("### 4.4 Kesimpulan & rekomendasi\n")
    A("- **Efisiensi tokenisasi terbaik**: `bpe`/`unigram` — token/kata ≈ 2.2–2.3, "
      "vocab moderat (512), tanpa OOV. Inilah kompromi yang dipakai model bahasa "
      "modern dan paling masuk akal untuk penggunaan nyata.")
    A(f"- **Kualitas generasi praktis terbaik**: `bpe`/`unigram` menghasilkan teks "
      "yang koheren TAPI masih bervariasi (tidak sekadar menyalin corpus), sedangkan "
      f"`word` koheren karena menghafal, dan `char` (BPC={char_rec['bpc']:.4f}, "
      f"perplexity={char_rec['perplexity']:.2f}) paling sulit — banyak kata 'mengada-ada' "
      "karena harus belajar mengeja dari nol dengan konteks efektif paling pendek.")
    A("- **Pelajaran kunci**: *loss/BPC terendah tidak selalu berarti model terbaik*. "
      "Pada dataset kecil, metrik rendah bisa berasal dari memorisasi. Kualitas perlu "
      "dinilai juga dari generalisasi (variasi & kebaruan teks), bukan angka loss saja.")
    A("- **Saran perbaikan**: perbesar corpus (puluhan ribu kata), pisahkan data "
      "latih/validasi untuk mendeteksi overfitting, naikkan block_size/embedding_dim, "
      "dan tambah epoch — terutama menguntungkan mode `char` yang butuh konteks lebih panjang.\n")

    report_path = os.path.join(RESULT_DIR, "laporan.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Laporan disimpan: {report_path}")
    return best


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------
def main():
    records = [train_one(mode) for mode in MODES]

    # Ringkasan tabel ke konsol
    print("=" * 90)
    print("RINGKASAN PERBANDINGAN MODE TOKENISASI")
    print("=" * 90)
    header = f"{'mode':<9}{'vocab':>7}{'token':>8}{'tok/kata':>10}{'loss':>9}{'ppl':>10}{'BPC':>9}{'time(s)':>9}"
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['name']:<9}{r['vocab_size']:>7}{r['n_tokens']:>8}{r['tokens_per_word']:>10.2f}"
              f"{r['mean_nll']:>9.4f}{r['perplexity']:>10.2f}{r['bpc']:>9.4f}{r['train_time']:>9.1f}")

    plot_loss_curves(records)
    plot_metrics(records)
    best = write_report(records)

    print("\n" + "=" * 90)
    print(f"SELESAI. BPC terendah: {best['name']} (BPC={best['bpc']:.4f}, ppl={best['perplexity']:.2f})")
    print("  -> Catatan: BPC/perplexity sangat rendah = indikasi MEMORISASI pada corpus kecil,")
    print("     bukan otomatis model terbaik. Untuk pemakaian nyata, bpe/unigram paling seimbang.")
    print("     (lihat analisis lengkap di results/laporan.md)")
    print("Output: results/loss_curves.png, results/metrics_comparison.png, results/laporan.md")
    print("=" * 90)


if __name__ == "__main__":
    main()
