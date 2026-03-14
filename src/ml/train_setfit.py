import pandas as pd
from pathlib import Path

# --- Tambalan perbaikan sistem setfit 1.1.3 + transformers >= 4.45 ---
# alat pelatihan SetFit membawa default_logdir milik transformers lama.
# injeksi fungsi kembali demi mencegah kelumpuhan kode model dasar.
import transformers.training_args as _ta
if not hasattr(_ta, "default_logdir"):
    import socket
    from datetime import datetime as _dt

    def _default_logdir() -> str:
        current_time = _dt.now().strftime("%b%d_%H-%M-%S")
        return str(Path("runs") / f"{current_time}_{socket.gethostname()}")

    _ta.default_logdir = _default_logdir
# --- Batas tambalan ---

from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import json
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# konfigurasi sistem utama
MODEL_NAME = "LazarusNLP/all-nusabert-base-v4"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CURATED_DIR = DATA_DIR / "curated"
OUTPUT_DIR = DATA_DIR / "setfit_model" / "nusabert_base_ft"
REPORTS_DIR = Path(__file__).parent.parent.parent / "notebooks" / "reports"

LABEL2ID = {"AMAN": 0, "WASPADA": 1, "TINGGI": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# pengaturan hiper-parameter SetFit bawaan
EPOCHS = 3
BATCH_SIZE = 4   # dipangkas paksa dari 16 menjadi 4 agar PC tidak mogok / OOM (os error 112)
NUM_ITERATIONS = 3  # kompromi 3 siklus pendaftaran pasangan bahasa untuk durasi (awalnya 5)

def train():
    """Aktivasi proses belajar kontras dengan jumlah pelatihan terbatas menggunakan dataset sintesis yang rata."""
    print("Sistem Pelatihan Mesin Cepat (Few-Shot Setfit) Project Cartensz")
    print("=" * 60)

    # tahapan awal: Muat himpunan data yang rata seimbang
    print("\nMenyerap kompilasi tabel basis data...")
    df = pd.read_csv(CURATED_DIR / "setfit_balanced_600.csv")
    
    # pecahan pembagian latih/uji 80/20 di kocok
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["label"] = df["label"].map(LABEL2ID)
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"   Baris Pelatihan: {len(train_df)} | Sampel Validasi: {len(test_df)}")
    print("   Populasi Angka Distribusi:")
    for k, v in ID2LABEL.items():
        print(f"      {v}: {len(train_df[train_df['label'] == k])}")

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # tahap kedua: Tarik rangka utuh NusaBERT via SentenceTransformer
    print(f"\nMemasukkan tulang punggung {MODEL_NAME} guna cegah galat kekosongan (404)...")
    model_body = SentenceTransformer(MODEL_NAME)
    model_head = LogisticRegression()
    
    model = SetFitModel(
        model_body=model_body,
        model_head=model_head,
        labels=[0, 1, 2],
    )

    # tahap ketiga: Pasang persneling pelatihan
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # cegah bom waktu harddisk, pertahankan model terbaik terakhir
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # tahap keempat: Putaran komparasi pembedahan bahasa (contrastive learning)
    print(f"\nInisialisasi putaran latihan beban tinggi ({EPOCHS} periode iterasi, batch={BATCH_SIZE}, {NUM_ITERATIONS} kaitan kata)...")
    trainer.train()
    
    # tahap kelima: Rapor
    print("\nPengecekan lembaran ujian silang...")
    metrics = trainer.evaluate()
    print(f"   Capaian Evaluasi: {metrics}")

    # tahap keenam: Pendedahan grafis matrik kegagalan klasifikasi (Confusion Matrix)
    preds = model.predict(test_df["text"].tolist())
    true_labels = test_df["label"].tolist()

    report = classification_report(
        true_labels,
        preds,
        target_names=["AMAN", "WASPADA", "TINGGI"],
        digits=4
    )
    print(f"\nRangkuman Detail Akurasi:\n{report}")

    cm = confusion_matrix(true_labels, preds)
    print(f"Matriks Galat Sistem:\n{cm}")

    # cabut rincian metrik individu agar bisa dicek ambang batas
    report_dict = classification_report(
        true_labels,
        preds,
        target_names=["AMAN", "WASPADA", "TINGGI"],
        output_dict=True
    )
    
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])
    tinggi_precision = float(report_dict.get("TINGGI", {}).get("precision", 0.0))
    tinggi_recall = float(report_dict.get("TINGGI", {}).get("recall", 0.0))

    # pertimbangan indikator kelulusan KPI (Key Performance Indicator)
    pass_f1 = weighted_f1 >= 0.70
    pass_tinggi = tinggi_precision >= 0.75

    print("\n" + "=" * 40)
    print("🚦 SERTIFIKASI TAYANG (KPI) 🚦")
    print(f"Rata Ukur F1 ≥ 0.70:       {'✅ LULUS' if pass_f1 else '❌ GAGAL'} ({weighted_f1:.4f})")
    print(f"Akurasi Tegas TINGGI ≥ 0.75: {'✅ LULUS' if pass_tinggi else '❌ GAGAL'} ({tinggi_precision:.4f})")
    print("=" * 40)

    # Cetak laporan arsip
    report_data = {
        "model": "SetFit",
        "backbone": MODEL_NAME,
        "dataset": "setfit_balanced_600",
        "train_size": len(train_df),
        "test_size": len(test_df),
        "hyperparams": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_iterations": NUM_ITERATIONS
        },
        "results": {
            "weighted_f1": weighted_f1,
            "tinggi_precision": tinggi_precision,
            "tinggi_recall": tinggi_recall
        },
        "confusion_matrix": cm.tolist()
    }
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "setfit_training_report.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    
    print("\nMerekam dan membakar instans model unggulan SetFit...")
    model.save_pretrained(str(OUTPUT_DIR))
    print(f"Berhasil menaruh model siap pakai di {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
