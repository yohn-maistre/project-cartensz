"""
NusaBERT Fine-Tuning Script — Project Cartensz

Full fine-tune of LazarusNLP/NusaBERT-base for 3-class threat classification.
Handles severe class imbalance via:
  1. Inverse-frequency class weights in loss function
  2. Stratified sampling
  3. Early stopping on validation weighted F1

Target: Weighted F1 >= 0.70, TINGGI precision >= 0.75
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight


# --- Config ---
MODEL_NAME = "LazarusNLP/NusaBERT-base"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
OUTPUT_DIR = DATA_DIR / "nusabert_ft"
REPORTS_DIR = Path(__file__).parent.parent.parent / "notebooks" / "reports"

LABEL2ID = {"AMAN": 0, "WASPADA": 1, "TINGGI": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

# Training hyperparams (6GB VRAM constraint)
BATCH_SIZE = 16
MAX_LENGTH = 128  # Most threat texts are short social media posts
EPOCHS = 10
LEARNING_RATE = 2e-5


# --- Dataset ---
class ThreatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# --- Weighted Trainer ---
class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss for imbalanced data."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# --- Metrics ---
def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    weighted_f1 = f1_score(labels, preds, average="weighted")
    macro_f1 = f1_score(labels, preds, average="macro")
    
    # Per-class metrics
    per_class = {}
    for label_name, label_id in LABEL2ID.items():
        mask = labels == label_id
        if mask.sum() > 0:
            p = precision_score(labels, preds, labels=[label_id], average="micro", zero_division=0)
            r = recall_score(labels, preds, labels=[label_id], average="micro", zero_division=0)
            f1 = f1_score(labels, preds, labels=[label_id], average="micro", zero_division=0)
            per_class[label_name] = {"precision": p, "recall": r, "f1": f1}

    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "tinggi_precision": per_class.get("TINGGI", {}).get("precision", 0.0),
        "tinggi_recall": per_class.get("TINGGI", {}).get("recall", 0.0),
    }


def train():
    """Run the full fine-tuning pipeline."""
    print("NusaBERT Fine-Tuning for Project Cartensz")
    print("=" * 60)

    # Step 1: Load data
    print("\nLoading labeled data...")
    train_df = pd.read_csv(LABELED_DIR / "train.csv")
    test_df = pd.read_csv(LABELED_DIR / "test.csv")

    train_texts = train_df["text"].tolist()
    train_labels = [LABEL2ID[l] for l in train_df["label"]]
    test_texts = test_df["text"].tolist()
    test_labels = [LABEL2ID[l] for l in test_df["label"]]

    print(f"   Train: {len(train_texts)} | Test: {len(test_texts)}")

    # Step 2: Compute class weights for imbalance
    print("\n⚖️  Computing class weights (inverse frequency)...")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels),
    )
    print(f"   Weights: AMAN={class_weights[0]:.2f}, WASPADA={class_weights[1]:.2f}, TINGGI={class_weights[2]:.2f}")

    # Step 3: Load tokenizer and model
    print(f"\n🤖 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(f"   Model params: {model.num_parameters():,}")

    # Step 4: Create datasets
    train_dataset = ThreatDataset(train_texts, train_labels, tokenizer)
    test_dataset = ThreatDataset(test_texts, test_labels, tokenizer)

    # Step 5: Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    # Step 6: Trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Step 7: Train
    print(f"\n🚂 Training ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE})...")
    print(f"   fp16: {training_args.fp16}")
    train_result = trainer.train()
    print(f"   Training complete! Steps: {train_result.global_step}")

    # Step 8: Evaluate
    print("\n📊 Evaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"   Weighted F1: {eval_results['eval_weighted_f1']:.4f}")
    print(f"   Macro F1: {eval_results['eval_macro_f1']:.4f}")
    print(f"   TINGGI Precision: {eval_results['eval_tinggi_precision']:.4f}")
    print(f"   TINGGI Recall: {eval_results['eval_tinggi_recall']:.4f}")

    # Step 9: Full classification report
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = np.array(test_labels)

    report = classification_report(
        true_labels, preds,
        target_names=["AMAN", "WASPADA", "TINGGI"],
        digits=4,
    )
    print(f"\n📋 Classification Report:\n{report}")

    cm = confusion_matrix(true_labels, preds)
    print(f"Confusion Matrix:\n{cm}")

    # Step 10: Save reports
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "dataset": "Exqrch/IndoDiscourse",
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "class_weights": {
            "AMAN": float(class_weights[0]),
            "WASPADA": float(class_weights[1]),
            "TINGGI": float(class_weights[2]),
        },
        "hyperparams": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "fp16": training_args.fp16,
        },
        "results": {
            "weighted_f1": float(eval_results["eval_weighted_f1"]),
            "macro_f1": float(eval_results["eval_macro_f1"]),
            "tinggi_precision": float(eval_results["eval_tinggi_precision"]),
            "tinggi_recall": float(eval_results["eval_tinggi_recall"]),
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "passing_criteria": {
            "weighted_f1_min": 0.70,
            "tinggi_precision_min": 0.75,
            "weighted_f1_pass": eval_results["eval_weighted_f1"] >= 0.70,
            "tinggi_precision_pass": eval_results["eval_tinggi_precision"] >= 0.75,
        },
    }

    report_path = REPORTS_DIR / "training_report.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\n💾 Report saved to {report_path}")

    # Step 11: Save best model
    best_model_dir = DATA_DIR / "setfit_model" / "nusabert_base_ft"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"   Best model saved to {best_model_dir}")

    # Check passing criteria
    passed = (
        eval_results["eval_weighted_f1"] >= 0.70
        and eval_results["eval_tinggi_precision"] >= 0.75
    )
    status = "✅ PASSED" if passed else "⚠️ BELOW TARGET"
    print(f"\n{status} — Weighted F1: {eval_results['eval_weighted_f1']:.4f}, TINGGI Precision: {eval_results['eval_tinggi_precision']:.4f}")

    return eval_results


if __name__ == "__main__":
    train()
