"""
Data Pipeline — Project Cartensz
Downloads IndoDiscourse from HuggingFace, cleans it, remaps labels to
AMAN/WASPADA/TINGGI, deduplicates, stratified train/test split.

Source: Exqrch/IndoDiscourse (https://huggingface.co/datasets/Exqrch/IndoDiscourse)
Label mapping:
  - threat_incitement_to_violence (majority) → TINGGI
  - toxicity OR identity_attack (majority, no violence) → WASPADA
  - Neither → AMAN
"""
import json
import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent.parent / "data"
LABELED_DIR = DATA_DIR / "labeled"
RAW_DIR = DATA_DIR / "raw"

# Ensure directories exist
LABELED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


def majority_vote(annotations: list[str]) -> int:
    """Majority vote from annotator labels (list of '0'/'1' strings)."""
    votes = [int(a) for a in annotations]
    return 1 if sum(votes) > len(votes) / 2 else 0


def map_to_threat_label(row: dict) -> str:
    """
    Map IndoDiscourse multi-label annotations to our 3-class system.
    Priority:
      1. If threat/incitement to violence → TINGGI
      2. If toxic OR identity_attack OR insults → WASPADA
      3. Otherwise → AMAN
    """
    is_violent = majority_vote(row["threat_incitement_to_violence"])
    is_toxic = majority_vote(row["toxicity"])
    is_identity_attack = majority_vote(row["identity_attack"])
    is_insults = majority_vote(row["insults"])
    is_spam = majority_vote(row["is_noise_or_spam_text"])

    # Skip spam/noise texts
    if is_spam:
        return "SKIP"

    if is_violent:
        return "TINGGI"
    elif is_toxic or is_identity_attack or is_insults:
        return "WASPADA"
    else:
        return "AMAN"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def run_pipeline():
    print("📥 Downloading IndoDiscourse dataset from HuggingFace...")
    ds = load_dataset("Exqrch/IndoDiscourse", "main", split="main")
    print(f"   Raw rows: {len(ds)}")

    # Convert to pandas for easier manipulation
    df = ds.to_pandas()

    # Save raw copy
    raw_path = RAW_DIR / "indodiscourse_raw.parquet"
    df.to_parquet(raw_path)
    print(f"   Raw saved to {raw_path}")

    # Step 1: Map labels
    print("\n🏷️  Mapping labels to AMAN/WASPADA/TINGGI...")
    df["label"] = df.apply(map_to_threat_label, axis=1)

    # Remove spam/noise
    before = len(df)
    df = df[df["label"] != "SKIP"].copy()
    print(f"   Removed {before - len(df)} spam/noise texts")

    # Step 2: Filter min length
    df = df[df["text"].str.len() >= 10].copy()
    print(f"   After min-length filter (≥10 chars): {len(df)} rows")

    # Step 3: Deduplicate by text hash
    df["text_hash"] = df["text"].apply(text_hash)
    before = len(df)
    df = df.drop_duplicates(subset="text_hash").copy()
    print(f"   Removed {before - len(df)} duplicates")

    # Step 4: Aggregate granular sub-labels (for UI display later)
    for col in ["toxicity", "identity_attack", "insults", "threat_incitement_to_violence",
                 "profanity_obscenity", "polarized", "sexually_explicit"]:
        df[f"{col}_agg"] = df[col].apply(majority_vote)

    # Step 5: Class distribution
    dist = df["label"].value_counts()
    print(f"\n📊 Class distribution:")
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")
    print(f"   Total: {len(df)}")

    # Step 5: Stratified 80/20 split
    print("\n✂️  Stratified 80/20 train/test split...")

    # For small classes, ensure at least 2 samples per class
    min_class_count = df["label"].value_counts().min()
    if min_class_count < 2:
        print(f"   ⚠️ Warning: smallest class has only {min_class_count} sample(s)")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    print(f"   Train: {len(train_df)} rows")
    print(f"   Test: {len(test_df)} rows")

    # Step 8: Save
    # Keep essential columns + granular sub-labels for UI/assessment

    keep_cols = [
        "text_id", "text", "label", "text_hash", "topic",
        "toxicity_agg", "identity_attack_agg", "insults_agg",
        "threat_incitement_to_violence_agg", "profanity_obscenity_agg",
        "polarized_agg", "sexually_explicit_agg",
    ]
    train_out = train_df[keep_cols].reset_index(drop=True)
    test_out = test_df[keep_cols].reset_index(drop=True)

    train_path = LABELED_DIR / "train.csv"
    test_path = LABELED_DIR / "test.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    print(f"\n💾 Saved:")
    print(f"   Train: {train_path} ({len(train_out)} rows)")
    print(f"   Test: {test_path} ({len(test_out)} rows)")

    # Step 7: Save metadata sidecar
    metadata = {
        "source": "Exqrch/IndoDiscourse (HuggingFace)",
        "source_url": "https://huggingface.co/datasets/Exqrch/IndoDiscourse",
        "config": "main",
        "raw_rows": len(ds),
        "cleaned_rows": len(df),
        "train_rows": len(train_out),
        "test_rows": len(test_out),
        "label_mapping": {
            "TINGGI": "threat_incitement_to_violence (majority vote)",
            "WASPADA": "toxicity OR identity_attack OR insults (majority vote, no violence)",
            "AMAN": "none of the above",
        },
        "class_distribution": {
            "train": train_out["label"].value_counts().to_dict(),
            "test": test_out["label"].value_counts().to_dict(),
        },
        "dedup_method": "SHA-256 hash of lowercased stripped text",
        "min_text_length": 10,
        "split_method": "stratified 80/20, random_state=42",
        "synthetic_proportion": 0.0,
        "labeling_criteria": {
            "AMAN": "Safe content: no toxicity, no threats, no identity attacks. Includes news reports, neutral discourse.",
            "WASPADA": "Toxic, insulting, or identity-attacking content that does NOT incite violence. Concerning but not directly actionable.",
            "TINGGI": "Content that incites violence or threatens. Directly actionable threat intelligence.",
        },
    }

    meta_path = LABELED_DIR / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   Metadata: {meta_path}")

    # Step 8: Print train distribution
    print(f"\n📊 Final train distribution:")
    for label, count in train_out["label"].value_counts().items():
        pct = count / len(train_out) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")

    print(f"\n📊 Final test distribution:")
    for label, count in test_out["label"].value_counts().items():
        pct = count / len(test_out) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")

    print("\n✅ Data pipeline complete!")
    return train_out, test_out


if __name__ == "__main__":
    run_pipeline()
