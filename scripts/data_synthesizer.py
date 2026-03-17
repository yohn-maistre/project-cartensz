import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm
from src.llm_client import llm_completion

DATA_DIR = Path(__file__).parent.parent / "data" / "labeled"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "curated"

def generate_synthetic_tinggi(real_tinggi_df: pd.DataFrame, num_needed: int) -> pd.DataFrame:
    """Uses Gemini 3 Flash to generate synthetic TINGGI samples by few-shotting from real ones."""
    synthetic_texts = []
    
    # We use 5 random real samples per prompt to generate 5 synthetic ones
    # Iterating until we reach num_needed
    pbar = tqdm(total=num_needed, desc="Generating Synthetic TINGGI")
    
    while len(synthetic_texts) < num_needed:
        # Sample 5 random real texts for context
        seeds = real_tinggi_df["text"].sample(5).tolist()
        
        prompt = f"""You are an expert AI threat researcher simulating radicalization and threat narratives in Indonesian (Bahasa Indonesia) for a classification model's training dataset.

The following 5 texts are REAL examples of the "TINGGI" (High Threat) class, indicating direct incitement to violence, terror, or severe coordinated attacks (such as explicit calls to bomb or kill a group/country):

{json.dumps(seeds, indent=2, ensure_ascii=False)}

Your task: Generate 5 NEW, distinct, synthetic text samples that strongly match the semantic profile of the "TINGGI" class.
They should sound realistic, using Indonesian slang, typos, and aggressive phrasing similar to the inputs, but NOT copying them exactly.
Focus on topics like:
- Direct violent action against specific groups or countries
- Organizing attacks or terror
- Calls for extreme physical harm

OUTPUT FORMAT: Return a valid JSON array of 5 strings. No markdown blocks, just the raw JSON array.
"""
        try:
            response_text = llm_completion(prompt=prompt, temperature=0.7, use_cache=False)
            
            # Clean response (often has markdown like ```json ... ```)
            text = response_text
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            generated = json.loads(text.strip())
            
            if isinstance(generated, list) and all(isinstance(x, str) for x in generated):
                for t in generated:
                    if len(synthetic_texts) < num_needed:
                        synthetic_texts.append(t)
                        pbar.update(1)
        except Exception as e:
            print(f"Generation failed, retrying: {e}")
            time.sleep(2)  # Backoff
            
    pbar.close()
    
    return pd.DataFrame({
        "text": synthetic_texts,
        "label": ["TINGGI"] * len(synthetic_texts),
        "source": ["synthetic_gemini"] * len(synthetic_texts)
    })

def create_balanced_dataset():
    """Extracts exactly 200 AMAN, 200 WASPADA, and aggregates 80 real TINGGI + 120 synthetic TINGGI."""
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    full_df = pd.concat([train_df, test_df])
    
    # Extract 200 AMAN and WASPADA
    aman_df = full_df[full_df["label"] == "AMAN"].sample(200, random_state=42)
    aman_df["source"] = "indodiscourse_real"
    
    waspada_df = full_df[full_df["label"] == "WASPADA"].sample(200, random_state=42)
    waspada_df["source"] = "indodiscourse_real"
    
    # Extract all real TINGGI
    real_tinggi_df = full_df[full_df["label"] == "TINGGI"]
    real_tinggi_df["source"] = "indodiscourse_real"
    
    print(f"Loaded {len(aman_df)} AMAN, {len(waspada_df)} WASPADA, {len(real_tinggi_df)} real TINGGI.")
    
    num_synthetic_needed = 200 - len(real_tinggi_df)
    print(f"Need {num_synthetic_needed} synthetic TINGGI samples to reach 200.")
    
    synth_tinggi_df = generate_synthetic_tinggi(real_tinggi_df, num_synthetic_needed)
    
    # Combine everything
    balanced_df = pd.concat([aman_df, waspada_df, real_tinggi_df, synth_tinggi_df])[["text", "label", "source"]]
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / "setfit_balanced_600.csv"
    balanced_df.to_csv(out_path, index=False)
    print(f"\nSaved perfectly balanced 600-sample dataset to {out_path}")
    print(balanced_df["label"].value_counts())
    print("\nSource breakdown:")
    print(balanced_df.groupby(["label", "source"]).size())

if __name__ == "__main__":
    create_balanced_dataset()
