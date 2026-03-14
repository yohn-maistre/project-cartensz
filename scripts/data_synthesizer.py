import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm
from src.llm_client import llm_completion

# lokasi penyimpanan
DATA_DIR = Path(__file__).parent.parent / "data" / "labeled"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "curated"

def generate_synthetic_tinggi(real_tinggi_df: pd.DataFrame, num_needed: int) -> pd.DataFrame:
    """buat sampel tulisan sintetis ancaman kelas TINGGI menggunakan Gemini 3 Flash."""
    synthetic_texts = []
    
    # putar keliling daftar dengan referensi acak sampai terpenuhi jumlah yang ditargetkan
    pbar = tqdm(total=num_needed, desc="Mencetak Sampel Sintesis Kelas TINGGI")
    
    while len(synthetic_texts) < num_needed:
        # ambil 5 tulisan asli secara acak untuk dijadikan basis contoh pembuatan teks
        seeds = real_tinggi_df["text"].sample(5).tolist()
        
        prompt = f"""Anda adalah pakar keamanan yang bertugas merumuskan narasi pemodelan radikalisme dalam Bahasa Indonesia untuk melatih model kecerdasan buatan.

Berikut adalah 5 kutipan NYATA dari kelas "TINGGI" (High Threat) yang mengandung ajakan kekerasan, pergerakan teror, atau serangan terorganisir (misalnya ajakan mencederai atau mengebom):

{json.dumps(seeds, indent=2, ensure_ascii=False)}

Misi Anda MURNI AKADEMIS: Ciptakan 5 (lima) sampel teks sintetis BARU yang menyerupai pola semantik kelompok "TINGGI". 
Tulisan harus terkesan nyata, menggunakan bahasa lisan Indonesia, salah ketik alami, dan emosi destruktif yang setara dengan sampel masukan, namun JANGAN menyalin persis.
Fokus pada:
- Rencana konfrontasi terhadap suku/agama/negara.
- Komunikasi operasional pergerakan ekstrem.
- Seruan bahaya maut secara eksplisit.

FORMAT KELUARAN: Berikan array JSON persis dengan 5 entri data teks. Tanpa format markdown.
"""
        try:
            response_text = llm_completion(prompt=prompt, temperature=0.7, use_cache=False)
            
            # singkirkan pernak-pernik tanda blok json markdown luar
            text = response_text
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            generated = json.loads(text.strip())
            
            # muat hasil pembuatan naskah
            if isinstance(generated, list) and all(isinstance(x, str) for x in generated):
                for t in generated:
                    if len(synthetic_texts) < num_needed:
                        synthetic_texts.append(t)
                        pbar.update(1)
        except Exception as e:
            print(f"kegagalan jaringan atau format. mencoba ulang: {e}")
            time.sleep(2)  # jeda anti blokade
            
    pbar.close()
    
    return pd.DataFrame({
        "text": synthetic_texts,
        "label": ["TINGGI"] * len(synthetic_texts),
        "source": ["synthetic_gemini"] * len(synthetic_texts)
    })

def create_balanced_dataset():
    """kumpulkan murni 200 AMAN, 200 WASPADA, dan setarakan 80 TINGGI asli + 120 TINGGI buatan mesin."""
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    full_df = pd.concat([train_df, test_df])
    
    # filter masing-masing 200 AMAN dan WASPADA
    aman_df = full_df[full_df["label"] == "AMAN"].sample(200, random_state=42)
    aman_df["source"] = "indodiscourse_real"
    
    waspada_df = full_df[full_df["label"] == "WASPADA"].sample(200, random_state=42)
    waspada_df["source"] = "indodiscourse_real"
    
    # ambil segenap temuan data TINGGI lapangan
    real_tinggi_df = full_df[full_df["label"] == "TINGGI"]
    real_tinggi_df["source"] = "indodiscourse_real"
    
    print(f"Memuat {len(aman_df)} AMAN, {len(waspada_df)} WASPADA, {len(real_tinggi_df)} TINGGI orisinal.")
    
    num_synthetic_needed = 200 - len(real_tinggi_df)
    print(f"Perlu {num_synthetic_needed} injeksi tambahan TINGGI demi mencapai kuota 200.")
    
    synth_tinggi_df = generate_synthetic_tinggi(real_tinggi_df, num_synthetic_needed)
    
    # lebur menjadi satu paket dataset mandiri
    balanced_df = pd.concat([aman_df, waspada_df, real_tinggi_df, synth_tinggi_df])[["text", "label", "source"]]
    
    # acak kocokan urutan sampel baris
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # ekspor hasil ramuan
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / "setfit_balanced_600.csv"
    balanced_df.to_csv(out_path, index=False)
    print(f"\nDisempurnakan. Data sintesis 600 baris disimpan rata pada {out_path}")
    print(balanced_df["label"].value_counts())
    print("\nAlokasi jenis asal mula data:")
    print(balanced_df.groupby(["label", "source"]).size())

if __name__ == "__main__":
    create_balanced_dataset()
