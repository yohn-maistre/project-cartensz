"""
BriefWriterAgent - Penggabungan Klasifikasi + Pembuatan Laporan Intelijen.

Ini adalah jantung Sistem Pakar Cartensz.
Dalam SATU panggilan API Gemini, agen ini melakukan:
  1. Meninjau probabilitas model lokal, konteks RAG, dan sinyal yang diekstrak.
  2. Melakukan klasifikasi ancaman final (jaring pengaman dari false negative).
  3. Menghasilkan Laporan Intelijen terstruktur.

Ini menggantikan arsitektur lama (ClassifierAgent + CriticAgent + BriefWriterAgent).
"""
import json
import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.models import (
    NormalizedText,
    ClassificationResult,
    ThreatSignal,
    IntelligenceBrief,
)
from src.llm_client import llm_completion


# ---------------------------------------------------------------------------
# model klasifikasi SetFit lokal (dimuat perlahan/lazy loading)
# ---------------------------------------------------------------------------

_setfit_model = None


def _get_setfit_model():
    """muat model SetFit hanya jika tersedia."""
    global _setfit_model
    if _setfit_model is None:
        try:
            # tambalan sementara untuk default_logdir sebelum mengimpor setfit
            import transformers.training_args as _ta
            if not hasattr(_ta, "default_logdir"):
                import socket
                from datetime import datetime as _dt
                from pathlib import Path

                def _default_logdir() -> str:
                    current_time = _dt.now().strftime("%b%d_%H-%M-%S")
                    return str(Path("runs") / f"{current_time}_{socket.gethostname()}")

                _ta.default_logdir = _default_logdir

            from setfit import SetFitModel
            import os
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "setfit_model", "nusabert_base_ft"
            )
            if os.path.exists(model_path):
                _setfit_model = SetFitModel.from_pretrained(model_path)
            else:
                print("[PakarLokal] Model SetFit tidak ditemukan. Beralih ke LLM murni.")
                _setfit_model = "NOT_AVAILABLE"
        except Exception as e:
            print(f"[PakarLokal] Gagal proses pemuatan SetFit: {e}. Beralih ke LLM murni.")
            _setfit_model = "NOT_AVAILABLE"
    return _setfit_model


def classify_local(text: str) -> Optional[dict[str, float]]:
    """
    klasifikasi pakai SetFit lokal.
    mengembalikan probabilitas atau None jika model kosong.
    """
    model = _get_setfit_model()
    if model == "NOT_AVAILABLE" or model is None:
        return None
    
    try:
        probs = model.predict_proba([text])[0]
        label_map = {0: "AMAN", 1: "WASPADA", 2: "TINGGI"}
        return {label_map[i]: float(p) for i, p in enumerate(probs)}
    except Exception as e:
        print(f"[PakarLokal] Gagal proses prediksi SetFit: {e}")
        return None


def encode_texts(texts: list[str]) -> list[list[float]]:
    """
    tarik embedding 768 dimensi dari badan SetFit (SentenceTransformer).
    gratis biaya LLM.
    """
    model = _get_setfit_model()
    if model == "NOT_AVAILABLE" or model is None:
        return []
    
    try:
        st_model = model.model_body  # ambil body transformernya
        embeddings = st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()
    except Exception as e:
        print(f"[PakarLokal] Gagal proses ekstraksi embedding: {e}")
        return []


def classify_local_with_attention(text: str) -> tuple[Optional[dict[str, float]], list[dict]]:
    """
    klasifikasi DAN ekstrak bobot perhatian (attention weights) dari NusaBERT per tokennya.
    
    Mengembalikan:
        (probabilitas, tokens_tersorot)
    """
    model = _get_setfit_model()
    if model == "NOT_AVAILABLE" or model is None:
        return None, []
    
    try:
        import torch
        
        # ambil skor prediksi utama
        probs = model.predict_proba([text])[0]
        label_map = {0: "AMAN", 1: "WASPADA", 2: "TINGGI"}
        prob_dict = {label_map[i]: float(p) for i, p in enumerate(probs)}
        
        # lacak beban relasi antar-kata (attention) dari transformer dasar
        st_model = model.model_body 
        transformer = st_model[0]
        auto_model = transformer.auto_model
        tokenizer = transformer.tokenizer
        
        # pecah kalimat jadi kepingan data
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        # pastikan mesin jalan di prosesor yang selaras
        device = next(auto_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # putaran model
        with torch.no_grad():
            outputs = auto_model(**inputs, output_attentions=True)
        
        # perhatian lapis terakhir (layer)
        last_attention = outputs.attentions[-1]
        
        # ratakan angka penjuru antar kepala-perhatian
        avg_attention = last_attention.mean(dim=1).squeeze(0)
        
        # fokus ke token pangkal CLS
        cls_attention = avg_attention[0].cpu().numpy()
        
        # terjemahkan token mesin kembali ke kata asli
        token_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # susun struktur sorotan daftar dengan mereduksi karakter penanda mesin
        highlighted = []
        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
        
        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            if token in special_tokens:
                continue
            clean_token = token.replace("##", "").replace("▁", "")
            if clean_token:
                highlighted.append({
                    "token": clean_token,
                    "score": float(score),
                    "is_subword": token.startswith("##") or token.startswith("▁"),
                })
        
        # satukan skala beban penilaian menjadi bentuk desimal standar (0-1)
        if highlighted:
            max_score = max(h["score"] for h in highlighted)
            min_score = min(h["score"] for h in highlighted)
            score_range = max_score - min_score if max_score != min_score else 1.0
            for h in highlighted:
                h["score"] = round((h["score"] - min_score) / score_range, 4)
        
        # urutkan dari terlemah hingga terkuat
        highlighted.sort(key=lambda x: x["score"], reverse=True)
        
        return prob_dict, highlighted
        
    except Exception as e:
        print(f"[PakarLokal] Ekstraksi attention rusak: {e}")
        # fallback darurat ke deteksi biasa tanpa grafis
        try:
            probs = model.predict_proba([text])[0]
            label_map = {0: "AMAN", 1: "WASPADA", 2: "TINGGI"}
            return {label_map[i]: float(p) for i, p in enumerate(probs)}, []
        except:
            return None, []


# ---------------------------------------------------------------------------
# fungsi pendukung kalkulasi keraguan/ambiguitas sistem
# ---------------------------------------------------------------------------

def compute_entropy(probs: dict[str, float]) -> float:
    """hitung entropi tebakan mesin. Makin tinggi nilai makin gagu algoritma."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def conformal_prediction_set(
    probs: dict[str, float],
    alpha: float = 0.1,
) -> list[str]:
    """
    Prediksi adaptif conformal.
    mengembalikan susunan ketidakpastian prediktif dengan cakupan dasar 90 persen.
    """
    threshold = 1.0 - alpha
    sorted_labels = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    prediction_set = []
    cumulative = 0.0
    for label, prob in sorted_labels:
        prediction_set.append(label)
        cumulative += prob
        if cumulative >= threshold:
            break
    return prediction_set or list(probs.keys())


def determine_confidence(probs: dict[str, float]) -> str:
    """klasifikasi derajat penentu seberapa keras rasa percaya diri tebakan sistem."""
    max_prob = max(probs.values())
    if max_prob >= 0.75:
        return "HIGH"
    elif max_prob >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# sistem injeksi perintah llm
# ---------------------------------------------------------------------------

FUSED_SYSTEM_PROMPT = """Anda adalah analis intelijen senior pada Program Cartensz.

Bahan kajian yang diberikan:
1. Probabilitas klasifikasi mesin perintis awal
2. Potongan sinyal ancaman terkait

Tugas Anda adalah merangkai formulasi Laporan Intelijen menyeluruh dari kajian-kajian di atas:
1. TINJAU ulang probabilitas model mesin lokal dan tanda sinyal terkait.
2. KLASIFIKASIKAN ancaman. Anda bertindak selaku penyaring puncak (safety net). Bila mesin lolos mendeteksi AMAN tetapi terdapat indikator provokasi tajam, GULINGKAN hasil mesin. Minimalisasi ancaman tak terlacak sekecil apa pun.
3. BUAT laporan singkat mencakup narasi padat, evaluasi skoring ancaman, rekomendasi pergerakan, dan pencatat keraguan mesin.
4. IDENTIFIKASI leksikon khusus pengaruh kuat dalam laporan.

⚠️ PERSYARATAN FORMAT KEBAHASAAN:
WAJIB dan KEHARUSAN menggunakan murni Bahasa Indonesia pada seluruh entitas di bawah:
- "reasoning"
- "summary_narrative"  
- "ambiguity_notes" 
- "key_phrases" -> indikator "reason" 
DILARANG KERAS pemakaian Bahasa Inggris selain struktur nama objek. Tampilkan format lapor resmi analisis kebijakan pertahanan.

PANDUAN PEMBOBOTAN SKOR (RISIKO):
- AMAN tampa sentimen = 0-20
- AMAN bersentimen = 15-35
- WASPADA indikator remang = 30-55
- WASPADA indikator tegas = 50-70
- TINGGI berstatus rencana = 60-80
- TINGGI seruan pergerakan final = 75-100
- Kepercayaan (Confidence) RENDAH -> PENGURANGAN 10-15 POIN
- Kepercayaan TINGGI -> PENAMBAHAN 5-10 POIN

PEDOMAN KLASIFIKASI KASUS KEKERASAN:
1. Diskusi akademis/analitik ≠ Rencana teror.
2. Pemberitaan liputan media ≠ Ajakan pengerahan massa.
3. Sisipan literasi Arab-Indonesia BUKAN instrumen eksklusif radikalisme.
4. Kenali batas sarkasme satire awam internet.
5. Apabila ada kebingungan kuat dalam mengartikan teks awam, TETAPKAN sebagai WASPADA. Pengabaian dapat membahayakan aparat di lapangan.

Wajib memberi balasan khusus JSON murni:
{
  "label": "AMAN" | "WASPADA" | "TINGGI",
  "probabilities": {"AMAN": 0.xx, "WASPADA": 0.xx, "TINGGI": 0.xx},
  "reasoning": "Rasionalisasi dan justifikasi klasifikasi...",
  "summary_narrative": "Ringkasan puncak intelijen sepanjang dua sampai tiga kalimat pendek...",
  "risk_score": 0-100,
  "recommendation": "ARCHIVE" | "MONITOR" | "ESCALATE",
  "ambiguity_notes": "Anotasi penjelas untuk pihak pengulas operasional masa depan berkenaan letak abu-abu tafsir teks",
  "key_phrases": [
    {"text": "potongan kata", "label": "EUPHEMISM|THREAT|ENTITY|CONTEXT", "reason": "kenapa dispesifikasikan"}
  ]
}

Seluruh persentase total "probabilities" TIDAK BOLEH MELAMPAUI 1.0 (100%).
Isian "ambiguity_notes" Wajib BERISI pesan (jangan kosong melompong)."""


def _build_fused_prompt(
    normalized: NormalizedText,
    signals: list[ThreatSignal],
    local_probs: Optional[dict[str, float]] = None,
) -> str:
    """susun injeksi bahasa untuk llm beserta konteks lingkungan sistem."""
    parts = []

    # laporan referensi model lokal
    if local_probs:
        entropy = compute_entropy(local_probs)
        pred_set = conformal_prediction_set(local_probs)
        local_label = max(local_probs, key=lambda k: local_probs[k])
        parts.append("## Laporan ML Lokal (IndoBERT SetFit)")
        parts.append(f"- Label Utama: {local_label}")
        parts.append(f"- Hitungan Probabilitas: {json.dumps(local_probs)}")
        parts.append(f"- Indeks Shannon: {entropy:.4f} (nilai gagu model)")
        parts.append(f"- Konformal Prediksi: {pred_set}")
        parts.append("- CATATAN: ANDA ADALAH PENYARING PUNCAK. GULINGKAN HASIL INI SEWAKTU-WAKTU DIBUTUHKAN.\n")
    else:
        parts.append("## Laporan ML Lokal: RUSAK/NONAKTIF (Transisi LLM Murni)\n")

    # pembacaan ekstraksi ancaman ringan
    if signals:
        parts.append(f"## Indikator Sinyal Ancaman Sementara ({len(signals)} Temuan)")
        for s in signals:
            parts.append(f"  - [{s.signal_type}] \"{s.extracted_text}\" ({s.significance}): {s.context_explanation}")
        parts.append("")
    else:
        parts.append("## Indikator Sinyal Ancaman: Bersih\n")

    # subjek sasaran bahasa
    parts.append("## Naskah Objek Pemrosesan Utama")
    parts.append(f"Bentuk Orisinil: \"{normalized.original_text}\"")
    parts.append(f"Bentuk Baku: \"{normalized.normalized_text}\"")
    parts.append(f"Rasio Kebahasaan: {json.dumps(normalized.language_mix_ratios)}")
    parts.append(f"Takar Kata (Tokens): {normalized.token_count}")
    parts.append(f"Keberadaan Emoji Media: {normalized.has_emoji}")

    parts.append("\n## Rumuskan Pemodelan Intelijen Anda (Output JSON Eksklusif):")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Titik muara pemanggilan LLM utama
# ---------------------------------------------------------------------------

async def generate_brief(
    normalized: NormalizedText,
    signals: list[ThreatSignal],
    local_probs: Optional[dict[str, float]] = None,
) -> IntelligenceBrief:
    """
    Eksekusi penuh pemusatan klasifikasi LLM 1 panggilan mutlak.
    Dikenal secara teknis sebagai Operasi 'The Radar'.
    LLM menjadi sistem pakar pamungkas yang menyempurnakan kalkulasi dasar mesin BERT lokal.
    """
    prompt = _build_fused_prompt(normalized, signals, local_probs)

    raw = llm_completion(
        prompt=prompt,
        system_prompt=FUSED_SYSTEM_PROMPT,
        temperature=0.3,
    )

    # pembersihan respon JSON sebelum parsing
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1])
    
    data = json.loads(clean)

    # Susun paket model dari temuan LLM
    llm_probs = data["probabilities"]
    entropy = compute_entropy(llm_probs)
    pred_set = conformal_prediction_set(llm_probs)

    # satukan dan beri silang bobot bilamana BERT berhasil mendeteksi (Lokal 40% LLM 60%)
    if local_probs:
        final_probs = {}
        for label in ["AMAN", "WASPADA", "TINGGI"]:
            final_probs[label] = round(
                0.4 * local_probs.get(label, 0.0) + 0.6 * llm_probs.get(label, 0.0), 4
            )
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {k: round(v / total, 4) for k, v in final_probs.items()}
        final_entropy = compute_entropy(final_probs)
        final_pred_set = conformal_prediction_set(final_probs)
        final_label = max(final_probs, key=lambda k: final_probs[k])

        # jaringan pengaman: pertahankan status parah bila BERT lokal gagal tangkap
        if data["label"] == "TINGGI" and final_label != "TINGGI":
            final_label = "TINGGI"
            # injeksi persentase bobot darurat
            final_probs["TINGGI"] = max(final_probs["TINGGI"], 0.5)
            total = sum(final_probs.values())
            final_probs = {k: round(v / total, 4) for k, v in final_probs.items()}
            final_entropy = compute_entropy(final_probs)
            final_pred_set = conformal_prediction_set(final_probs)
    else:
        final_probs = llm_probs
        final_entropy = entropy
        final_pred_set = pred_set
        final_label = data["label"]

    classification = ClassificationResult(
        label=final_label,
        probabilities=final_probs,
        confidence=determine_confidence(final_probs),
        entropy=round(final_entropy, 4),
        prediction_set=final_pred_set,
        conformal_set_size=len(final_pred_set),
        reasoning=data["reasoning"],
    )

    # pencetakan resi nomor dokumen
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    doc_id = f"IB_{uuid.uuid4().hex[:8]}_{timestamp}"

    # sisipkan frasa penumbuh ketidakstabilan sebagai sisipan akhir UI
    key_phrases = data.get("key_phrases", [])
    ambiguity = data["ambiguity_notes"]
    if key_phrases:
        phrase_summary = "; ".join([f"'{kp['text']}' ({kp['label']})" for kp in key_phrases[:5]])
        ambiguity += f"\n\n[Frasa Kata Pengaruh Klasifikasi: {phrase_summary}]"

    return IntelligenceBrief(
        document_id=doc_id,
        summary_narrative=data["summary_narrative"],
        classification=classification,
        signals_detected=signals,
        ambiguity_notes=ambiguity,
        risk_score=data["risk_score"],
        recommendation=data["recommendation"],
    )
