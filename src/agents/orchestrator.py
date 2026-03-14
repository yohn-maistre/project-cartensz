"""
Orkestrator ADK - Jantung Sistem Pakar Cartensz.

Alur 5 tahap terpadu:
  1. PreprocessorAgent -> NormalizedText
  2. SignalExtractorAgent -> List[ThreatSignal] (jalur kaidah, tanpa LLM)
  3. Pakar Lokal Classifier -> probabilitas SetFit (tanpa LLM)
  4. Fused BriefWriterAgent -> ClassificationResult + IntelligenceBrief (1 panggilan LLM)

Total pemanggilan LLM per naskah: 1 (mode the radar) atau 0 (mode the sweep).
"""
import asyncio
from src.models import (
    NormalizedText,
    ClassificationResult,
    ThreatSignal,
    IntelligenceBrief,
)


async def run_pipeline(text: str) -> IntelligenceBrief:
    """
    jalankan alur analisis ancaman utuh pada satu masukan teks.
    operasi 'The Radar' — pisau bedah analisis tajam dengan 1 pemanggilan LLM.
    mengembalikan struktur IntelligenceBrief tervalidasi.
    """
    import time
    import hashlib
    from src.db import log_analysis
    
    start_time = time.time()

    # pemuatan tunda agar tidak menabrak rute antar pustaka
    from src.agents.preprocessor import preprocess
    from src.agents.signal_extractor import extract_signals
    from src.agents.brief_writer import classify_local, generate_brief

    # tahap 1: praproses teks
    normalized: NormalizedText = preprocess(text)

    # tahap 2: pencabutan sinyal rawan (murni kalkulasi kaidah)
    signals: list[ThreatSignal] = extract_signals(text, normalized)

    # tahap 3: deteksi pakar mesin (setfit bahasa lokal murni)
    local_probs = classify_local(normalized.normalized_text)

    # tahap 4: penulisan pandangan intelijen utuh melalui LLM
    brief: IntelligenceBrief = await generate_brief(
        normalized=normalized,
        signals=signals,
        local_probs=local_probs,
    )

    latency_ms = (time.time() - start_time) * 1000
    doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
    
    # rekam log ke DuckDB
    try:
        log_analysis(
            doc_id=doc_id,
            text=text,
            predicted_label=brief.classification.label,
            risk_score=brief.risk_score,
            confidence=brief.classification.confidence,
            entropy=brief.classification.entropy,
            is_ambiguous=len(brief.classification.prediction_set) > 1,
            latency_ms=latency_ms,
            pipeline_mode="RADAR"
        )
    except Exception as e:
        print(f"peringatan: gagal merekam jejak analisis ke DuckDB: {e}")

    return brief


async def batch_classify(texts: list[str]) -> list[dict]:
    """
    sistem klasifikasi massal - 'The Sweep'.
    hanya menggunakan model SetFit lokal. nol panggilan LLM.
    mengembalikan luaran deteksi kilat (triage) meliputi:
    - sorotan sinyal ancaman berbasis aturan
    - sorotan bobot pemusatan kata (attention) dari mesin dasar NusaBERT
    - graf penyebaran kalimat spasial 768 dimensi
    """
    from src.agents.preprocessor import preprocess
    from src.agents.signal_extractor import extract_signals
    from src.agents.brief_writer import (
        classify_local_with_attention,
        compute_entropy,
        conformal_prediction_set,
        determine_confidence,
        encode_texts,
    )
    import math

    # ekstraksi grafis seluruh badan wacana dalam sekali operasi
    embeddings = encode_texts(texts)

    results = []
    for idx, text in enumerate(texts):
        normalized = preprocess(text)
        signals = extract_signals(text, normalized)
        
        # klasifikasi kilat disertai ekstraksi bobot fokus kalimat
        local_probs, attention_highlights = classify_local_with_attention(normalized.normalized_text)

        if local_probs:
            label = max(local_probs, key=lambda k: local_probs[k])
            entropy = compute_entropy(local_probs)
            pred_set = conformal_prediction_set(local_probs)
            confidence = determine_confidence(local_probs)
        else:
            # skenario pengamanan jika model mesin tak terdeteksi
            label = "WASPADA"
            local_probs = {"AMAN": 0.33, "WASPADA": 0.34, "TINGGI": 0.33}
            entropy = math.log2(3)
            pred_set = ["AMAN", "WASPADA", "TINGGI"]
            confidence = "LOW"
            attention_highlights = []

        is_ambiguous = len(pred_set) > 1
        
        # rekam log ke DuckDB
        try:
            import hashlib
            from src.db import log_analysis
            import time
            doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
            log_analysis(
                doc_id=doc_id,
                text=text,
                predicted_label=label,
                risk_score=0, # ditangguhkan pada operasi massal
                confidence=confidence,
                entropy=round(entropy, 4),
                is_ambiguous=is_ambiguous,
                latency_ms=0.0,
                pipeline_mode="SWEEP"
            )
        except Exception as e:
            print(f"peringatan: gagal merekam analisis massal ke DuckDB: {e}")

        # persiapkan format sinyal pemantauan antarmuka grafis
        signal_highlights = [
            {"text": s.extracted_text, "type": s.signal_type, "significance": s.significance}
            for s in signals
        ]

        result = {
            "text": text[:200],
            "label": label,
            "confidence": confidence,
            "entropy": round(entropy, 4),
            "probabilities": local_probs,
            "prediction_set": pred_set,
            "signal_count": len(signals),
            "signals": [
                {"type": s.signal_type, "text": s.extracted_text, "significance": s.significance}
                for s in signals
            ],
            "signal_highlights": signal_highlights,
            "attention_highlights": [
                {"token": h["token"], "score": h["score"]}
                for h in attention_highlights[:15]  # pangkas di 15 token utama terpanas
            ],
        }
        
        # sertakan metadata pemeta ruang grafis spasial jika tersedia
        if embeddings and idx < len(embeddings):
            result["embedding"] = embeddings[idx]
        
        results.append(result)

    return results


def analyze(text: str) -> IntelligenceBrief:
    """sarung sinkron pembungkus operasi analisis asinkron utuh."""
    return asyncio.run(run_pipeline(text))
