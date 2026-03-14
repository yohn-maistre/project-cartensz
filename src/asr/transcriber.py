"""
ASR Transcriber — Qwen3-ASR-0.6B untuk Pengenalan Suara Bahasa Indonesia.

Mengubah file audio menjadi teks menggunakan model Qwen3-ASR, 
lalu memproses hasilnya menggunakan pipeline analisis ancaman.
Format yang didukung: wav, mp3, ogg, flac, m4a.
"""
import os
import tempfile
from typing import Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# Pemuatan lambat model ASR (berat — dimuat hanya saat dibutuhkan)
# ---------------------------------------------------------------------------

_asr_model = None


def _get_asr_model():
    """Memuat model Qwen3-ASR-0.6B secara lambat. Pemanggilan pertama mengunduh ~1.2GB."""
    global _asr_model
    if _asr_model is None:
        import torch
        from qwen_asr import Qwen3ASRModel

        print("[ASR] Memuat Qwen3-ASR-0.6B... (unduhan perdana ~1.2GB)")
        _asr_model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            max_inference_batch_size=4,
            max_new_tokens=512,  # Mendukung durasi audio panjang
        )
        print("[ASR] Qwen3-ASR-0.6B berhasil dimuat.")
    return _asr_model


def transcribe(
    audio_path: str,
    language: Optional[str] = "Indonesian",
) -> dict:
    """
    Mentranskripsi file audio menjadi teks menggunakan Qwen3-ASR-0.6B.
    
    Args:
        audio_path: Identitas path file audio (wav, mp3, ogg, flac, m4a).
        language: Paksaan bahasa (bawaan: "Indonesian"). Kosongkan untuk deteksi otomatis.
    
    Returns:
        Kamus (dict) berisi:
            - text: Hasil transkripsi
            - language: Bahasa yang terdeteksi/dipaksa
            - audio_path: Path file audio asal
    """
    model = _get_asr_model()

    results = model.transcribe(
        audio=audio_path,
        language=language,
    )

    result = results[0]
    return {
        "text": result.text.strip(),
        "language": result.language,
        "audio_path": audio_path,
    }


def transcribe_bytes(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = "Indonesian",
) -> dict:
    """
    Mentranskripsi audio langsung dari urutan byte mentah (misalnya dari file unggahan).
    Menyimpan ke file sementara, memproses, lalu menghapus file sisa.
    """
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        return transcribe(tmp_path, language=language)
    finally:
        os.unlink(tmp_path)


async def transcribe_and_analyze(
    audio_path: str,
    language: Optional[str] = "Indonesian",
) -> dict:
    """
    Saluran Analisis ASR → Ancaman:
    1. Transkripsi audio menjadi teks (Qwen3-ASR)
    2. Jalankan pipeline analisis (The Radar — 1 panggilan LLM)
    
    Mengembalikan dict berisi transkripsi dan IntelligenceBrief.
    """
    from src.agents.orchestrator import run_pipeline

    # Langkah 1: Transkripsi bunyi
    asr_result = transcribe(audio_path, language=language)
    transcribed_text = asr_result["text"]

    if not transcribed_text or len(transcribed_text.strip()) < 5:
        return {
            "transcription": asr_result,
            "brief": None,
            "error": "Teks hasil transkripsi terlalu pendek atau kosong — analisis dihentikan.",
        }

    # Langkah 2: Eksekusi deteksi ancaman
    brief = await run_pipeline(transcribed_text)

    return {
        "transcription": asr_result,
        "brief": brief,
        "error": None,
    }


def compute_wer(reference: str, hypothesis: str) -> dict:
    """
    Menghitung Word Error Rate (WER) perbandingan referensi dan hipotesis.
    Menggunakan modul jiwer sebagai standar kalkulasi WER.
    
    Returns:
        Dict berisi nilai wer, mer, wil, wip, substitutions, deletions, insertions
    """
    from jiwer import wer, mer, wil, wip, process_words

    output = process_words(reference, hypothesis)

    return {
        "wer": round(output.wer, 4),
        "mer": round(output.mer, 4),
        "wil": round(output.wil, 4),
        "wip": round(output.wip, 4),
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "reference_length": len(reference.split()),
        "hypothesis_length": len(hypothesis.split()),
    }
