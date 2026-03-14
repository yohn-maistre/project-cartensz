"""
FastAPI REST API untuk Pengklasifikasi Ancaman Narasi GSP.
Titik Akhir (Endpoints):
  POST /analyze — Analisis teks tunggal → Ringkasan Intelijen
  POST /batch — Analisis massal → Daftar Ringkasan Intelijen
  GET /health — Tes kebugaran server
"""
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from src.agents.orchestrator import run_pipeline, batch_classify
from src.models import IntelligenceBrief

app = FastAPI(
    title="GSP Threat Narrative Classifier",
    description="Analisis narasi ancaman berbasis AI untuk teks bahasa Indonesia. "
                "Mengategorikan teks sebagai AMAN/WASPADA/TINGGI dengan kemampuan menjelaskan alasan.",
    version="1.0.0",
)


# --- Skema Permintaan/Tanggapan ---

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Teks Indonesia yang disorot untuk analisis ancaman")


class AnalyzeResponse(BaseModel):
    success: bool
    brief: Optional[IntelligenceBrief] = None
    error: Optional[str] = None


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=20)


class BatchResponse(BaseModel):
    success: bool
    briefs: List[dict] = []
    errors: List[str] = []

class FeedbackRequest(BaseModel):
    text_hash: str
    original_label: str
    corrected_label: str
    notes: Optional[str] = None


# --- Titik Akhir Komunikasi (Endpoints) ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gsp-threat-classifier"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Menganalisis teks tunggal berbahasa Indonesia untuk mendeteksi unsur ancaman."""
    try:
        brief = await run_pipeline(request.text)
        return AnalyzeResponse(success=True, brief=brief)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
async def batch_analyze(request: BatchRequest):
    """Memproses banyak teks menggunakan The Sweep (0 biaya pemanggilan LLM). Cepat dan efisien."""
    try:
        results = await batch_classify(request.texts)
        return BatchResponse(success=True, briefs=results)
    except Exception as e:
        return BatchResponse(success=False, errors=[str(e)])

from src.db import save_feedback

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Menyimpan perbaikan manual analis untuk memperkuat pelabelan aktif."""
    save_feedback(
        text_hash=request.text_hash,
        original_label=request.original_label,
        corrected_label=request.corrected_label,
        notes=request.notes
    )
    print(f"[Sinkronisasi Umpan Balik DuckDB] {request.text_hash}: {request.original_label} -> {request.corrected_label}")
    return {"success": True, "message": "Catatan intervensi disimpan ke pangkalan data DuckDB untuk revisi latihan berikutnya."}


@app.post("/retrain")
async def trigger_retraining():
    """Memicu pelatihan ulang model SetFit menggunakan basis data kurasi dan koreksi terbaru."""
    try:
        # Menjalankan proses di tugas balik layar atau subproses terpisah untuk skala komersial.
        from src.ml.train_setfit import train
        import threading
        
        # Mengeksekusi rangkaian pembelajaran berbasis utas supaya API langsung responsif
        t = threading.Thread(target=train)
        t.start()
        
        return {"success": True, "message": "Proses pelatihan perbaikan berlangsung di latar belakang."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melaksanakan pelatihan dasar ulang: {str(e)}")


from fastapi import UploadFile, File, Form

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("Indonesian"),
    run_analysis: bool = Form(True),
):
    """
    Mentranskripsi berkas arsip modul audio menggunakan standar Qwen3-ASR-0.6B, 
    kemudian dapat meneruskan keluaran teks melintasi rel spesialisasi penangkal ancaman (The Radar).
    
    Format Menerima: wav, mp3, ogg, flac, m4a
    """
    allowed_types = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    import os
    ext = os.path.splitext(file.filename or "audio.wav")[1].lower()
    if ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Struktur ekstensi audio di luar dukungan: {ext}. Terizinkan hanya: {allowed_types}")

    audio_bytes = await file.read()

    try:
        from src.asr.transcriber import transcribe_bytes, transcribe_and_analyze
        import tempfile

        if run_analysis:
            # Membuang ke medium memori sesaat (temp) bagi keperluan saluran transcribe_and_analyze
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                result = await transcribe_and_analyze(tmp_path, language=language or None)
                return {
                    "success": True,
                    "transcription": result["transcription"],
                    "brief": result["brief"].model_dump() if result["brief"] else None,
                    "error": result.get("error"),
                }
            finally:
                os.unlink(tmp_path)
        else:
            asr_result = transcribe_bytes(audio_bytes, filename=file.filename or "audio.wav", language=language or None)
            return {
                "success": True,
                "transcription": asr_result,
                "brief": None,
                "error": None,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sistem ASR gagal operasi: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
