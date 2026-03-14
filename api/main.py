"""
FastAPI REST API untuk Pengklasifikasi Narasi Ancaman GSP.
Titik Akhir (Endpoints):
  POST /analyze — Analisis tunggal teks → Intelligence Brief
  POST /batch — Analisis massal kilat → Kumpulan Intelligence Brief
  GET /health — Pengecekan status layanan
"""
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from src.agents.orchestrator import run_pipeline, batch_classify
from src.models import IntelligenceBrief

app = FastAPI(
    title="GSP Threat Narrative Classifier",
    description="Sistem analisis narasi ancaman berbasis AI untuk teks berbahasa Indonesia. "
                "Mengklasifikasikan dokumen ke dalam skala AMAN/WASPADA/TINGGI beserta penjelesannya.",
    version="1.0.0",
)


# --- Skema Penukaran Data (Request/Response) ---

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Teks berbahasa Indonesia untuk dianalisis")


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


# --- Titik Akses Jaringan (Endpoints) ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gsp-threat-classifier"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Jalankan analisis mendalam pada satu dokumen teks."""
    try:
        brief = await run_pipeline(request.text)
        return AnalyzeResponse(success=True, brief=brief)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
async def batch_analyze(request: BatchRequest):
    """Lakukan pemindaian multi teks melalui The Sweep (0 LLM call). Super ringan & kilat."""
    try:
        results = await batch_classify(request.texts)
        return BatchResponse(success=True, briefs=results)
    except Exception as e:
        return BatchResponse(success=False, errors=[str(e)])

from src.db import save_feedback

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Beri masukan manual analis intel untuk perbaikan akurasi rute aktif (Active Learning)."""
    save_feedback(
        text_hash=request.text_hash,
        original_label=request.original_label,
        corrected_label=request.corrected_label,
        notes=request.notes
    )
    print(f"[Feedback Tersimpan di DuckDB] {request.text_hash}: {request.original_label} -> {request.corrected_label}")
    return {"success": True, "message": "Koreksi tercatat dalam DuckDB untuk putaran latih ulang berikutnya."}


@app.post("/retrain")
async def trigger_retraining():
    """Nyalakan rutinitas pelatihan ulang SetFit memakai dataset terkurasi terbaru."""
    try:
        # Jalankan di baliknya latar menggunakan thread.
        # Catatan: idealnya gunakan subprocess atau pelayan eksternal pada skala produksi.
        from src.ml.train_setfit import train
        import threading
        
        # Mulai siklus di urat nadinya sehingga antarmuka tetap menyala (non-blocking)
        t = threading.Thread(target=train)
        t.start()
        
        return {"success": True, "message": "Prosedur pembelajaran ulang telah beroperasi di balik layar."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menginisiasi program latihan: {str(e)}")


from fastapi import UploadFile, File, Form

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("Indonesian"),
    run_analysis: bool = Form(True),
):
    """
    Ekstrak tulisan dari pelaporan oral (audio) memakai Qwen3-ASR-0.6B, 
    kemudian dapat disusul dengan pelacakan jejak ancaman.
    
    Berlaku bagi: wav, mp3, ogg, flac, m4a
    """
    allowed_types = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    import os
    ext = os.path.splitext(file.filename or "audio.wav")[1].lower()
    if ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Ektensi audio tak dikenal: {ext}. Yg diperbolehkan: {allowed_types}")

    audio_bytes = await file.read()

    try:
        from src.asr.transcriber import transcribe_bytes, transcribe_and_analyze
        import tempfile

        if run_analysis:
            # Pindahkan ke ruang sementara untuk ditinjau oleh transcriber utuh
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
        raise HTTPException(status_code=500, detail=f"Unit ASR tidak merespons: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
