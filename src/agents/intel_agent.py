"""
Cartensz Intel Agent — Asisten Intelijen Ancaman Berbasis Obrolan (ADK).

Menggunakan Google ADK (Agent Development Kit) dengan function-calling Gemini
agar analis dapat mengecek data ancaman melalui obrolan di sidebar Streamlit.

Alat:
  - search_threats(label, date_range, limit) → Kueri DuckDB
  - get_daily_stats() → distribusi ancaman hari ini
  - get_trend(days) → data tren ancaman
  - deep_analyze(text) → jalankan pipeline Radar penuh (1 call LLM)

Kepribadian: Analis intelijen profesional anti-basa-basi. Ringkas, bahasa Indonesia lugas.
"""
import os
import sys
import asyncio
from datetime import datetime, timedelta

# Pastikan root proyek dapat diimpor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Memuat .env untuk GEMINI_API_KEY
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ADK menggunakan google-genai yang mensyaratkan GOOGLE_API_KEY, bukan GEMINI_API_KEY
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types


# ─── Fungsi Perangkat ───────────────────────────────────────────────────
# Skema ADK diambil otomatis dari tipe data dan docstrings.

def search_threats(
    label: str = "",
    days_back: int = 7,
    limit: int = 20,
    keyword: str = "",
) -> dict:
    """Cari teks ancaman dari database DuckDB berdasarkan label, rentang waktu, atau keyword.

    Args:
        label: Filter berdasarkan label prediksi (AMAN, WASPADA, TINGGI). Kosongkan untuk semua.
        days_back: Berapa hari ke belakang untuk pencarian. Default 7 hari.
        limit: Jumlah maksimum hasil. Default 20.
        keyword: Kata kunci untuk pencarian dalam teks. Kosongkan untuk semua.

    Returns:
        Dictionary berisi daftar hasil threat dan jumlah total.
    """
    import duckdb
    from src.db import get_connection
    db_path = get_connection()

    conditions = ["1=1"]
    params = []

    if label and label.upper() in ("AMAN", "WASPADA", "TINGGI"):
        conditions.append("predicted_label = ?")
        params.append(label.upper())

    if days_back > 0:
        conditions.append(f"timestamp >= current_date - interval '{days_back}' day")

    if keyword:
        conditions.append("input_text ILIKE ?")
        params.append(f"%{keyword}%")

    where = " AND ".join(conditions)
    query = f"""
        SELECT id, timestamp, input_text, predicted_label, risk_score,
               confidence, entropy, pipeline_mode
        FROM analysis_logs
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT {limit}
    """

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            result = conn.execute(query, params).fetchdf()
            rows = []
            for _, row in result.iterrows():
                rows.append({
                    "timestamp": str(row["timestamp"]),
                    "text": str(row["input_text"])[:150],
                    "label": str(row["predicted_label"]),
                    "risk_score": int(row["risk_score"]) if row["risk_score"] else 0,
                    "confidence": str(row["confidence"]),
                    "entropy": round(float(row["entropy"]), 4) if row["entropy"] else 0,
                    "mode": str(row["pipeline_mode"]),
                })
            return {"count": len(rows), "results": rows}
    except Exception as e:
        return {"error": str(e), "count": 0, "results": []}


def get_daily_stats() -> dict:
    """Ambil statistik ancaman hari ini dari DuckDB.

    Returns:
        Dictionary berisi distribusi label hari ini plus total analisis.
    """
    from src.db import get_connection
    import duckdb
    db_path = get_connection()

    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            stats = conn.execute("""
                SELECT predicted_label, COUNT(*) as cnt,
                       AVG(entropy) as avg_entropy,
                       AVG(risk_score) as avg_risk
                FROM analysis_logs
                WHERE DATE(timestamp) = current_date
                GROUP BY predicted_label
            """).fetchdf()
    
            total = conn.execute("""
                SELECT COUNT(*) as total FROM analysis_logs
                WHERE DATE(timestamp) = current_date
            """).fetchone()[0]
            
            fb_count = 0
            try:
                fb_count = conn.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            except Exception:
                pass

        dist = {}
        for _, row in stats.iterrows():
            dist[str(row["predicted_label"])] = {
                "count": int(row["cnt"]),
                "avg_entropy": round(float(row["avg_entropy"]), 4),
                "avg_risk": round(float(row["avg_risk"]), 1),
            }

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_today": int(total),
            "distribution": dist,
            "feedback_collected": int(fb_count)
        }
    except Exception as e:
        return {"error": str(e)}


def get_trend(days: int = 7) -> dict:
    """Ambil tren ancaman harian selama N hari terakhir.

    Args:
        days: Jumlah hari ke belakang. Default 7.

    Returns:
        Dictionary berisi tren per hari dengan distribusi label.
    """
    from src.db import get_db
    conn = get_db()

    try:
        trend = conn.execute(f"""
            SELECT DATE(timestamp) as dt, predicted_label, COUNT(*) as cnt
            FROM analysis_logs
            WHERE timestamp >= current_date - interval '{days}' day
            GROUP BY dt, predicted_label
            ORDER BY dt ASC
        """).fetchdf()

        daily = {}
        for _, row in trend.iterrows():
            dt = str(row["dt"])
            if dt not in daily:
                daily[dt] = {"AMAN": 0, "WASPADA": 0, "TINGGI": 0}
            daily[dt][str(row["predicted_label"])] = int(row["cnt"])

        return {"days": days, "trend": daily}
    except Exception as e:
        return {"error": str(e)}


def deep_analyze(text: str) -> dict:
    """Jalankan analisis mendalam (The Radar) pada satu teks. Menggunakan 1 LLM call.

    Args:
        text: Teks berbahasa Indonesia yang akan dianalisis.

    Returns:
        Dictionary berisi hasil klasifikasi, risk score, sinyal, dan ringkasan.
    """
    from src.agents.orchestrator import run_pipeline

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                brief = pool.submit(asyncio.run, run_pipeline(text)).result()
        else:
            brief = asyncio.run(run_pipeline(text))

        return {
            "label": brief.classification.label,
            "confidence": brief.classification.confidence,
            "entropy": round(brief.classification.entropy, 4),
            "risk_score": brief.risk_score,
            "recommendation": brief.recommendation,
            "summary": brief.summary_narrative,
            "ambiguity": brief.ambiguity_notes,
            "signals": [
                {"type": s.signal_type, "text": s.extracted_text}
                for s in brief.signals_detected
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def get_latest_triage(limit: int = 20) -> dict:
    """Ambil hasil triage batch terbaru (The Sweep) dari DuckDB.

    Args:
        limit: Jumlah hasil terbaru yang dikembalikan. Default 20.

    Returns:
        Dictionary berisi daftar hasil triage terbaru beserta ringkasan distribusi.
    """
    from src.db import get_db
    conn = get_db()

    try:
        result = conn.execute(f"""
            SELECT input_text, predicted_label, confidence, entropy, 
                   is_ambiguous, timestamp
            FROM analysis_logs
            WHERE pipeline_mode = 'SWEEP'
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).fetchdf()

        rows = []
        for _, row in result.iterrows():
            rows.append({
                "text": str(row["input_text"])[:120],
                "label": str(row["predicted_label"]),
                "confidence": str(row["confidence"]),
                "entropy": round(float(row["entropy"]), 4) if row["entropy"] else 0,
                "ambiguous": bool(row["is_ambiguous"]),
            })

        # Summary stats
        dist = {}
        for r in rows:
            dist[r["label"]] = dist.get(r["label"], 0) + 1

        return {
            "count": len(rows),
            "distribution": dist,
            "results": rows,
        }
    except Exception as e:
        return {"error": str(e), "count": 0, "results": []}


# ─── Definisi Agen ─────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """Kamu adalah analis intelijen senior Project Cartensz — sistem klasifikasi ancaman narasi untuk PT Gemilang Satria Perkasa.

KEPRIBADIAN:
- Profesional, to-the-point, tanpa basa-basi
- Bahasa Indonesia formal tapi ringkas. Jawab langsung, jangan bertele-tele
- Kalau data kosong, bilang saja "Belum ada data." Jangan ciptakan data palsu
- Kamu paham soal keamanan, intelijen, dan analisis ancaman

KEMAMPUAN:
- Cari ancaman di database (search_threats): filter berdasarkan label, tanggal, keyword
- Lihat statistik harian (get_daily_stats): distribusi AMAN/WASPADA/TINGGI hari ini
- Lihat tren (get_trend): tren ancaman selama N hari terakhir
- Analisis mendalam (deep_analyze): klasifikasi lengkap 1 teks dengan LLM
- Lihat hasil triage terbaru (get_latest_triage): hasil batch triage dari The Sweep

FORMAT JAWABAN:
- Gunakan angka dan fakta, bukan opini
- JANGAN gunakan asterisk (**bold**) atau format markdown lain. Tulis teks biasa saja. Gunakan dash (-) untuk bullet
- Kalau diminta rangkuman, buat singkat 2-3 kalimat
- Kalau ada TINGGI, selalu flag langsung tanpa diminta
- Jangan pernah bilang "saya tidak bisa" — kalau data belum ada, bilang "belum ada data di database"

KONTEKS TEKNIS:
- Database: DuckDB dengan tabel analysis_logs dan feedback_logs
- Pipeline: SetFit (0 LLM) untuk batch triage, Gemini 3 Flash (1 LLM) untuk deep analysis
- Label: AMAN (aman), WASPADA (ambigu/perlu pantau), TINGGI (ancaman nyata)
"""

# Build ADK agent
intel_agent = Agent(
    name="cartensz_intel",
    model="gemini-3-flash-preview",
    description="Asisten intelijen ancaman Project Cartensz",
    instruction=AGENT_SYSTEM_PROMPT,
    tools=[search_threats, get_daily_stats, get_trend, deep_analyze, get_latest_triage],
)

# ─── Sesi & Runner (singleton) ────────────────────────────────────

_session_service = InMemorySessionService()
_runner = None
_session_id = None


def _get_runner():
    """Inisiasi tertunda pada Runner ADK."""
    global _runner
    if _runner is None:
        _runner = Runner(
            agent=intel_agent,
            app_name="cartensz_intel",
            session_service=_session_service,
        )
    return _runner


async def _ensure_session(user_id: str = "analyst_default"):
    """Validasi pembukaan sesi pengguna, buat jika perlu."""
    global _session_id
    runner = _get_runner()

    if _session_id is None:
        session = await _session_service.create_session(
            app_name="cartensz_intel",
            user_id=user_id,
        )
        _session_id = session.id

    return _session_id


async def _arun_agent(user_message: str, user_id: str = "analyst_default") -> str:
    """Jalankan agen secara asinkron dan ambil teks respons akhir."""
    runner = _get_runner()
    session_id = await _ensure_session(user_id)

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=user_message)]
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
            break

    return response_text or "Tidak ada respons dari agent."


def run_agent(user_message: str, user_id: str = "analyst_default") -> str:
    """Pembungkus sinkron untuk agen — digunakan oleh Streamlit.
    
    Selalu jalankan agen asinkron di event loop baru pada thread latar belakang
    untuk menghindari error 'no current event loop' di Streamlit.
    """
    import concurrent.futures

    def _run_in_new_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_arun_agent(user_message, user_id))
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_new_loop)
            return future.result(timeout=60)
    except Exception as e:
        return f"Error: {e}"


def reset_session():
    """Reset sesi ADK agar LLM melupakan konteks sebelumnya."""
    global _session_id
    _session_id = None


# ─── Tes Cepat ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Cartensz Intel Agent — Interactive Mode")
    print("Ketik pesan, atau 'exit' untuk keluar.\n")
    while True:
        msg = input("Anda: ").strip()
        if msg.lower() in ("exit", "quit", "q"):
            break
        reply = run_agent(msg)
        print(f"\n🛡️ Agent: {reply}\n")
