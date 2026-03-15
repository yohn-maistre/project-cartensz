"""
Cartensz Intel Agent — Pendamping Intelijen ADK.

Menggunakan Google ADK (Agent Development Kit) bersenjatakan Gemini
untuk meladeni obrolan analis terkait data ancaman via bilah antarmuka Streamlit.

Alat Tempur (Tools):
  - search_threats(label, date_range, limit) → Kuari DuckDB
  - get_daily_stats() → Pantauan radar ancaman hari ini
  - get_trend(days) → Data fluktuasi multi-hari
  - deep_analyze(text) → Meletikkan The Radar (1 kali potong kuota LLM)
  - get_latest_triage(limit) → Sapuan The Sweep terbaru

Karakter Bot: Profesional. Tanpa basa-basi. Hemat kata tapi padat.
"""
import os
import sys
import asyncio
from datetime import datetime, timedelta

# pastikan akar pohon proyek bisa ditembus
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# baca tempelan .env buat kail GEMINI_API_KEY
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ADK melongok GOOGLE_API_KEY di bawah kap mesin google-genai, bukan GEMINI_API_KEY
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types


# ─── Senjata Pendamping Pembantu ───────────────────────────────────────────────────
# ADK nyomot otomatis skema parameter berdasarkan type hints dan string kutipan di bawah.

def search_threats(
    label: str = "",
    days_back: int = 7,
    limit: int = 20,
    keyword: str = "",
) -> dict:
    """Mengorek timbunan catatan ancaman dari brankas DuckDB.

    Args:
        label: Filter menurut tajuk (AMAN, WASPADA, TINGGI). Kosong = semua diangkut.
        days_back: Jarak waktu mundur harfiah. Setelan bawaan 7 hari.
        limit: Kouta tangkapan baris data tersaji. Setelan bawaan 20.
        keyword: Sandi lacak lema teks. Kosong = biarin.

    Returns:
        Kamus isi gerbong raupan dokumen dan total angkanya.
    """
    from src.db import get_db
    conn = get_db()

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
    """Membaca mesin pendaftaran DuckDB demi setoran angka laporan hari ini.

    Returns:
        Kamus sebaran muatan ancaman perhari.
    """
    from src.db import get_db
    conn = get_db()

    try:
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

        dist = {}
        for _, row in stats.iterrows():
            dist[str(row["predicted_label"])] = {
                "count": int(row["cnt"]),
                "avg_entropy": round(float(row["avg_entropy"]), 4),
                "avg_risk": round(float(row["avg_risk"]), 1),
            }

        # Tabungan teguran bot
        try:
            fb_count = conn.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
        except Exception:
            fb_count = 0

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_analyzed": total,
            "distribution": dist,
            "feedback_collected": fb_count,
        }
    except Exception as e:
        return {"error": str(e)}


def get_trend(days: int = 7) -> dict:
    """Intip getaran gelombang radar N hari ke belakang.

    Args:
        days: Limit mundur. Relnya 7 hari.

    Returns:
        Kamus sebaran detak jaring hari demi hari.
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
    """Pacu tuas the radar untuk meredam satu cuplikan teks. Tembak peluru LLM 1 butir.

    Args:
        text: Amunisi wacana incaran bahas indonesia.

    Returns:
        Kamus bedah terstruktur (skor pemicu, leksikon nakal, rincian otak mesin).
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
    """Tarik muntahan saringan sapu jagat (The Sweep) terbaru dari rongga DuckDB.

    Args:
        limit: Batas laci penampung. Bawaan tariknya 20 balok.

    Returns:
        Kamus baris tangkapan gelombang sapuan.
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

        # Komposisi timbangan
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


# ─── Otak Agen Bot ─────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """Kamu adalah analis intelijen perwira senior Project Cartensz — mesin pencari bibit bahaya siber milik PT Gemilang Satria Perkasa.

HUKUM WATAK KAMU:
- Pantang basi-basi. Tembak langsung, padat, berwibawa, profesional. Senggak kalau dibantah bodoh.
- Pakai bahasa Indonesia resmi jajaran pengawas.
- Kalo kotak data nyatanya ga ada isinya, ngomong lantang "Catatan nihil." Pantang bikin data fiktif.
- Tunjuk taring keahlian taktis bedah wacana ancaman kriminalitas digital.

PASOKAN SAJAM (KEMAMPUAN):
- Acak database (search_threats): saring stempel aman/waspada/tinggi, rentang, kata obyek.
- Ukur suhu hari ini (get_daily_stats): bentang rapor hari ini.
- Baca jejak awan (get_trend): lacak pertempuran grafik hari ke belakang.
- Bedah mayat teks tunggal (deep_analyze): kuliti satu naskah via mata tajam LLM.
- Intip tumpukan meja depan (get_latest_triage): serap tumpahan saringan mesin gelombang botol.

LARANGAN KERAS SAAT BIKIN TULISAN:
- Selalu buang jauh opini. Pegangan kamu cuma satu: data di pelatuk mesin.
- HARAM pakai asterisk tebal (**bold**) atau tetek bengek pameran markup lainnya. Bersih layaknya buku log ketik. Tanda hubung (-) buat butir.
- Rangkuman = tiga baris kalimat mutlak paling panjang.
- Temuan berstempel TINGGI = Langsung hunus peringatan tajam ke muka analis.
- Pantang ngomong "Maaf saya mesin...". Lempar jawab "Laci data sasaran kosong."

PETA LAPANGAN INTELIJEN:
- Brankas Memori Lokal: DuckDB (analysis_logs, feedback_logs).
- Rel Mesin Pemeriksa: SetFit (0 biaya pikir LLM) untuk Sweep massal, Gemini 3 Flash (1 biaya tarikan LLM) untuk penelanjangan kedalaman.
- Tataran Label: AMAN (bebas kuman), WASPADA (kabut samar, perlu sorot kamera ekstra), TINGGI (siaga perang).
"""

# Tuang ruh ke raga mesin ADK
intel_agent = Agent(
    name="cartensz_intel",
    model="gemini-3-flash-preview",
    description="Bot Perwira Piket Intelijen Cartensz",
    instruction=AGENT_SYSTEM_PROMPT,
    tools=[search_threats, get_daily_stats, get_trend, deep_analyze, get_latest_triage],
)

# ─── Pembungkus Otak Tetap (Singleton) ────────────────────────────────────

_session_service = InMemorySessionService()
_runner = None
_session_id = None


def _get_runner():
    """Pompa injeksi ADK Runner pertama manakala disentuh."""
    global _runner
    if _runner is None:
        _runner = Runner(
            agent=intel_agent,
            app_name="cartensz_intel",
            session_service=_session_service,
        )
    return _runner


async def _ensure_session(user_id: str = "analyst_default"):
    """Patrik jalur interkom antara agen bot dan masinis analis biar nggak melantur."""
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
    """Lempar kabel tanya ke bot lalu tunggui di sudut ruang sambil nyeruput kopi (async)."""
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

    return response_text or "Sambungan putus. Perwira bot membisu di radio."


def run_agent(user_message: str, user_id: str = "analyst_default") -> str:
    """Colokan asisten bot ke Streamlit. Putar baut pancing gelombang baru.
    
    Supaya mesin tenun Streamlit nggak mrepet "no current event loop",
    paksa bot buat mikir di ruangan gelap beda dimensi (background thread).
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
        return f"Konslet sumbu siber: {e}"


def reset_session():
    """Pretel sakelar radio ingatan ADK sampe kosong ampas (amnesia lobotomi)."""
    global _session_id
    _session_id = None


# ─── Bangku Uji Kasar ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Gelombang Uji Bot Cartensz Intel — Sandar Darat")
    print("Lempar komando ketik, 'exit' putus kabel.\n")
    while True:
        msg = input("Perwira Jaga: ").strip()
        if msg.lower() in ("exit", "quit", "q"):
            break
        reply = run_agent(msg)
        print(f"\n🛡️ Bot Intel: {reply}\n")
