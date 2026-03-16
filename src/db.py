"""
Logger Analisis DuckDB Project Cartensz.
"""
import duckdb
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "cartensz.duckdb")

_conn = None

def get_db_path():
    os.makedirs(DATA_DIR, exist_ok=True)
    db_path = DB_PATH
    # Inisialisasi tabel jika belum ada
    with duckdb.connect(db_path) as conn:
        _init_db(conn)
    return db_path

get_connection = get_db_path

def _init_db(conn):
    """Buat tabel jika belum ada."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP,
            latency_ms DOUBLE,
            input_text TEXT,
            true_label VARCHAR DEFAULT NULL,
            predicted_label VARCHAR,
            risk_score INTEGER,
            confidence VARCHAR,
            entropy DOUBLE,
            is_ambiguous BOOLEAN,
            pipeline_mode VARCHAR -- 'RADAR' (1 LLM call) or 'SWEEP' (0 LLM call)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_logs (
            text_hash VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP,
            original_label VARCHAR,
            corrected_label VARCHAR,
            notes TEXT
        )
    """)

def log_analysis(
    doc_id: str,
    text: str,
    predicted_label: str,
    risk_score: int,
    confidence: str,
    entropy: float,
    is_ambiguous: bool,
    latency_ms: float,
    pipeline_mode: str = "RADAR"
):
    """Catat hasil analisis ke DuckDB."""
    db_path = get_db_path()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with duckdb.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO analysis_logs (
                id, timestamp, latency_ms, input_text,
                predicted_label, risk_score, confidence, entropy,
                is_ambiguous, pipeline_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
        """, (
            doc_id, current_time, latency_ms, text,
            predicted_label, risk_score, confidence, entropy,
            is_ambiguous, pipeline_mode
        ))

def save_feedback(text_hash: str, original_label: str, corrected_label: str, notes: str = None):
    """Simpan koreksi feedback analis."""
    db_path = get_db_path()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with duckdb.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO feedback_logs (text_hash, timestamp, original_label, corrected_label, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (text_hash) DO UPDATE SET 
                timestamp = EXCLUDED.timestamp,
                corrected_label = EXCLUDED.corrected_label,
                notes = EXCLUDED.notes
        """, (text_hash, current_time, original_label, corrected_label, notes or ""))

def fetch_feedback():
    """Ambil semua feedback (SetFit)."""
    db_path = get_db_path()
    with duckdb.connect(db_path) as conn:
        return conn.execute("SELECT * FROM feedback_logs").df()

def close_db():
    pass  # Tidak lagi diperlukan
