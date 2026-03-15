"""
Project Cartensz: Intelligence Command Center V4
"""
import streamlit as st
import sys
import os
import json
import time
import hashlib
import pandas as pd
import numpy as np
import requests

# Pastikan path utama proyek dapat diakses.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Tambalan sementara untuk setfit/transformers ---
import transformers.training_args as _ta
if not hasattr(_ta, "default_logdir"):
    import socket
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    def _default_logdir() -> str:
        current_time = _dt.now().strftime("%b%d_%H-%M-%S")
        return str(_Path("runs") / f"{current_time}_{socket.gethostname()}")
    _ta.default_logdir = _default_logdir
# --- Akhir tambalan ---

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Cartensz: Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Kustomisasi CSS (Mode Gelap Kontras Tinggi) ───────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { background-color: #0d0614 !important; color: #f3e8ff; font-family: 'Inter', sans-serif; }

    /* Perbaikan visibilitas bilah samping */
    [data-testid="stSidebar"] { background-color: #13091c !important; border-right: 1px solid #3b1f5b; padding-top: 10px; }
    [data-testid="stSidebar"] * { color: #d8b4fe; }
    [data-testid="stSidebar"] h1, h2, h3 { color: #faf5ff; }
    .stTextArea textarea, .stTextInput input { background-color: #1c0d28 !important; color: #f3e8ff !important; border: 1px solid #3b1f5b !important; border-radius: 8px; }
    .stTextArea textarea:focus, .stTextInput input:focus { border-color: #a855f7 !important; box-shadow: 0 0 0 1px #a855f7 !important; }
    ::placeholder { color: #a78bfa !important; opacity: 0.6 !important; }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { color: #d8b4fe !important; font-weight: 500; font-size: 1.05rem; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p { color: #f3e8ff !important; font-weight: 700; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #a855f7 !important; }
    
    .stButton>button { background: linear-gradient(135deg, #7e22ce, #a855f7) !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.3s ease; }
    .stButton>button:hover { background: linear-gradient(135deg, #9333ea, #c084fc) !important; box-shadow: 0 4px 14px 0 rgba(168, 85, 247, 0.39) !important; }
    
    .main-header { background: radial-gradient(circle at top left, #2b1744, #0b0510); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(168, 85, 247, 0.3); box-shadow: 0 10px 40px -10px rgba(0,0,0,0.8); }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #d8b4fe, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    .metric-card { background: linear-gradient(180deg, #1c0d28 0%, #13091c 100%); border-radius: 10px; padding: 1.2rem; text-align: center; border: 1px solid rgba(168, 85, 247, 0.4); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .metric-card h3 { margin: 0; font-size: 1.8rem; font-weight: 700; color: #f3e8ff; font-family: 'JetBrains Mono', monospace; }
    .metric-card p { margin: 0.2rem 0 0 0; color: #d8b4fe; font-size: 0.8rem; text-transform: uppercase; }
    
    .label-aman { color: #34d399; font-weight: 600; }
    .label-waspada { color: #fbbf24; font-weight: 600; }
    .label-tinggi { color: #ef4444; font-weight: 600; }
    
    .risk-bar { height: 8px; border-radius: 4px; margin-top: 0.5rem; }

    /* Kotak obrolan agen yang dapat digulir */
    .agent-chat-box {
        max-height: 350px;
        overflow-y: auto;
        padding: 0.5rem;
        border: 1px solid #2b1744;
        border-radius: 8px;
        background: #0f0718;
        margin-bottom: 0.5rem;
    }
    .agent-chat-box::-webkit-scrollbar { width: 4px; }
    .agent-chat-box::-webkit-scrollbar-track { background: transparent; }
    .agent-chat-box::-webkit-scrollbar-thumb { background: #3b1f5b; border-radius: 4px; }
    .agent-msg { padding: 0.5rem 0.7rem; border-radius: 8px; margin-bottom: 0.4rem; font-size: 0.85rem; line-height: 1.4; word-wrap: break-word; }
    .agent-msg-user { background: #2b1744; color: #e9d5ff; text-align: right; }
    .agent-msg-bot { background: #1c0d28; color: #d8b4fe; border: 1px solid #2b1744; }
    .agent-msg-label { font-size: 0.65rem; color: #7c3aed; margin-bottom: 2px; font-weight: 600; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ─── Fungsi Bantuan ─────────────────────────────────────────────────
def get_risk_color(score: int) -> str:
    if score <= 30: return "#10b981"
    elif score <= 60: return "#f59e0b"
    else: return "#ef4444"

def render_signals(signals):
    if not signals:
        st.info("Tidak ada sinyal ancaman.")
        return
    sig_icons = {"euphemism": "🔮", "call_to_action": "📢", "temporal_urgency": "⏰",
                 "code_switching": "🔀", "temporal": "⏰", "entity": "🏷️", "other": "⚡"}
    sig_colors = {"euphemism": "#c084fc", "call_to_action": "#f97316", "temporal_urgency": "#eab308",
                  "code_switching": "#38bdf8", "temporal": "#eab308", "entity": "#06b6d4", "other": "#a855f7"}
    for s in signals:
        if isinstance(s, dict):
            sig_type = str(s.get('signal_type', s.get('type', 'OTHER'))).lower()
            sig_text = str(s.get('extracted_text', s.get('text', '')))
            sig_sig = str(s.get('significance', 'LOW'))
            sig_ctx = str(s.get('context_explanation', s.get('reason', s.get('context', ''))))
        else:
            sig_type = str(getattr(s, 'signal_type', 'OTHER')).lower()
            sig_text = str(getattr(s, 'extracted_text', ''))
            sig_sig = str(getattr(s, 'significance', 'LOW'))
            sig_ctx = str(getattr(s, 'context_explanation', ''))
        icon = sig_icons.get(sig_type, "⚡")
        color = sig_colors.get(sig_type, "#a855f7")
        st.markdown(
            f'<div style="background:{color}11; border-left:3px solid {color}; padding:8px 12px; margin-bottom:8px; border-radius:4px;">'
            f'<span style="color:{color}; font-weight:bold; font-size:0.85em; font-family:JetBrains Mono;">{icon} {sig_type.upper()}</span><br>'
            f'<strong>"{sig_text}"</strong> <span style="font-size:0.85em; color:#a78bfa;">({sig_sig})</span> - {sig_ctx}'
            f'</div>', unsafe_allow_html=True
        )

def render_deep_analysis(brief):
    """Menampilkan hasil analisis mendalam (Radar)."""
    c = brief["classification"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_color = get_risk_color(brief["risk_score"])
        st.markdown(f'<div class="metric-card"><h3 style="color:{risk_color}">{brief["risk_score"]}</h3><p>Skor Risiko (0-100)</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{c["label"]}</h3><p>Keputusan</p></div>', unsafe_allow_html=True)
    with col3:
        rec_map = {"ARCHIVE": "ARSIP", "MONITOR": "PANTAU", "ESCALATE": "ESKALASI"}
        rec_label = rec_map.get(brief["recommendation"], brief["recommendation"])
        st.markdown(f'<div class="metric-card"><h3>{rec_label}</h3><p>Rekomendasi</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{c.get("confidence", "")}</h3><p>Keyakinan</p></div>', unsafe_allow_html=True)
    
    # Balok visual skor risiko
    risk_pct = brief["risk_score"]
    bar_color = get_risk_color(risk_pct)
    st.markdown(f'<div class="risk-bar" style="background: linear-gradient(to right, {bar_color} {risk_pct}%, #1c0d28 {risk_pct}%);"></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Ringkasan eksekutif
    st.markdown("#### 📝 Ringkasan Eksekutif")
    st.write(brief["summary_narrative"])
    
    # Kepercayaan model
    st.markdown("#### 📊 Probabilitas")
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    probs = c.get("probabilities", {})
    prob_col1.metric("AMAN", f"{probs.get('AMAN', 0)*100:.1f}%")
    prob_col2.metric("WASPADA", f"{probs.get('WASPADA', 0)*100:.1f}%")
    prob_col3.metric("TINGGI", f"{probs.get('TINGGI', 0)*100:.1f}%")
    
    stats_col1, stats_col2 = st.columns(2)
    stats_col1.metric("Entropi Shannon", f"{c.get('entropy', 0):.4f}")
    stats_col2.metric("Himpunan Prediksi", str(c.get('prediction_set', [])))
    
    # Sinyal temuan
    st.markdown(f"#### 🚨 Sinyal Ancaman ({len(brief.get('signals_detected', []))})")
    render_signals(brief.get("signals_detected", []))
    
    # Alasan AI
    with st.expander("🧠 Alasan Model (Chain-of-Thought)", expanded=False):
        st.write(c.get("reasoning", ""))
    
    # Ambiguitas
    if brief.get("ambiguity_notes"):
        with st.expander("⚠️ Catatan Ambiguitas", expanded=False):
            st.warning(brief["ambiguity_notes"])

# ─── Navigasi Samping ────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="margin-bottom:0;"><span style="color:#a855f7;">🛡️</span> Cartensz</h2>
        <p style="color:#d8b4fe; font-size:0.85rem; margin-top:0.2rem;">Pusat Komando Intelijen</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Sistem", expanded=False):
        st.markdown("""
        Cartensz mendeteksi ancaman berbahasa Indonesia menggunakan pipeline fusi.
        
        **⚙️ Arsitektur:**
        1. **Prapemrosesan:** Normalisasi teks.
        2. **Sinyal:** Ekstraksi fitur leksikal.
        3. **Filter Cepat:** NusaBERT + SetFit (0 biaya LLM).
        4. **Analisis Dalam:** Gemini 3 Flash (1 biaya LLM).
        
        **🏷️ Label:**
        - 🟢 **AMAN** (Aman)
        - 🟡 **WASPADA** (Pantau)
        - 🔴 **TINGGI** (Bahaya)
        """)
        
    st.markdown("<hr style='border-color:#2b1744;'>", unsafe_allow_html=True)
    st.markdown("### 📥 Sumber Data")
    
    input_method = st.radio("Metode Cerap Data:", ["🕸️ OSINT Scraper", "✍️ Manual", "📂 Unggah File", "🎤 Audio"], label_visibility="collapsed")
    
    # ─── Scraper OSINT ────────────────────────────────────
    if "OSINT Scraper" in input_method:
        st.caption("Target OSINT:")
        use_reddit = st.checkbox("🔴 Reddit", value=True)
        use_rss = st.checkbox("📰 RSS (Detik, Tempo)", value=True)
        use_youtube = st.checkbox("🎬 YouTube", value=True)
        use_telegram = st.checkbox("✈️ Telegram", value=False, disabled=True)
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        use_custom = st.checkbox("🔗 URL Kustom")
        if use_custom:
            custom_url = st.text_input("URL:", placeholder="https://...")
        
        if st.button("🕸️ Jalankan Scraper", use_container_width=True):
            with st.spinner("Mengunduh data..."):
                try:
                    from scripts.collector import (
                        scrape_reddit, scrape_rss, scrape_youtube,
                        REDDIT_TARGETS, RSS_TARGETS, YOUTUBE_TARGETS
                    )
                    all_items = []
                    if use_reddit:
                        for t in REDDIT_TARGETS: all_items.extend(scrape_reddit(t, debug=False))
                    if use_rss:
                        for t in RSS_TARGETS: all_items.extend(scrape_rss(t, debug=False))
                    if use_youtube:
                        for t in YOUTUBE_TARGETS: all_items.extend(scrape_youtube(t, debug=False))
                    # Normalisasi data
                    normalized = []
                    for it in all_items:
                        if isinstance(it, dict) and it.get("text", "").strip():
                            it.setdefault("source", "OSINT")
                            it.setdefault("url", "")
                            normalized.append(it)
                    all_items = normalized
                    st.session_state["raw_triage_items"] = all_items
                    st.session_state["raw_triage_texts"] = [it["text"] for it in all_items]
                    st.success(f"Berhasil: {len(all_items)} teks.")
                except Exception as e:
                    st.error(f"Gagal: {e}")
                    
    # ─── Teks Manual ──────────────────────────────────────────
    elif "Manual" in input_method:
        example_choice = st.selectbox("Pilih contoh:", [
            "(Ketik bebas)",
            "Kemacetan terpantau di Jalan Sudirman pagi ini.",
            "Mereka dari barat mulai ikut campur urusan kita. Jangan biarkan mereka mengambil tanah leluhur.",
            "Siapkan senjata untuk jihad melawan musuh Allah. Tunggu aba-aba selanjutnya untuk menyerang malam ini.",
            "Demo buruh menuntut kenaikan UMR besok di depan gedung DPR."
        ])
        
        default_val = "" if example_choice == "(Ketik bebas)" else example_choice
        manual_text = st.text_area("Masukkan teks:", value=default_val, height=120)
        
        if st.button("🔍 Analisis", use_container_width=True):
            lines = [l.strip() for l in manual_text.strip().split("\n") if l.strip()]
            if lines:
                items = [{"text": l, "source": "Manual", "url": ""} for l in lines]
                st.session_state["raw_triage_items"] = items
                st.session_state["raw_triage_texts"] = [it["text"] for it in items]
                # Modus analisis cepat jika hanya 1 kalimat
                if len(lines) == 1:
                    st.session_state["auto_radar_text"] = lines[0]
                    
    # ─── Unggah File ──────────────────────────────
    elif "Unggah" in input_method:
        upload_file = st.file_uploader("File CSV / TXT", type=["csv", "txt"])
        if upload_file and st.button("Proses File", use_container_width=True):
            if upload_file.name.endswith('.csv'):
                df = pd.read_csv(upload_file)
                texts = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                texts = [l.strip() for l in upload_file.read().decode('utf-8').split("\n") if l.strip()]
            texts = [t for t in texts if t.strip()]
            items = [{"text": t, "source": f"File: {upload_file.name}", "url": ""} for t in texts]
            st.session_state["raw_triage_items"] = items
            st.session_state["raw_triage_texts"] = [it["text"] for it in items]

    # ─── Audio ASR ────────────────────────────────────────────
    elif "Audio" in input_method:
        st.caption("Transkripsi suara otomatis (Qwen3-ASR)")
        audio_file = st.file_uploader("File Audio", type=["wav", "mp3", "ogg", "flac", "m4a", "webm"], label_visibility="collapsed")
        if audio_file:
            st.audio(audio_file)
            if st.button("🎤 Transkripsi & Analisis", type="primary", use_container_width=True):
                with st.spinner("Memproses audio..."):
                    try:
                        from src.asr.transcriber import transcribe_bytes
                        audio_bytes = audio_file.read()
                        asr_result = transcribe_bytes(audio_bytes, filename=audio_file.name)
                        transcribed = asr_result["text"]
                        st.success(f"Bahasa terdeteksi: {asr_result['language']}")
                        st.text_area("Hasil Transkripsi:", transcribed, height=100, disabled=True)
                        if transcribed.strip():
                            items = [{"text": transcribed, "source": f"Audio: {audio_file.name}", "url": ""}]
                            st.session_state["raw_triage_items"] = items
                            st.session_state["raw_triage_texts"] = [transcribed]
                            st.session_state["auto_radar_text"] = transcribed
                    except Exception as e:
                        st.error(f"Error proses audio: {e}")

    # Sapuan massal otomatis (Sweep)
    if st.session_state.get("raw_triage_texts") and not st.session_state.get("auto_radar_text"):
        n_texts = len(st.session_state["raw_triage_texts"])
        st.caption(f"📋 **{n_texts}** teks masuk antrean.")
        if st.button("⚡ JALANKAN TRIAGE", type="primary", use_container_width=True):
            with st.spinner(f"Memproses {n_texts} teks..."):
                try:
                    all_briefs = []
                    texts = st.session_state["raw_triage_texts"]
                    chunk_size = 20
                    for i in range(0, len(texts), chunk_size):
                        chunk = texts[i:i + chunk_size]
                        resp = requests.post(f"{API_URL}/batch", json={"texts": chunk})
                        if resp.status_code == 200:
                            all_briefs.extend(resp.json().get("briefs", []))
                        else:
                            st.error(f"Error API batch {i}: {resp.text}")
                    st.session_state["triage_results"] = all_briefs
                    st.success("Triage selesai.")
                except Exception as e:
                    st.error(f"Koneksi gagal: {e}")

    # ─── Chatbot Asisten (ADK Agent) ──────────────────────────────────
    st.markdown("<hr style='border-color:#2b1744; margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Asisten Analis")

    # Inisialisasi memori
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    # Area cuplikan obrolan
    chat_html = '<div class="agent-chat-box" id="agent-chat-box">'
    if not st.session_state["agent_messages"]:
        chat_html += '<div style="color:#6b5b7b; font-size:0.8rem; text-align:center; padding:1rem;">Tanyakan soal ancaman langsung di sini.<br/>Contoh: "Ringkasan hari ini"</div>'
    else:
        for msg in st.session_state["agent_messages"]:
            if msg["role"] == "user":
                chat_html += f'<div class="agent-msg agent-msg-user"><div class="agent-msg-label">🔍 Anda</div>{msg["content"]}</div>'
            else:
                content = msg["content"].replace("\n", "<br/>")
                chat_html += f'<div class="agent-msg agent-msg-bot"><div class="agent-msg-label">🛡️ Asisten</div>{content}</div>'
    chat_html += '</div>'
    chat_html += '<script>var el=document.getElementById("agent-chat-box");if(el)el.scrollTop=el.scrollHeight;</script>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input pesan
    def _on_agent_submit():
        msg = st.session_state.get("_agent_input", "").strip()
        if msg:
            st.session_state["agent_messages"].append({"role": "user", "content": msg})
            st.session_state["_agent_pending"] = msg
            st.session_state["_agent_input"] = ""

    st.text_input("💬", placeholder="Ketik perintah...", key="_agent_input",
                  on_change=_on_agent_submit, label_visibility="collapsed")

    # Eksekusi pesan tertunda
    if st.session_state.get("_agent_pending"):
        pending = st.session_state.pop("_agent_pending")
        status_placeholder = st.empty()
        status_placeholder.markdown('<div style="color:#7c3aed; font-size:0.75rem;">⏳ Memproses permintaan...</div>', unsafe_allow_html=True)
        try:
            from src.agents.intel_agent import run_agent
            reply = run_agent(pending)
            st.session_state["agent_messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state["agent_messages"].append({"role": "assistant", "content": f"❌ Error: {e}"})
        status_placeholder.empty()
        st.rerun()

    # Reset chat
    if st.session_state["agent_messages"]:
        if st.button("🗑️ Hapus riwayat", key="clear_chat", type="tertiary"):
            st.session_state["agent_messages"] = []
            try:
                from src.agents.intel_agent import reset_session
                reset_session()
            except Exception:
                pass
            st.rerun()

    # ─── Statistik Model ─────────────────
    st.markdown("<hr style='border-color:#2b1744; margin-top:1.5rem;'>", unsafe_allow_html=True)
    
    with st.expander("📊 Metrik Model", expanded=False):
        report_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "reports", "setfit_training_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f: report = json.load(f)
            st.metric("Akurasi F1 (Tertimbang)", f"{report['results']['weighted_f1']:.4f}")
            st.metric("Presisi TINGGI", f"{report['results']['tinggi_precision']:.4f}")
            st.metric("Recall TINGGI", f"{report['results']['tinggi_recall']:.4f}")
        else:
            st.caption("Hasil pelatihan belum ada.")
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        
        # Hitung umpan balik
        try:
            from src.db import get_connection as _gc
            _cdb = _gc()
            fb_count = _cdb.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            st.caption(f"💬 Jumlah Umpan Balik: **{fb_count}** / 50")
            can_retrain = fb_count >= 50
        except Exception:
            st.caption("💬 Database DuckDB mati.")
            can_retrain = False
        
        st.button("🔄 Pelatihan Ulang", disabled=not can_retrain, 
                   help="Membutuhkan minimal 50 umpan balik.", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# KONTEN UTAMA
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛡️ Pusat Komando Cartensz</h1>
    <p style="margin:0; color:#e9d5ff; font-family:JetBrains Mono;">Pemantauan Keamanan Intelijen Taktis</p>
</div>
""", unsafe_allow_html=True)

# ─── Skor Harian ─────────────────────────────────────────
try:
    from src.db import get_connection
    conn = get_connection()
    trend_data = conn.execute("""
        SELECT DATE(timestamp) as dt, predicted_label, COUNT(*) as cnt
        FROM analysis_logs
        WHERE timestamp >= current_date - interval '7' day
        GROUP BY dt, predicted_label
        ORDER BY dt ASC
    """).df()
    today_stats = conn.execute("""
        SELECT predicted_label, COUNT(*) as cnt
        FROM analysis_logs
        WHERE DATE(timestamp) = current_date
        GROUP BY predicted_label
    """).df()
except Exception:
    trend_data = pd.DataFrame()
    today_stats = pd.DataFrame()

col1, col2, col3, col4 = st.columns(4)
total_today = today_stats['cnt'].sum() if not today_stats.empty else 0
tinggi_today = today_stats[today_stats['predicted_label'] == 'TINGGI']['cnt'].sum() if not today_stats.empty else 0
waspada_today = today_stats[today_stats['predicted_label'] == 'WASPADA']['cnt'].sum() if not today_stats.empty else 0
aman_today = today_stats[today_stats['predicted_label'] == 'AMAN']['cnt'].sum() if not today_stats.empty else 0

with col1: st.markdown(f'<div class="metric-card"><h3>{total_today}</h3><p>Total Harian</p></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><h3 class="label-tinggi">{tinggi_today}</h3><p>🔴 TINGGI</p></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><h3 class="label-waspada">{waspada_today}</h3><p>🟡 WASPADA</p></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card"><h3 class="label-aman">{aman_today}</h3><p>🟢 AMAN</p></div>', unsafe_allow_html=True)

# ─── Hasil Borongan ──────────────────
if "triage_results" in st.session_state and st.session_state["triage_results"]:
    results = st.session_state["triage_results"]
    n_total = len(results)
    n_tinggi = sum(1 for r in results if r["label"] == "TINGGI")
    n_waspada = sum(1 for r in results if r["label"] == "WASPADA")
    n_aman = sum(1 for r in results if r["label"] == "AMAN")
    avg_entropy = sum(r.get("entropy", 0) for r in results) / max(n_total, 1)
    
    # Hitung sumber
    items = st.session_state.get("raw_triage_items", [])
    source_counts = {}
    for it in items:
        src = it.get("source", "Tidak Diketahui") if isinstance(it, dict) else "Tidak Diketahui"
        source_counts[src] = source_counts.get(src, 0) + 1
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    summary_parts = [
        f"Analisis Selesai Atas **{n_total}** Teks:",
        f"🔴 **{n_tinggi}** TINGGI | 🟡 **{n_waspada}** WASPADA | 🟢 **{n_aman}** AMAN",
        f"Rata-rata entropi model: **{avg_entropy:.4f}** | Sumber terbanyak: {', '.join([f'{s[0]} ({s[1]})' for s in top_sources])}",
    ]
    
    if n_tinggi > 0:
        pct = n_tinggi / n_total * 100
        summary_parts.append(f"⚠️ **{pct:.1f}%** ancaman tinggi terdeteksi. Silakan tinjau baris hasil di bawah.")
    
    st.markdown(
        f'<div style="background:#1c0d28; border:1px solid #3b1f5b; border-radius:10px; padding:1rem; margin:1rem 0;">'
        f'<strong style="color:#d8b4fe;">📋 Ringkasan Analisis Batch</strong><br>'
        f'<span style="color:#e9d5ff; font-size:0.9rem;">{"<br>".join(summary_parts)}</span>'
        f'</div>', unsafe_allow_html=True
    )

# ─── Peta Visual ──────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 📈 Tren Ancaman (7 Hari)")
    if not trend_data.empty:
        chart_data = trend_data.pivot(index='dt', columns='predicted_label', values='cnt').fillna(0)
        for c in ['AMAN', 'WASPADA', 'TINGGI']:
            if c not in chart_data.columns: chart_data[c] = 0
        st.area_chart(chart_data[['TINGGI', 'WASPADA', 'AMAN']], height=220, color=["#ef4444", "#f59e0b", "#10b981"])
    else:
        st.info("Log riwayat data masih bersih.")

with chart_col2:
    st.markdown("#### 🌌 Klaster Semantik (PCA)")
    if "triage_results" in st.session_state and st.session_state["triage_results"]:
        results = st.session_state["triage_results"]
        # Pastikan data vektor tersimpan
        has_embeddings = any(r.get("embedding") for r in results)
        
        if has_embeddings:
            from sklearn.decomposition import PCA
            import altair as alt
            
            emb_data = [(r["embedding"], r["label"]) for r in results if r.get("embedding")]
            embeddings_matrix = np.array([e[0] for e in emb_data])
            labels = [e[1] for e in emb_data]
            
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings_matrix)
            
            df_pca = pd.DataFrame({
                'PC1': coords[:, 0],
                'PC2': coords[:, 1],
                'Label': labels
            })
            color_scale = alt.Scale(domain=['AMAN', 'WASPADA', 'TINGGI'], range=['#10b981', '#f59e0b', '#ef4444'])
            scatter = alt.Chart(df_pca).mark_circle(size=70, opacity=0.8).encode(
                x=alt.X('PC1:Q', title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'),
                y=alt.Y('PC2:Q', title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'),
                color=alt.Color('Label:N', scale=color_scale, legend=None),
                tooltip=['Label']
            ).properties(height=220).interactive()
            st.altair_chart(scatter, use_container_width=True)
            st.caption(f"Klaster vektor semantik. Akurasi proyeksi PCA: {sum(pca.explained_variance_ratio_)*100:.1f}%")
        else:
            import altair as alt
            # Fallback manual
            df_pca = pd.DataFrame({
                'Tingkat Ambiguitas (Entropi)': [r.get("entropy", 0.1) for r in results],
                'Panjang Teks': [len(r.get("text", "")) for r in results],
                'Label': [r["label"] for r in results]
            })
            color_scale = alt.Scale(domain=['AMAN', 'WASPADA', 'TINGGI'], range=['#10b981', '#f59e0b', '#ef4444'])
            scatter = alt.Chart(df_pca).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X('Tingkat Ambiguitas (Entropi):Q'),
                y=alt.Y('Panjang Teks:Q'),
                color=alt.Color('Label:N', scale=color_scale, legend=None),
                tooltip=['Label', 'Tingkat Ambiguitas (Entropi)']
            ).properties(height=220).interactive()
            st.altair_chart(scatter, use_container_width=True)
            st.caption("⚠️ Metode Alternatif: Entropi vs Panjang (Embedding vektor belum di-generate API).")
    else:
        st.info("Peta klaster akan dimuat setelah analisis berjalan.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# ANALISIS LLM TUNGGAL
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.get("auto_radar_text"):
    single_text = st.session_state["auto_radar_text"]
    st.markdown("## 🔍 Analisis Terperinci LLM")
    st.info(f"**Teks Sasaran:** _{single_text}_")
    
    with st.spinner("🚀 Mnjalankan SetFit dan Gemini Flash..."):
        try:
            response = requests.post(f"{API_URL}/analyze", json={"text": single_text}, timeout=60)
            response.raise_for_status()
            brief = response.json().get("brief")
            
            if brief:
                render_deep_analysis(brief)
                
                # Umpan balik klasifikasi tunggal
                st.markdown("---")
                st.markdown("### 💬 Koreksi Pakar")
                fb1, fb2, fb3 = st.columns([1, 2, 1])
                with fb1:
                    corrected = st.selectbox("Keputusan Akhir:", ["(Sesuai)", "AMAN", "WASPADA", "TINGGI"], key="fb_single")
                with fb2:
                    notes = st.text_input("Catatan Analis:", key="notes_single")
                with fb3:
                    st.write(""); st.write("")
                    if st.button("Simpan Umpan Balik", key="save_single"):
                        if corrected != "(Sesuai)":
                            th = hashlib.sha256(single_text.encode()).hexdigest()[:16]
                            requests.post(f"{API_URL}/feedback", json={
                                "text_hash": th, "original_label": brief["classification"]["label"],
                                "corrected_label": corrected, "notes": notes
                            })
                            st.success("Tersimpan dalam database.")
        except Exception as e:
            st.error(f"Gagal memproses LLM: {e}")
    
    # Hapus jejak
    if st.button("🔄 Lakukan Ulang Analisis Baru", use_container_width=True):
        del st.session_state["auto_radar_text"]
        if "raw_triage_texts" in st.session_state:
            del st.session_state["raw_triage_texts"]
        if "raw_triage_items" in st.session_state:
            del st.session_state["raw_triage_items"]
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# TABEL TRIAGE MULTI-BARIS
# ═══════════════════════════════════════════════════════════════════════
elif "triage_results" in st.session_state and st.session_state["triage_results"]:
    st.markdown("### 🚦 Hasil Analisis Batch (SetFit / Nol LLM)")
    st.caption("Pilih salah satu hasil untuk membedah teks dengan teliti menggunakan LLM.")
    
    hide_aman = st.toggle("👁️ Kecualikan status AMAN", value=True)
    
    items = st.session_state.get("raw_triage_items", [])
    
    df_rows = []
    for i, r in enumerate(st.session_state["triage_results"]):
        if hide_aman and r["label"] == "AMAN":
            continue
        # lacak sumber teks
        source = items[i]["source"] if i < len(items) else "—"
        url = items[i].get("url", "") if i < len(items) else ""
        sigs = ", ".join([s.get("type", "?") for s in r.get("signal_highlights", [])])
        df_rows.append({
            "No_Identitas": i,
            "Label Akhir": r["label"],
            "Keyakinan": r.get("confidence", ""),
            "Entropi": round(r.get("entropy", 0), 4),
            "Fitur Menarik": sigs if sigs else "—",
            "Sumber Data": source,
            "Cuplikan": r.get("text", "")[:100] + "..."
        })
    
    df = pd.DataFrame(df_rows)
    
    if not df.empty:
        event = st.dataframe(
            df, use_container_width=True, hide_index=True,
            selection_mode="single-row", on_select="rerun"
        )
        
        selected_rows = event.selection.rows if hasattr(event, "selection") else []
        
        if selected_rows:
            sel_display_idx = selected_rows[0]
            sel_real_idx = int(df.iloc[sel_display_idx]["No_Identitas"])
            sel_text = st.session_state["triage_results"][sel_real_idx]["text"]
            sel_source = items[sel_real_idx]["source"] if sel_real_idx < len(items) else "—"
            sel_url = items[sel_real_idx].get("url", "") if sel_real_idx < len(items) else ""
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 🔍 Detail Lanjutan LLM")
            
            src_display = f"**Informasi Sumber:** {sel_source}"
            if sel_url:
                src_display += f" — [Buka halaman sumber]({sel_url})"
            st.markdown(src_display)
            st.info(f"**Teks Asli:** _{sel_text}_")
            
            with st.spinner("🚀 Menghubungi API Gemini untuk analisis terperinci..."):
                try:
                    response = requests.post(f"{API_URL}/analyze", json={"text": sel_text}, timeout=60)
                    response.raise_for_status()
                    brief = response.json().get("brief")
                    
                    if brief:
                        render_deep_analysis(brief)
                        
                        # Kotak Umpan Balik Batch
                        st.markdown("---")
                        st.markdown("### 💬 Edit Klasifikasi Otomatis (Umpan Balik)")
                        fb1, fb2, fb3 = st.columns([1, 2, 1])
                        with fb1:
                            corrected = st.selectbox("Tindakan Korektif:", ["(Valid)", "AMAN", "WASPADA", "TINGGI"], key="fb_batch")
                        with fb2:
                            notes = st.text_input("Saran Koreksi:", key="notes_batch")
                        with fb3:
                            st.write(""); st.write("")
                            if st.button("Kirim Rekaman", key="save_batch"):
                                if corrected != "(Valid)":
                                    th = hashlib.sha256(sel_text.encode()).hexdigest()[:16]
                                    requests.post(f"{API_URL}/feedback", json={
                                        "text_hash": th, "original_label": brief["classification"]["label"],
                                        "corrected_label": corrected, "notes": notes
                                    })
                                    st.success("Tercatat dalam memori pembaruan.")
                except Exception as e:
                    st.error(f"Komunikasi LLM gagal terhubung: {e}")
    else:
        st.info("✨ Daftar bersih. Semua teks terpindai tergolong AMAN. Buang filter di atas apabila ingin melihat hasil utuh.")
else:
    st.info("👈 Masukkan data ancaman melaui panel di sisi kiri layar untuk memicu proses pemindaian mendalam.")
