"""
Project Cartensz — Intelligence Command Center V4
"""
import streamlit as st
import sys
import os
import json
import re
import time
import hashlib
import pandas as pd
import numpy as np
import requests

# Pastikan root proyek dapat diimpor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Patch kompatibilitas setfit/transformers ---
import transformers.training_args as _ta
if not hasattr(_ta, "default_logdir"):
    import socket
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    def _default_logdir() -> str:
        current_time = _dt.now().strftime("%b%d_%H-%M-%S")
        return str(_Path("runs") / f"{current_time}_{socket.gethostname()}")
    _ta.default_logdir = _default_logdir
# --- Akhir patch ---

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Konfigurasi Halaman ──────────────────────────────────────────────
st.set_page_config(
    page_title="Project Cartensz — Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Kustom ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { background-color: #0b0510 !important; color: #f3e8ff; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #13091c !important; border-right: 1px solid #2b1744; }
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

    /* Intel Agent Chat — wadah chat yang bisa diskrol */
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

# ─── Fungsi Bantuan ───────────────────────────────────────────────────
def get_risk_color(score: int) -> str:
    if score <= 30: return "#10b981"
    elif score <= 60: return "#f59e0b"
    else: return "#ef4444"

def md_to_html(text: str) -> str:
    """Convert simple markdown (bold, bullets, newlines) to HTML for agent chat."""
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'\*(.+?)\*', r'<em>\1</em>', t)
    t = re.sub(r'^\s*[-•]\s+(.+)$', r'<span style="padding-left:0.8rem;">• \1</span>', t, flags=re.MULTILINE)
    t = t.replace("\n", "<br/>")
    return t

def render_signals(signals):
    if not signals:
        st.info("Tidak ada sinyal ancaman terdeteksi.")
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
            f'<strong>"{sig_text}"</strong> <span style="font-size:0.85em; color:#a78bfa;">({sig_sig})</span> — {sig_ctx}'
            f'</div>', unsafe_allow_html=True
        )

def render_deep_analysis(brief):
    """Tampilkan panel Radar untuk hasil analisis tunggal."""
    c = brief["classification"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_color = get_risk_color(brief["risk_score"])
        st.markdown(f'<div class="metric-card"><h3 style="color:{risk_color}">{brief["risk_score"]}</h3><p>Risk Score (0-100)</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{c["label"]}</h3><p>Keputusan (Fusi)</p></div>', unsafe_allow_html=True)
    with col3:
        rec_map = {"ARCHIVE": "ARSIP", "MONITOR": "PANTAU", "ESCALATE": "ESKALASI"}
        rec_label = rec_map.get(brief["recommendation"], brief["recommendation"])
        st.markdown(f'<div class="metric-card"><h3>{rec_label}</h3><p>Rekomendasi</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{c.get("confidence", "")}</h3><p>Confidence</p></div>', unsafe_allow_html=True)
    
    # Risk bar
    risk_pct = brief["risk_score"]
    bar_color = get_risk_color(risk_pct)
    st.markdown(f'<div class="risk-bar" style="background: linear-gradient(to right, {bar_color} {risk_pct}%, #1c0d28 {risk_pct}%);"></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown("#### 📝 Executive Summary")
    st.write(brief["summary_narrative"])
    
    # Probabilities & Uncertainty
    st.markdown("#### 📊 Probabilitas & Ketidakpastian")
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    probs = c.get("probabilities", {})
    prob_col1.metric("AMAN", f"{probs.get('AMAN', 0)*100:.1f}%")
    prob_col2.metric("WASPADA", f"{probs.get('WASPADA', 0)*100:.1f}%")
    prob_col3.metric("TINGGI", f"{probs.get('TINGGI', 0)*100:.1f}%")
    
    stats_col1, stats_col2 = st.columns(2)
    stats_col1.metric("Shannon Entropy", f"{c.get('entropy', 0):.4f}")
    stats_col2.metric("Prediction Set", str(c.get('prediction_set', [])))
    
    # Threat Signals
    st.markdown(f"#### 🚨 Sinyal Ancaman ({len(brief.get('signals_detected', []))} terdeteksi)")
    render_signals(brief.get("signals_detected", []))
    
    # Chain of Thought
    with st.expander("🧠 Penalaran AI (Chain-of-Thought)", expanded=False):
        st.write(c.get("reasoning", ""))
    
    # Ambiguity
    if brief.get("ambiguity_notes"):
        with st.expander("⚠️ Catatan Ambiguitas", expanded=False):
            st.warning(brief["ambiguity_notes"])

# ─── Panel Samping (Tentang + Input Data) ─────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="margin-bottom:0;"><span style="color:#a855f7;">🛡️</span> Project Cartensz</h2>
        <p style="color:#d8b4fe; font-size:0.85rem; margin-top:0.2rem;">Threat Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Cartensz", expanded=False):
        st.markdown("""
        **Apa ini?** Cartensz mendeteksi narasi ancaman berbahasa Indonesia 
        menggunakan pipeline NLP fusi khusus.
        
        **⚙️ Arsitektur:**
        1. **Preprocessor:** Normalisasi slang/code-switching.
        2. **Sinyal:** Ekstraksi Leksikon (eufemisme, CTA).
        3. **Ahli Lokal:** `NusaBERT` + `SetFit` (0 LLM cost).
        4. **Safety Net:** `Gemini 3 Flash` CoT reasoning (1 LLM call).
        
        **📊 Uncertainty Quantification:**
        - Shannon Entropy dari distribusi probabilitas.
        - Conformal Prediction Set (Adaptive).
        - Attention Weights dari NusaBERT.
        
        **🏷️ Label:**
        - 🟢 **AMAN** — Tidak ada ancaman.
        - 🟡 **WASPADA** — Ambigu, perlu pantauan.
        - 🔴 **TINGGI** — Ancaman nyata terdeteksi.
        """)
        
    st.markdown("<hr style='border-color:#2b1744;'>", unsafe_allow_html=True)
    st.markdown("### 📥 Masukkan Data")
    
    input_method = st.radio("Metode:", ["🕸️ OSINT Scraper", "✍️ Manual / Paste", "📂 Unggah File", "🎤 Audio (ASR)"], label_visibility="collapsed")
    
    # ─── OSINT Scraper ────────────────────────────────────────────────
    if "OSINT Scraper" in input_method:
        st.caption("Pilih sumber data OSINT:")
        use_reddit = st.checkbox("🔴 Reddit", value=True)
        use_rss = st.checkbox("📰 RSS (Detik, Tempo)", value=True)
        use_youtube = st.checkbox("🎬 YouTube", value=True)
        use_telegram = st.checkbox("✈️ Telegram", value=False, disabled=True, help="Belum dikonfigurasi")
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        use_custom = st.checkbox("🔗 Custom URL")
        if use_custom:
            custom_url = st.text_input("Masukkan URL:", placeholder="https://...")
        
        if st.button("🕸️ Jalankan Scraper", use_container_width=True):
            with st.spinner("Scraping dari sumber terpilih..."):
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
                    # Normalisasi: pastikan semua item adalah dict
                    normalized = []
                    for it in all_items:
                        if isinstance(it, dict) and it.get("text", "").strip():
                            # Pastikan key source dan url ada
                            it.setdefault("source", "OSINT")
                            it.setdefault("url", "")
                            normalized.append(it)
                    all_items = normalized
                    st.session_state["raw_triage_items"] = all_items
                    st.session_state["raw_triage_texts"] = [it["text"] for it in all_items]
                    st.success(f"Berhasil: {len(all_items)} teks.")
                except Exception as e:
                    st.error(f"Gagal: {e}")
                    
    # ─── Manual / Paste ───────────────────────────────────────────────
    elif "Manual" in input_method:
        example_choice = st.selectbox("Atau mulai dengan contoh:", [
            "(Ketik sendiri)",
            "Kemacetan terpantau di Jalan Sudirman pagi ini.",
            "Mereka dari barat mulai ikut campur urusan kita. Jangan biarkan mereka mengambil tanah leluhur.",
            "Siapkan senjata untuk jihad melawan musuh Allah. Tunggu aba-aba selanjutnya untuk menyerang malam ini.",
            "Demo buruh menuntut kenaikan UMR besok di depan gedung DPR."
        ])
        
        default_val = "" if example_choice == "(Ketik sendiri)" else example_choice
        manual_text = st.text_area("Masukkan teks:", value=default_val, height=120)
        
        if st.button("🔍 Analisis", use_container_width=True):
            lines = [l.strip() for l in manual_text.strip().split("\n") if l.strip()]
            if lines:
                if len(lines) == 1:
                    # Teks tunggal: picu dialog langsung
                    st.session_state["_dialog_text"] = lines[0]
                    st.rerun()
                else:
                    # Banyak baris: tambahkan ke antrean triage
                    items = [{"text": l, "source": "Manual", "url": ""} for l in lines]
                    st.session_state["raw_triage_items"] = items
                    st.session_state["raw_triage_texts"] = [it["text"] for it in items]
                    
    # ─── Unggah File ──────────────────────────────────────────────────
    elif "Unggah" in input_method:
        upload_file = st.file_uploader("CSV / TXT", type=["csv", "txt"])
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

    # ─── Upload Audio (ASR) ────────────────────────────────────────────
    elif "Audio" in input_method:
        st.caption("Transkripsi otomatis menggunakan Qwen3-ASR-0.6B")
        asr_tab_file, asr_tab_mic = st.tabs(["📂 Upload", "🎙️ Rekam"])

        audio_bytes = None
        audio_name = "recording.wav"

        with asr_tab_file:
            audio_file = st.file_uploader("Audio", type=["wav", "mp3", "ogg", "flac", "m4a", "webm"], label_visibility="collapsed")
            if audio_file:
                st.audio(audio_file)
                audio_bytes = audio_file.read()
                audio_name = audio_file.name

        with asr_tab_mic:
            mic_audio = st.audio_input("Klik untuk mulai merekam")
            if mic_audio:
                audio_bytes = mic_audio.read()
                audio_name = mic_audio.name

        if audio_bytes and st.button("🎤 Transkripsi & Analisis", type="primary", use_container_width=True):
            st.info("💡 **Catatan Utama ASR:** Proses transkripsi mungkin memakan waktu 1-3 menit. Jika ini pertama kalinya, server akan mengunduh model ~1.2GB di latar belakang.")
            with st.spinner("Mentranskripsi audio dengan API Cartensz..."):
                try:
                    files = {"file": (audio_name, audio_bytes, "audio/wav")}
                    data = {"language": "Indonesian", "run_analysis": False}
                    response = requests.post(f"{API_URL}/transcribe", files=files, data=data, timeout=300)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        if res_data.get("success"):
                            asr_result = res_data.get("transcription", {})
                            transcribed = asr_result.get("text", "")
                            st.success(f"Bahasa: {asr_result.get('language', 'Indonesian')}")
                            st.text_area("Hasil Transkripsi:", transcribed, height=100, disabled=True)
                            if transcribed.strip():
                                st.session_state["_dialog_text"] = transcribed
                        else:
                            st.error(f"ASR Error: {res_data.get('error', 'Unknown Error')}")
                    else:
                        st.error(f"Gagal menghubungi server ASR (Code: {response.status_code}). Detail: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("Waktu tunggu (timeout) habis. Server mungkin sedang mengunduh model secara lambat. Coba beberapa saat lagi.")
                except Exception as e:
                    st.error(f"Koneksi/ASR Error: {e}")

    # Jalankan Triage (di bawah input data)
    if st.session_state.get("raw_triage_texts") and not st.session_state.get("auto_radar_text"):
        n_texts = len(st.session_state["raw_triage_texts"])
        st.caption(f"📋 **{n_texts}** teks siap untuk triage.")
        if st.button("⚡ JALANKAN TRIAGE", type="primary", use_container_width=True):
            with st.spinner(f"Classifying {n_texts} texts via SetFit..."):
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
                            st.error(f"Batch Error chunk {i}: {resp.text}")
                    st.session_state["triage_results"] = all_briefs
                    st.success("Triage selesai!")
                except Exception as e:
                    st.error(f"API Error: {e}")

    # ─── Intel Agent Chat (ADK) ──────────────────────────────────────────
    st.markdown("<hr style='border-color:#2b1744; margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Intel Agent")

    # Inisialisasi riwayat chat
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    # Wadah chat yang bisa diskrol
    chat_html = '<div class="agent-chat-box" id="agent-chat-box">'
    if not st.session_state["agent_messages"]:
        chat_html += '<div style="color:#6b5b7b; font-size:0.8rem; text-align:center; padding:1rem;">Tanyakan data ancaman langsung.<br/>"Berapa TINGGI hari ini?" · "Rangkum triage terakhir"</div>'
    else:
        for msg in st.session_state["agent_messages"]:
            if msg["role"] == "user":
                chat_html += f'<div class="agent-msg agent-msg-user"><div class="agent-msg-label">🔍 Anda</div>{msg["content"]}</div>'
            else:
                content = md_to_html(msg["content"])
                chat_html += f'<div class="agent-msg agent-msg-bot"><div class="agent-msg-label">🛡️ Agent</div>{content}</div>'
    chat_html += '</div>'
    chat_html += '<script>var el=document.getElementById("agent-chat-box");if(el)el.scrollTop=el.scrollHeight;</script>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input chat — gunakan callback untuk hindari loop rerun
    def _on_agent_submit():
        msg = st.session_state.get("_agent_input", "").strip()
        if msg:
            st.session_state["agent_messages"].append({"role": "user", "content": msg})
            st.session_state["_agent_pending"] = msg
            st.session_state["_agent_input"] = ""

    st.text_input("💬", placeholder="Tanya agent...", key="_agent_input",
                  on_change=_on_agent_submit, label_visibility="collapsed")

    # Proses pesan agen yang tertunda
    if st.session_state.get("_agent_pending"):
        pending = st.session_state.pop("_agent_pending")
        # Indikator loading halus
        status_placeholder = st.empty()
        status_placeholder.markdown('<div style="color:#7c3aed; font-size:0.75rem;">⏳ Agent sedang berpikir...</div>', unsafe_allow_html=True)
        try:
            from src.agents.intel_agent import run_agent
            reply = run_agent(pending)
            st.session_state["agent_messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state["agent_messages"].append({"role": "assistant", "content": f"❌ Error: {e}"})
        status_placeholder.empty()
        st.rerun()

    # ─── Metrik Model ────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#2b1744; margin-top:1.5rem;'>", unsafe_allow_html=True)
    
    with st.expander("📊 Model & Retraining", expanded=False):
        report_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "reports", "setfit_training_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f: report = json.load(f)
            st.metric("Weighted F1", f"{report['results']['weighted_f1']:.4f}")
            st.metric("TINGGI Precision", f"{report['results']['tinggi_precision']:.4f}")
            st.metric("TINGGI Recall", f"{report['results']['tinggi_recall']:.4f}")
        else:
            st.caption("Report pelatihan belum tersedia.")
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        
        # Penghitung feedback
        try:
            import duckdb
            from src.db import get_connection as _get_db_path
            db_path = _get_db_path()
            with duckdb.connect(db_path, read_only=True) as _cdb:
                fb_count = _cdb.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
                st.caption(f"💬 Feedback terkumpul: **{fb_count}** · Threshold: **50**")
                can_retrain = fb_count >= 50
        except Exception:
            st.caption("💬 Feedback: DuckDB belum aktif.")
            can_retrain = False
        
        st.button("🔄 Latih Ulang Model", disabled=not can_retrain, 
                   help="Akan aktif setelah 50+ feedback terkumpul.", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# KONTEN UTAMA
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛡️ Project Cartensz</h1>
    <p style="margin:0; color:#e9d5ff; font-family:JetBrains Mono;">Threat Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# ─── Metrik Atas ──────────────────────────────────────────────────────
try:
    import duckdb
    from src.db import get_connection
    db_path = get_connection()
    with duckdb.connect(db_path, read_only=True) as conn:
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
        all_time_stats = conn.execute("""
            SELECT predicted_label, COUNT(*) as cnt
            FROM analysis_logs
            GROUP BY predicted_label
        """).df()
except Exception:
    trend_data = pd.DataFrame()
    today_stats = pd.DataFrame()
    all_time_stats = pd.DataFrame()

# Data hari ini
total_today = today_stats['cnt'].sum() if not today_stats.empty else 0
tinggi_today = today_stats[today_stats['predicted_label'] == 'TINGGI']['cnt'].sum() if not today_stats.empty else 0
waspada_today = today_stats[today_stats['predicted_label'] == 'WASPADA']['cnt'].sum() if not today_stats.empty else 0
aman_today = today_stats[today_stats['predicted_label'] == 'AMAN']['cnt'].sum() if not today_stats.empty else 0

# Data sepanjang waktu
total_all = all_time_stats['cnt'].sum() if not all_time_stats.empty else 0
tinggi_all = all_time_stats[all_time_stats['predicted_label'] == 'TINGGI']['cnt'].sum() if not all_time_stats.empty else 0
waspada_all = all_time_stats[all_time_stats['predicted_label'] == 'WASPADA']['cnt'].sum() if not all_time_stats.empty else 0
aman_all = all_time_stats[all_time_stats['predicted_label'] == 'AMAN']['cnt'].sum() if not all_time_stats.empty else 0

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric-card"><h3>{total_today}</h3><p>Hari Ini</p><span style="color:#a78bfa; font-size:0.7rem;">{total_all} total</span></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><h3 class="label-tinggi">{tinggi_today}</h3><p>🔴 TINGGI</p><span style="color:#a78bfa; font-size:0.7rem;">{tinggi_all} total</span></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><h3 class="label-waspada">{waspada_today}</h3><p>🟡 WASPADA</p><span style="color:#a78bfa; font-size:0.7rem;">{waspada_all} total</span></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card"><h3 class="label-aman">{aman_today}</h3><p>🟢 AMAN</p><span style="color:#a78bfa; font-size:0.7rem;">{aman_all} total</span></div>', unsafe_allow_html=True)

# ─── Ringkasan Eksekutif Batch ───────────────────────────────────────
if "triage_results" in st.session_state and st.session_state["triage_results"]:
    results = st.session_state["triage_results"]
    n_total = len(results)
    n_tinggi = sum(1 for r in results if r["label"] == "TINGGI")
    n_waspada = sum(1 for r in results if r["label"] == "WASPADA")
    n_aman = sum(1 for r in results if r["label"] == "AMAN")
    avg_entropy = sum(r.get("entropy", 0) for r in results) / max(n_total, 1)
    
    # Distribusi sumber
    items = st.session_state.get("raw_triage_items", [])
    source_counts = {}
    for it in items:
        src = it.get("source", "Unknown") if isinstance(it, dict) else "Unknown"
        source_counts[src] = source_counts.get(src, 0) + 1
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    summary_parts = [
        f"Dari **{n_total}** teks yang dianalisis:",
        f"🔴 **{n_tinggi}** TINGGI · 🟡 **{n_waspada}** WASPADA · 🟢 **{n_aman}** AMAN",
        f"Rata-rata entropy: **{avg_entropy:.4f}** · Sumber utama: {', '.join([f'{s[0]} ({s[1]})' for s in top_sources])}",
    ]
    
    if n_tinggi > 0:
        pct = n_tinggi / n_total * 100
        summary_parts.append(f"⚠️ **{pct:.1f}%** teks terdeteksi memiliki ancaman tinggi. Klik baris di bawah untuk Deep Analysis.")
    
    st.markdown(
        f'<div style="background:#1c0d28; border:1px solid #3b1f5b; border-radius:10px; padding:1rem; margin:1rem 0;">'
        f'<strong style="color:#d8b4fe;">📋 Ringkasan Eksekutif Batch</strong><br>'
        f'<span style="color:#e9d5ff; font-size:0.9rem;">{"<br>".join(summary_parts)}</span>'
        f'</div>', unsafe_allow_html=True
    )

# ─── Grafik ───────────────────────────────────────────────────────────
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
        st.info("Belum ada data historis.")

with chart_col2:
    st.markdown("#### 🌌 Peta Semantik Ancaman (PCA)")
    if "triage_results" in st.session_state and st.session_state["triage_results"]:
        results = st.session_state["triage_results"]
        # Cek ketersediaan embedding
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
            st.caption(f"PCA dari embedding NusaBERT 768d. Variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
            st.markdown('<p style="color:#a78bfa; font-size:0.75rem; margin-top:0.2rem;">Titik yang berdekatan = teks serupa secara semantik. Klaster warna yang terpisah menunjukkan model berhasil membedakan level ancaman.</p>', unsafe_allow_html=True)
        else:
            import altair as alt
            # Cadangan: gunakan entropy vs panjang teks
            df_pca = pd.DataFrame({
                'Entropy': [r.get("entropy", 0.1) for r in results],
                'Text Length': [len(r.get("text", "")) for r in results],
                'Label': [r["label"] for r in results]
            })
            color_scale = alt.Scale(domain=['AMAN', 'WASPADA', 'TINGGI'], range=['#10b981', '#f59e0b', '#ef4444'])
            scatter = alt.Chart(df_pca).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X('Entropy:Q'),
                y=alt.Y('Text Length:Q'),
                color=alt.Color('Label:N', scale=color_scale, legend=None),
                tooltip=['Label', 'Entropy']
            ).properties(height=220).interactive()
            st.altair_chart(scatter, use_container_width=True)
            st.caption("⚠️ Fallback: Entropy vs Panjang Teks (embeddings belum tersedia dari API).")
    else:
        st.info("Peta Semantik muncul setelah Triage dijalankan.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# DIALOG TEKS TUNGGAL
# ═══════════════════════════════════════════════════════════════════════
@st.dialog("🔍 Analisis Mendalam (Radar — 1 LLM Call)", width="large")
def _show_analysis_dialog(text):
    """Pop-up analisis mendalam."""
    st.info(f"**Teks:** _{text}_")
    with st.spinner("🚀 Memanggil SetFit + Gemini 3 Flash..."):
        try:
            response = requests.post(f"{API_URL}/analyze", json={"text": text}, timeout=60)
            response.raise_for_status()
            brief = response.json().get("brief")
            if brief:
                render_deep_analysis(brief)
                # Feedback di dalam dialog
                st.markdown("---")
                st.markdown("### 💬 Koreksi Analis")
                fb1, fb2 = st.columns(2)
                with fb1:
                    corrected = st.selectbox("Label:", ["(Sesuai)", "AMAN", "WASPADA", "TINGGI"], key="fb_dialog")
                with fb2:
                    notes = st.text_input("Catatan:", key="notes_dialog")
                if st.button("Simpan", key="save_dialog"):
                    if corrected != "(Sesuai)":
                        th = hashlib.sha256(text.encode()).hexdigest()[:16]
                        requests.post(f"{API_URL}/feedback", json={
                            "text_hash": th, "original_label": brief["classification"]["label"],
                            "corrected_label": corrected, "notes": notes
                        })
                        st.success("Tersimpan!")
        except Exception as e:
            st.error(f"Gagal: {e}")

# Picu dialog jika ada teks tertunda
if st.session_state.get("_dialog_text"):
    text_for_dialog = st.session_state.pop("_dialog_text")
    _show_analysis_dialog(text_for_dialog)

# ═══════════════════════════════════════════════════════════════════════
# TABEL TRIAGE
# ═══════════════════════════════════════════════════════════════════════
if "triage_results" in st.session_state and st.session_state["triage_results"]:
    st.markdown("### 🚦 Hasil Triage (The Sweep — 0 LLM)")
    st.caption("Pilih baris untuk meluncurkan Deep Analysis dengan LLM.")
    
    hide_aman = st.toggle("👁️ Sembunyikan AMAN", value=True)
    
    items = st.session_state.get("raw_triage_items", [])
    
    df_rows = []
    for i, r in enumerate(st.session_state["triage_results"]):
        if hide_aman and r["label"] == "AMAN":
            continue
        # Ambil sumber dari item jika ada
        source = items[i]["source"] if i < len(items) else "—"
        url = items[i].get("url", "") if i < len(items) else ""
        sigs = ", ".join([s.get("type", "?") for s in r.get("signal_highlights", [])])
        df_rows.append({
            "ID": i,
            "Label": r["label"],
            "Confidence": r.get("confidence", ""),
            "Entropy": round(r.get("entropy", 0), 4),
            "Signals": sigs if sigs else "—",
            "Source": source,
            "Text": r.get("text", "")[:100] + "..."
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
            sel_real_idx = int(df.iloc[sel_display_idx]["ID"])
            sel_text = st.session_state["triage_results"][sel_real_idx]["text"]
            sel_source = items[sel_real_idx]["source"] if sel_real_idx < len(items) else "—"
            sel_url = items[sel_real_idx].get("url", "") if sel_real_idx < len(items) else ""
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 🔍 Radar Panel — Deep Analysis")
            
            src_display = f"**Sumber:** {sel_source}"
            if sel_url:
                src_display += f" — [Buka link]({sel_url})"
            st.markdown(src_display)
            st.info(f"**Teks:** _{sel_text}_")
            
            with st.spinner("🚀 Memanggil LLM Director (Gemini 3 Flash)..."):
                try:
                    response = requests.post(f"{API_URL}/analyze", json={"text": sel_text}, timeout=60)
                    response.raise_for_status()
                    brief = response.json().get("brief")
                    
                    if brief:
                        render_deep_analysis(brief)
                        
                        # Feedback
                        st.markdown("---")
                        st.markdown("### 💬 Koreksi Analis")
                        fb1, fb2, fb3 = st.columns([1, 2, 1])
                        with fb1:
                            corrected = st.selectbox("Label:", ["(Sesuai)", "AMAN", "WASPADA", "TINGGI"], key="fb_batch")
                        with fb2:
                            notes = st.text_input("Catatan:", key="notes_batch")
                        with fb3:
                            st.write(""); st.write("")
                            if st.button("Simpan", key="save_batch"):
                                if corrected != "(Sesuai)":
                                    th = hashlib.sha256(sel_text.encode()).hexdigest()[:16]
                                    requests.post(f"{API_URL}/feedback", json={
                                        "text_hash": th, "original_label": brief["classification"]["label"],
                                        "corrected_label": corrected, "notes": notes
                                    })
                                    st.success("Tersimpan!")
                except Exception as e:
                    st.error(f"Gagal melakukan deep analysis: {e}")
    else:
        st.info("✨ Semua teks dilabeli AMAN. Matikan filter di atas untuk melihat semua.")

# ─── Analisis Terbaru ────────────────────────────────────────────────
if not ("triage_results" in st.session_state and st.session_state["triage_results"]):
    st.markdown("---")
    st.markdown("### 📜 Riwayat Analisis Terbaru")
    try:
        import duckdb
        from src.db import get_connection as _gc2
        db_path = _gc2()
        with duckdb.connect(db_path, read_only=True) as _hist_conn:
            history_df = _hist_conn.execute("""
                SELECT timestamp, predicted_label, confidence, entropy, 
                       SUBSTRING(input_text, 1, 120) || '...' as text_preview,
                       pipeline_mode
                FROM analysis_logs
                ORDER BY timestamp DESC
                LIMIT 25
            """).df()
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("👈 Masukkan data di panel kiri untuk memulai analisis.")
    except Exception:
        st.info("👈 Masukkan data di panel kiri untuk memulai analisis.")
