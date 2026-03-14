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

# Pastikan path utama proyek dapat diakses.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Tambalan sementara untuk kompatibilitas setfit/transformers ---
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

# ─── Kustomisasi CSS ───────────────────────────────────────────────────────
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

    /* Kotak Obrolan Agen Intelijen - kontainer dapat digulir dan tinggi tetap */
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

def md_to_html(text: str) -> str:
    """Mengonversi markdown sederhana menjadi HTML untuk obrolan agen."""
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
    """Menampilkan panel Radar secara penuh untuk satu hasil analisis."""
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
    
    # Indikator skor risiko visual
    risk_pct = brief["risk_score"]
    bar_color = get_risk_color(risk_pct)
    st.markdown(f'<div class="risk-bar" style="background: linear-gradient(to right, {bar_color} {risk_pct}%, #1c0d28 {risk_pct}%);"></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Ringkasan Eksekutif
    st.markdown("#### 📝 Laporan Eksekutif")
    st.write(brief["summary_narrative"])
    
    # Probabilitas dan Ketidakpastian
    st.markdown("#### 📊 Probabilitas & Ketidakpastian")
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    probs = c.get("probabilities", {})
    prob_col1.metric("AMAN", f"{probs.get('AMAN', 0)*100:.1f}%")
    prob_col2.metric("WASPADA", f"{probs.get('WASPADA', 0)*100:.1f}%")
    prob_col3.metric("TINGGI", f"{probs.get('TINGGI', 0)*100:.1f}%")
    
    stats_col1, stats_col2 = st.columns(2)
    stats_col1.metric("Entropi Shannon", f"{c.get('entropy', 0):.4f}")
    stats_col2.metric("Himpunan Prediksi", str(c.get('prediction_set', [])))
    
    # Sinyal Ancaman
    st.markdown(f"#### 🚨 Sinyal Ancaman ({len(brief.get('signals_detected', []))} terdeteksi)")
    render_signals(brief.get("signals_detected", []))
    
    # Alur Logika Keputusan
    with st.expander("🧠 Penalaran AI (Chain-of-Thought)", expanded=False):
        st.write(c.get("reasoning", ""))
    
    # Ambiguitas
    if brief.get("ambiguity_notes"):
        with st.expander("⚠️ Catatan Ambiguitas", expanded=False):
            st.warning(brief["ambiguity_notes"])

# ─── Panel Navigasi Samping ────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="margin-bottom:0;"><span style="color:#a855f7;">🛡️</span> Project Cartensz</h2>
        <p style="color:#d8b4fe; font-size:0.85rem; margin-top:0.2rem;">Pusat Komando Intelijen</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Sistem", expanded=False):
        st.markdown("""
        Cartensz mendeteksi narasi ancaman berbahasa Indonesia 
        menggunakan pipeline NLP berbasis fusi.
        
        **⚙️ Arsitektur:**
        1. **Prapemrosesan:** Normalisasi teks.
        2. **Sinyal:** Ekstraksi Leksikon (eufemisme, CTA).
        3. **Filter Cepat:** `NusaBERT` + `SetFit` (bebas biaya LLM).
        4. **Analisis Dalam:** Penalaran `Gemini 3 Flash` (1 panggilan LLM).
        
        **📊 Kuantifikasi Keraguan:**
        - Entropi Shannon dari distribusi prediksi.
        - Himpunan Prediksi Konformal adaptif.
        - Bobot Perhatian dari NusaBERT.
        
        **🏷️ Kategori Status:**
        - 🟢 **AMAN** — Aman.
        - 🟡 **WASPADA** — Ambigu, memerlukan pemantauan.
        - 🔴 **TINGGI** — Terdapat indikasi teror.
        """)
        
    st.markdown("<hr style='border-color:#2b1744;'>", unsafe_allow_html=True)
    st.markdown("### 📥 Sumber Masukan")
    
    input_method = st.radio("Metode:", ["🕸️ OSINT Scraper", "✍️ Manual / Paste", "📂 Unggah File", "🎤 Audio (ASR)"], label_visibility="collapsed")
    
    # ─── Pengumpulan OSINT Otomatis ────────────────────────────────────────────────
    if "OSINT Scraper" in input_method:
        st.caption("Pilih sasaran:")
        use_reddit = st.checkbox("🔴 Reddit", value=True)
        use_rss = st.checkbox("📰 RSS (Detik, Tempo)", value=True)
        use_youtube = st.checkbox("🎬 YouTube", value=True)
        use_telegram = st.checkbox("✈️ Telegram", value=False, disabled=True, help="Belum diaktifkan")
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        use_custom = st.checkbox("🔗 Tautan Spesifik")
        if use_custom:
            custom_url = st.text_input("Tautan (URL):", placeholder="https://...")
        
        if st.button("🕸️ Eksekusi Scraper", use_container_width=True):
            with st.spinner("Sedang mengekstraksi data..."):
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
                    # Normalisasi format input menjadi dictionary standar
                    normalized = []
                    for it in all_items:
                        if isinstance(it, dict) and it.get("text", "").strip():
                            it.setdefault("source", "OSINT")
                            it.setdefault("url", "")
                            normalized.append(it)
                    all_items = normalized
                    st.session_state["raw_triage_items"] = all_items
                    st.session_state["raw_triage_texts"] = [it["text"] for it in all_items]
                    st.success(f"Terekstrak: {len(all_items)} dokumen.")
                except Exception as e:
                    st.error(f"Kesalahan koneksi: {e}")
                    
    # ─── Pemasukan Teks Manual ───────────────────────────────────────────────
    elif "Manual" in input_method:
        example_choice = st.selectbox("Contoh kalimat:", [
            "(Ketik sendiri)",
            "Kemacetan terpantau di Jalan Sudirman pagi ini.",
            "Mereka dari barat mulai ikut campur urusan kita. Jangan biarkan mereka mengambil tanah leluhur.",
            "Siapkan senjata untuk jihad melawan musuh Allah. Tunggu aba-aba selanjutnya untuk menyerang malam ini.",
            "Demo buruh menuntut kenaikan UMR besok di depan gedung DPR."
        ])
        
        default_val = "" if example_choice == "(Ketik sendiri)" else example_choice
        manual_text = st.text_area("Konten:", value=default_val, height=120)
        
        if st.button("🔍 Lakukan Analisis", use_container_width=True):
            lines = [l.strip() for l in manual_text.strip().split("\n") if l.strip()]
            if lines:
                if len(lines) == 1:
                    # Baris tunggal: luncurkan jendela dialog
                    st.session_state["_dialog_text"] = lines[0]
                    st.rerun()
                else:
                    # Baris ganda: masukkan ke dalam antrean klasifikasi massal
                    items = [{"text": l, "source": "Manual", "url": ""} for l in lines]
                    st.session_state["raw_triage_items"] = items
                    st.session_state["raw_triage_texts"] = [it["text"] for it in items]
                    
    # ─── Pengolahan Melalui Berkas ──────────────────────────────────────────────────
    elif "Unggah" in input_method:
        upload_file = st.file_uploader("Format CSV / TXT", type=["csv", "txt"])
        if upload_file and st.button("Mulai Proses File", use_container_width=True):
            if upload_file.name.endswith('.csv'):
                df = pd.read_csv(upload_file)
                texts = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                texts = [l.strip() for l in upload_file.read().decode('utf-8').split("\n") if l.strip()]
            texts = [t for t in texts if t.strip()]
            items = [{"text": t, "source": f"File Import: {upload_file.name}", "url": ""} for t in texts]
            st.session_state["raw_triage_items"] = items
            st.session_state["raw_triage_texts"] = [it["text"] for it in items]

    # ─── Perekaman dan Transkripsi Suara ────────────────────────────────────────────
    elif "Audio" in input_method:
        st.caption("Penerjemah audio bahasa melalui Qwen3-ASR-0.6B")
        asr_tab_file, asr_tab_mic = st.tabs(["📂 Pilih File", "🎙️ Rekam Langsung"])

        audio_bytes = None
        audio_name = "recording.wav"

        with asr_tab_file:
            audio_file = st.file_uploader("Berkas Suara", type=["wav", "mp3", "ogg", "flac", "m4a", "webm"], label_visibility="collapsed")
            if audio_file:
                st.audio(audio_file)
                audio_bytes = audio_file.read()
                audio_name = audio_file.name

        with asr_tab_mic:
            mic_audio = st.audio_input("Gunakan mikrofon untuk berbicara")
            if mic_audio:
                audio_bytes = mic_audio.read()
                audio_name = "mic_recording.wav"

        if audio_bytes and st.button("🎤 Lakukan Transkripsi dan Analisis", type="primary", use_container_width=True):
            with st.spinner("Mengubah suara menjadi teks..."):
                try:
                    from src.asr.transcriber import transcribe_bytes
                    asr_result = transcribe_bytes(audio_bytes, filename=audio_name)
                    transcribed = asr_result["text"]
                    st.success(f"Logat terdeteksi: {asr_result['language']}")
                    st.text_area("Cuplikan Naskah:", transcribed, height=100, disabled=True)
                    if transcribed.strip():
                        st.session_state["_dialog_text"] = transcribed
                except Exception as e:
                    st.error(f"Gagal melakukan proses asr: {e}")

    # Tombol Eksekusi Triage Massal (tampil tepat di bawah blok data masukan)
    if st.session_state.get("raw_triage_texts") and not st.session_state.get("auto_radar_text"):
        n_texts = len(st.session_state["raw_triage_texts"])
        st.caption(f"📋 Terdeteksi **{n_texts}** item di memori tunggu.")
        if st.button("⚡ INISIASI KATEGORISASI (TRIAGE)", type="primary", use_container_width=True):
            with st.spinner(f"Sortir {n_texts} item dengan mesin SetFit..."):
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
                            st.error(f"Kegagalan blok penyortiran indeks {i}: {resp.text}")
                    st.session_state["triage_results"] = all_briefs
                    st.success("Prioritas kategorisasi berhasil disesuaikan!")
                except Exception as e:
                    st.error(f"Komunikasi API tumbang: {e}")

    # ─── Chatbot Analis Keamanan Siber (ADK) ──────────────────────────────────────────
    st.markdown("<hr style='border-color:#2b1744; margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Agen Pendamping")

    # Penyusunan memori sesi obrolan awal
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    # Kontainer pembungkus ruang ketik
    chat_html = '<div class="agent-chat-box" id="agent-chat-box">'
    if not st.session_state["agent_messages"]:
        chat_html += '<div style="color:#6b5b7b; font-size:0.8rem; text-align:center; padding:1rem;">Ketuk agen untuk asisten intelijen.<br/>Contoh: "Berapa banyak risiko TINGGI periode ini?"</div>'
    else:
        for msg in st.session_state["agent_messages"]:
            if msg["role"] == "user":
                chat_html += f'<div class="agent-msg agent-msg-user"><div class="agent-msg-label">🔍 Operator</div>{msg["content"]}</div>'
            else:
                content = md_to_html(msg["content"])
                chat_html += f'<div class="agent-msg agent-msg-bot"><div class="agent-msg-label">🛡️ Cartensz Agent</div>{content}</div>'
    chat_html += '</div>'
    chat_html += '<script>var el=document.getElementById("agent-chat-box");if(el)el.scrollTop=el.scrollHeight;</script>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Menangani kiriman (hindari muat ulang penuh UI)
    def _on_agent_submit():
        msg = st.session_state.get("_agent_input", "").strip()
        if msg:
            st.session_state["agent_messages"].append({"role": "user", "content": msg})
            st.session_state["_agent_pending"] = msg
            st.session_state["_agent_input"] = ""

    st.text_input("💬", placeholder="Tulis instruksi...", key="_agent_input",
                  on_change=_on_agent_submit, label_visibility="collapsed")

    # Menerjemahkan pesan dari obrolan menunggu
    if st.session_state.get("_agent_pending"):
        pending = st.session_state.pop("_agent_pending")
        # Ikon pemuatan kecil di latar belakang
        status_placeholder = st.empty()
        status_placeholder.markdown('<div style="color:#7c3aed; font-size:0.75rem;">⏳ memproses jawaban...</div>', unsafe_allow_html=True)
        try:
            from src.agents.intel_agent import run_agent
            reply = run_agent(pending)
            st.session_state["agent_messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state["agent_messages"].append({"role": "assistant", "content": f"❌ Gangguan fatal: {e}"})
        status_placeholder.empty()
        st.rerun()

    # ─── Indikator Validasi Pelatihan Model Dasar ────────────────────────────────
    st.markdown("<hr style='border-color:#2b1744; margin-top:1.5rem;'>", unsafe_allow_html=True)
    
    with st.expander("📊 Pelatihan & Model SetFit", expanded=False):
        report_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "reports", "setfit_training_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f: report = json.load(f)
            st.metric("Skor F1 Bobot", f"{report['results']['weighted_f1']:.4f}")
            st.metric("Tingkat Kepastian TINGGI", f"{report['results']['tinggi_precision']:.4f}")
            st.metric("Daya Telan TINGGI", f"{report['results']['tinggi_recall']:.4f}")
        else:
            st.caption("Buku catatan model masih kosong.")
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        
        # Penanda jumlah bantuan manual pengawas
        try:
            from src.db import get_connection as _gc
            _cdb = _gc()
            fb_count = _cdb.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            st.caption(f"💬 Catatan Revisi Aktif: **{fb_count}** · Ketentuan Retrain: **50**")
            can_retrain = fb_count >= 50
        except Exception:
            st.caption("💬 Database DuckDB otomatis tidak terlihat.")
            can_retrain = False
        
        st.button("🔄 Latih Ulang Model Prediktor", disabled=not can_retrain, 
                   help="Tombol dipicu setelah perkembang 50 perbaikan intervensi sistem.", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAMPILAN RUANG KERJA UTAMA
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛡️ Proyek Intelijen Cartensz</h1>
    <p style="margin:0; color:#e9d5ff; font-family:JetBrains Mono;">Pemantauan Serangan Taktik</p>
</div>
""", unsafe_allow_html=True)

# ─── Perhitungan Angka Pemantauan Standar ──────────────────────────────────────────────────────
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
    all_time_stats = conn.execute("""
        SELECT predicted_label, COUNT(*) as cnt
        FROM analysis_logs
        GROUP BY predicted_label
    """).df()
except Exception:
    trend_data = pd.DataFrame()
    today_stats = pd.DataFrame()
    all_time_stats = pd.DataFrame()

# Catatan Hari Ini
total_today = today_stats['cnt'].sum() if not today_stats.empty else 0
tinggi_today = today_stats[today_stats['predicted_label'] == 'TINGGI']['cnt'].sum() if not today_stats.empty else 0
waspada_today = today_stats[today_stats['predicted_label'] == 'WASPADA']['cnt'].sum() if not today_stats.empty else 0
aman_today = today_stats[today_stats['predicted_label'] == 'AMAN']['cnt'].sum() if not today_stats.empty else 0

# Koleksi Historis Terangkum Total
total_all = all_time_stats['cnt'].sum() if not all_time_stats.empty else 0
tinggi_all = all_time_stats[all_time_stats['predicted_label'] == 'TINGGI']['cnt'].sum() if not all_time_stats.empty else 0
waspada_all = all_time_stats[all_time_stats['predicted_label'] == 'WASPADA']['cnt'].sum() if not all_time_stats.empty else 0
aman_all = all_time_stats[all_time_stats['predicted_label'] == 'AMAN']['cnt'].sum() if not all_time_stats.empty else 0

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric-card"><h3>{total_today}</h3><p>Temuan Terakhir</p><span style="color:#a78bfa; font-size:0.7rem;">{total_all} semua waktu</span></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><h3 class="label-tinggi">{tinggi_today}</h3><p>🔴 TINGGI</p><span style="color:#a78bfa; font-size:0.7rem;">{tinggi_all} semua waktu</span></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><h3 class="label-waspada">{waspada_today}</h3><p>🟡 WASPADA</p><span style="color:#a78bfa; font-size:0.7rem;">{waspada_all} semua waktu</span></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card"><h3 class="label-aman">{aman_today}</h3><p>🟢 AMAN</p><span style="color:#a78bfa; font-size:0.7rem;">{aman_all} semua waktu</span></div>', unsafe_allow_html=True)

# ─── Cuplikan Rangkuman Borongan Triage Baru Masuk ──────────────────
if "triage_results" in st.session_state and st.session_state["triage_results"]:
    results = st.session_state["triage_results"]
    n_total = len(results)
    n_tinggi = sum(1 for r in results if r["label"] == "TINGGI")
    n_waspada = sum(1 for r in results if r["label"] == "WASPADA")
    n_aman = sum(1 for r in results if r["label"] == "AMAN")
    avg_entropy = sum(r.get("entropy", 0) for r in results) / max(n_total, 1)
    
    # Pendataan sumur data
    items = st.session_state.get("raw_triage_items", [])
    source_counts = {}
    for it in items:
        src = it.get("source", "Tidak Diketahui") if isinstance(it, dict) else "Tidak Diketahui"
        source_counts[src] = source_counts.get(src, 0) + 1
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    summary_parts = [
        f"Tertunda ada **{n_total}** subyek:",
        f"🔴 **{n_tinggi}** TINGGI · 🟡 **{n_waspada}** WASPADA · 🟢 **{n_aman}** AMAN",
        f"Rerata Keraguan (Entropy): **{avg_entropy:.4f}** · Lintasan Sumber: {', '.join([f'{s[0]} ({s[1]})' for s in top_sources])}",
    ]
    
    if n_tinggi > 0:
        pct = n_tinggi / n_total * 100
        summary_parts.append(f"⚠️ Perhatian: **{pct:.1f}%** ancaman tingkat parah terpantau. Segera teliti di kolom bawah dengan Radar.")
    
    st.markdown(
        f'<div style="background:#1c0d28; border:1px solid #3b1f5b; border-radius:10px; padding:1rem; margin:1rem 0;">'
        f'<strong style="color:#d8b4fe;">📋 Hasil Singkat Triage Cepat</strong><br>'
        f'<span style="color:#e9d5ff; font-size:0.9rem;">{"<br>".join(summary_parts)}</span>'
        f'</div>', unsafe_allow_html=True
    )

# ─── Ilustrasi Data Grafis ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 📈 Intensitas (Jarak 7 Hari)")
    if not trend_data.empty:
        chart_data = trend_data.pivot(index='dt', columns='predicted_label', values='cnt').fillna(0)
        for c in ['AMAN', 'WASPADA', 'TINGGI']:
            if c not in chart_data.columns: chart_data[c] = 0
        st.area_chart(chart_data[['TINGGI', 'WASPADA', 'AMAN']], height=220, color=["#ef4444", "#f59e0b", "#10b981"])
    else:
        st.info("Log operasional terpantau kosong.")

with chart_col2:
    st.markdown("#### 🌌 Klasterisasi Peta Semantik Kata (PCA)")
    if "triage_results" in st.session_state and st.session_state["triage_results"]:
        results = st.session_state["triage_results"]
        # Pastikan lapisan vektor dimensi sudah dibuat
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
            st.caption(f"Pemangkasan vektor dimensi 768. Varian data terekstrak {sum(pca.explained_variance_ratio_)*100:.1f}%")
            st.markdown('<p style="color:#a78bfa; font-size:0.75rem; margin-top:0.2rem;">Sumbu dekat menandakan kesamaan narasi. Pemecahan warna mencerminkan daya identifikasi model.</p>', unsafe_allow_html=True)
        else:
            import altair as alt
            # Fallback prosedur rasio darurat
            df_pca = pd.DataFrame({
                'Tingkat Ketidakpastian (Entropi)': [r.get("entropy", 0.1) for r in results],
                'Panjang Karakter': [len(r.get("text", "")) for r in results],
                'Label': [r["label"] for r in results]
            })
            color_scale = alt.Scale(domain=['AMAN', 'WASPADA', 'TINGGI'], range=['#10b981', '#f59e0b', '#ef4444'])
            scatter = alt.Chart(df_pca).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X('Tingkat Ketidakpastian (Entropi):Q'),
                y=alt.Y('Panjang Karakter:Q'),
                color=alt.Color('Label:N', scale=color_scale, legend=None),
                tooltip=['Label', 'Tingkat Ketidakpastian (Entropi)']
            ).properties(height=220).interactive()
            st.altair_chart(scatter, use_container_width=True)
            st.caption("⚠️ Operasi Alternatif: Entropi tebakan vs Skala kalimat.")
    else:
        st.info("Peta bintang otomatis terlukis saat operasi Triage ditarik tuasnya.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# MODAL LAYAR UNTUK SATU TEKS SAJA 
# ═══════════════════════════════════════════════════════════════════════
@st.dialog("🔍 Analisa Intelijen Dalam (Radar Mode)", width="large")
def _show_analysis_dialog(text):
    """Buka wendela terpisah untuk melancarkan hasil tanpa menimpa tabel."""
    st.info(f"**Objek Telaah:** _{text}_")
    with st.spinner("🚀 Melingkupkan setelan keamanan dan pemahaman Gemini..."):
        try:
            response = requests.post(f"{API_URL}/analyze", json={"text": text}, timeout=60)
            response.raise_for_status()
            brief = response.json().get("brief")
            if brief:
                render_deep_analysis(brief)
                # Form validitas prediksi dari tanggapan pengguna
                st.markdown("---")
                st.markdown("### 💬 Laporan Pengawas Lapangan")
                fb1, fb2 = st.columns(2)
                with fb1:
                    corrected = st.selectbox("Label Sebenarnya:", ["(Patokan Sudah Benar)", "AMAN", "WASPADA", "TINGGI"], key="fb_dialog")
                with fb2:
                    notes = st.text_input("Goresan Tinta Evaluasi:", key="notes_dialog")
                if st.button("Kunci ke Laci Server", key="save_dialog"):
                    if corrected != "(Patokan Sudah Benar)":
                        th = hashlib.sha256(text.encode()).hexdigest()[:16]
                        requests.post(f"{API_URL}/feedback", json={
                            "text_hash": th, "original_label": brief["classification"]["label"],
                            "corrected_label": corrected, "notes": notes
                        })
                        st.success("Terdokumentasi tanpa cacat!")
        except Exception as e:
            st.error(f"Mesin tercekik: {e}")

# Pemanggil jendela dialog otomatis jika slot teks dipicu
if st.session_state.get("_dialog_text"):
    text_for_dialog = st.session_state.pop("_dialog_text")
    _show_analysis_dialog(text_for_dialog)

# ═══════════════════════════════════════════════════════════════════════
# LEMARI ARSIP (Laci Susunan Tabel) The Sweep
# ═══════════════════════════════════════════════════════════════════════
if "triage_results" in st.session_state and st.session_state["triage_results"]:
    st.markdown("### 🚦 Lembar Sortir Cepat Output (Triage)")
    st.caption("Pilih ruas di bawah untuk memasukkan tulisan ke ruang periksa presisi tinggi LLM.")
    
    hide_aman = st.toggle("👁️ Lipat Barisan Status AMAN", value=True)
    
    items = st.session_state.get("raw_triage_items", [])
    
    df_rows = []
    for i, r in enumerate(st.session_state["triage_results"]):
        if hide_aman and r["label"] == "AMAN":
            continue
        # Ambil indikator tautan sumber
        source = items[i]["source"] if i < len(items) else "—"
        url = items[i].get("url", "") if i < len(items) else ""
        sigs = ", ".join([s.get("type", "?") for s in r.get("signal_highlights", [])])
        df_rows.append({
            "No. Baris": i,
            "Kesimpulan Label": r["label"],
            "Akurasi Duga": r.get("confidence", ""),
            "Tingkat Ambiguitas": round(r.get("entropy", 0), 4),
            "Faktor Peringatan Ancaman": sigs if sigs else "—",
            "Lintasan Sumber Asli": source,
            "Cuplikan Teks Pantau": r.get("text", "")[:100] + "..."
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
            sel_real_idx = int(df.iloc[sel_display_idx]["No. Baris"])
            sel_text = st.session_state["triage_results"][sel_real_idx]["text"]
            sel_source = items[sel_real_idx]["source"] if sel_real_idx < len(items) else "—"
            sel_url = items[sel_real_idx].get("url", "") if sel_real_idx < len(items) else ""
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 🔍 Ruang Detail Panel Telaah Tambahan")
            
            src_display = f"**Telur Cikal Bakal Data:** {sel_source}"
            if sel_url:
                src_display += f" — [Lintas Tautan Eksternal]({sel_url})"
            st.markdown(src_display)
            st.info(f"**Naskah Sasaran:** _{sel_text}_")
            
            with st.spinner("🚀 Mengurai rute komputasi Gemini..."):
                try:
                    response = requests.post(f"{API_URL}/analyze", json={"text": sel_text}, timeout=60)
                    response.raise_for_status()
                    brief = response.json().get("brief")
                    
                    if brief:
                        render_deep_analysis(brief)
                        
                        # Kolom penilaian manual susulan
                        st.markdown("---")
                        st.markdown("### 💬 Intervensi Pengawas")
                        fb1, fb2, fb3 = st.columns([1, 2, 1])
                        with fb1:
                            corrected = st.selectbox("Timpa Ketetapan:", ["(Simpan Prediksi Asli)", "AMAN", "WASPADA", "TINGGI"], key="fb_batch")
                        with fb2:
                            notes = st.text_input("Goresan Tinta Evaluasi:", key="notes_batch")
                        with fb3:
                            st.write(""); st.write("")
                            if st.button("Cap Sah Revisi", key="save_batch"):
                                if corrected != "(Simpan Prediksi Asli)":
                                    th = hashlib.sha256(sel_text.encode()).hexdigest()[:16]
                                    requests.post(f"{API_URL}/feedback", json={
                                        "text_hash": th, "original_label": brief["classification"]["label"],
                                        "corrected_label": corrected, "notes": notes
                                    })
                                    st.success("Arsip perbaikan tercetak kuat!")
                except Exception as e:
                    st.error(f"Gugur saat memanggil LLM pakar: {e}")
    else:
        st.info("✨ Daftar bersih, seluruh rekaman terpindai berstatus AMAN.")

# ─── Gulungan Rekaman DuckDB ─────────────────────────────────
if not ("triage_results" in st.session_state and st.session_state["triage_results"]):
    st.markdown("---")
    st.markdown("### 📜 Memori Tapak Tilas Data Lama")
    try:
        from src.db import get_connection as _gc2
        _hist_conn = _gc2()
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
            st.info("👈 Sorong tumpukan data log ke keranjang masukan.")
    except Exception:
        st.info("👈 Silakan gunakan menu rekam muatan di sisi kiri navigasi.")
