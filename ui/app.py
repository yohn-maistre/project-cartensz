"""
Project Cartensz — Intelligence Command Center V4
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

# pastikan jalur utama proyek bisa diakses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- tambalan sementara untuk setfit/transformers ---
import transformers.training_args as _ta
if not hasattr(_ta, "default_logdir"):
    import socket
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    def _default_logdir() -> str:
        current_time = _dt.now().strftime("%b%d_%H-%M-%S")
        return str(_Path("runs") / f"{current_time}_{socket.gethostname()}")
    _ta.default_logdir = _default_logdir
# --- akhir tambalan ---

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── konfigurasi halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Cartensz — Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── kustomisasi css (Pemolesan Mode Gelap Kontras Tinggi) ───────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { background-color: #0d0614 !important; color: #f3e8ff; font-family: 'Inter', sans-serif; }

    /* Perbaikan visibilitas bilah samping (Sidebar UI Polish) */
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

    /* kotak obrolan intel agent — scroll dapat digulir sesuai isi */
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

# ─── alat pembantu ─────────────────────────────────────────────────
def get_risk_color(score: int) -> str:
    if score <= 30: return "#10b981"
    elif score <= 60: return "#f59e0b"
    else: return "#ef4444"

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
    """tampilkan panel The Radar secara utuh untuk detail satu tangkapan teks."""
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
    
    # balok visual skor risiko
    risk_pct = brief["risk_score"]
    bar_color = get_risk_color(risk_pct)
    st.markdown(f'<div class="risk-bar" style="background: linear-gradient(to right, {bar_color} {risk_pct}%, #1c0d28 {risk_pct}%);"></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ringkasan eksekutif
    st.markdown("#### 📝 Laporan Eksekutif")
    st.write(brief["summary_narrative"])
    
    # angka probabilitas dan ukur keraguan mesin
    st.markdown("#### 📊 Probabilitas & Ketidakpastian")
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    probs = c.get("probabilities", {})
    prob_col1.metric("AMAN", f"{probs.get('AMAN', 0)*100:.1f}%")
    prob_col2.metric("WASPADA", f"{probs.get('WASPADA', 0)*100:.1f}%")
    prob_col3.metric("TINGGI", f"{probs.get('TINGGI', 0)*100:.1f}%")
    
    stats_col1, stats_col2 = st.columns(2)
    stats_col1.metric("Shannon Entropy", f"{c.get('entropy', 0):.4f}")
    stats_col2.metric("Kumpulan Prediksi Konformal", str(c.get('prediction_set', [])))
    
    # sinyal temuan
    st.markdown(f"#### 🚨 Sinyal Ancaman ({len(brief.get('signals_detected', []))} terdeteksi)")
    render_signals(brief.get("signals_detected", []))
    
    # jalan pikiran ai
    with st.expander("🧠 Penalaran AI (Chain-of-Thought)", expanded=False):
        st.write(c.get("reasoning", ""))
    
    # ambiguitas
    if brief.get("ambiguity_notes"):
        with st.expander("⚠️ Catatan Ambiguitas", expanded=False):
            st.warning(brief["ambiguity_notes"])

# ─── bilah navigasi samping (info + pasokan data) ────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h2 style="margin-bottom:0;"><span style="color:#a855f7;">🛡️</span> Cartensz</h2>
        <p style="color:#d8b4fe; font-size:0.85rem; margin-top:0.2rem;">Intelligence Command Center</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ Tentang Cartensz", expanded=False):
        st.markdown("""
        **Apa ini?** Cartensz mendeteksi narasi ancaman berbahasa Indonesia 
        menggunakan pipeline NLP fusi khusus.
        
        **⚙️ Arsitektur:**
        1. **Preprocessor:** Normalisasi slang/code-switching.
        2. **Sinyal:** Ekstraksi Leksikon (eufemisme, CTA).
        3. **Ahli Lokal:** `NusaBERT` + `SetFit` (0 biaya LLM).
        4. **Jaring Pengaman:** Penalaran `Gemini 3 Flash` (1 panggilan LLM).
        
        **📊 Kuantifikasi Keraguan Algoritma:**
        - Shannon Entropy dari sebaran tebakan.
        - Conformal Prediction Set (Adaptif).
        - Attention Weights dari kedalaman lapis NusaBERT.
        
        **🏷️ Klasifikasi Tingkat Kerawanan:**
        - 🟢 **AMAN** — Nol bahaya laten.
        - 🟡 **WASPADA** — Makna rancu, tetap periksa perlahan.
        - 🔴 **TINGGI** — Provokasi terstruktur/aktif terciduk.
        """)
        
    st.markdown("<hr style='border-color:#2b1744;'>", unsafe_allow_html=True)
    st.markdown("### 📥 Pasokan Pemantauan")
    
    input_method = st.radio("Metode Cerap Data:", ["🕸️ OSINT Scraper", "✍️ Manual / Paste", "📂 Unggah File", "🎤 Audio (ASR)"], label_visibility="collapsed")
    
    # ─── mode penarik cerdas osint ────────────────────────────────────
    if "OSINT Scraper" in input_method:
        st.caption("Bidik target data radar OSINT:")
        use_reddit = st.checkbox("🔴 Reddit", value=True)
        use_rss = st.checkbox("📰 RSS (Detik, Tempo)", value=True)
        use_youtube = st.checkbox("🎬 YouTube", value=True)
        use_telegram = st.checkbox("✈️ Telegram", value=False, disabled=True, help="Jalur terenkripsi/Tertunda proses koneksi")
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        use_custom = st.checkbox("🔗 Tautkan Jalur Sendiri (Custom)")
        if use_custom:
            custom_url = st.text_input("Garis Alamat (URL):", placeholder="https://...")
        
        if st.button("🕸️ Terjunkan Agen Scraper", use_container_width=True):
            with st.spinner("Scraping diam-diam..."):
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
                    # bentuk format standar
                    normalized = []
                    for it in all_items:
                        if isinstance(it, dict) and it.get("text", "").strip():
                            # set kunci wajib bila copong
                            it.setdefault("source", "OSINT")
                            it.setdefault("url", "")
                            normalized.append(it)
                    all_items = normalized
                    st.session_state["raw_triage_items"] = all_items
                    st.session_state["raw_triage_texts"] = [it["text"] for it in all_items]
                    st.success(f"Disedot paksa: {len(all_items)} dokumen wacana.")
                except Exception as e:
                    st.error(f"Scraper tepergok / error: {e}")
                    
    # ─── mode ketik tangan & copas ──────────────────────────────────────────
    elif "Manual" in input_method:
        example_choice = st.selectbox("Ambil cuplikan demo:", [
            "(Ketik sendiri bebas)",
            "Kemacetan terpantau di Jalan Sudirman pagi ini.",
            "Mereka dari barat mulai ikut campur urusan kita. Jangan biarkan mereka mengambil tanah leluhur.",
            "Siapkan senjata untuk jihad melawan musuh Allah. Tunggu aba-aba selanjutnya untuk menyerang malam ini.",
            "Demo buruh menuntut kenaikan UMR besok di depan gedung DPR."
        ])
        
        default_val = "" if example_choice == "(Ketik sendiri bebas)" else example_choice
        manual_text = st.text_area("Masukkan materi teks:", value=default_val, height=120)
        
        if st.button("🔍 Inisiasi Pemeriksaan", use_container_width=True):
            lines = [l.strip() for l in manual_text.strip().split("\n") if l.strip()]
            if lines:
                items = [{"text": l, "source": "Input Tangan Kosong", "url": ""} for l in lines]
                st.session_state["raw_triage_items"] = items
                st.session_state["raw_triage_texts"] = [it["text"] for it in items]
                # hidupkan The Radar untuk satu gelombang lempar teks tunggal
                if len(lines) == 1:
                    st.session_state["auto_radar_text"] = lines[0]
                    
    # ─── mode bongkahan file terunggah ──────────────────────────────
    elif "Unggah" in input_method:
        upload_file = st.file_uploader("Seret berkas CSV / TXT", type=["csv", "txt"])
        if upload_file and st.button("Eksekusi Skrip Berkas", use_container_width=True):
            if upload_file.name.endswith('.csv'):
                df = pd.read_csv(upload_file)
                texts = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                texts = [l.strip() for l in upload_file.read().decode('utf-8').split("\n") if l.strip()]
            texts = [t for t in texts if t.strip()]
            items = [{"text": t, "source": f"File Import: {upload_file.name}", "url": ""} for t in texts]
            st.session_state["raw_triage_items"] = items
            st.session_state["raw_triage_texts"] = [it["text"] for it in items]

    # ─── mode sadap suara ────────────────────────────────────────────
    elif "Audio" in input_method:
        st.caption("Buang file suara agar diproses oleh agen alih bahasa dengar (Qwen3-ASR)")
        audio_file = st.file_uploader("Kaset Audio", type=["wav", "mp3", "ogg", "flac", "m4a", "webm"], label_visibility="collapsed")
        if audio_file:
            st.audio(audio_file)
            if st.button("🎤 Dengarkan & Sadap", type="primary", use_container_width=True):
                with st.spinner("Merangkai kata dari getaran audio (Qwen3-ASR)..."):
                    try:
                        from src.asr.transcriber import transcribe_bytes
                        audio_bytes = audio_file.read()
                        asr_result = transcribe_bytes(audio_bytes, filename=audio_file.name)
                        transcribed = asr_result["text"]
                        st.success(f"Terjemahan disetir sebagai bahasa: {asr_result['language']}")
                        st.text_area("Buah Sadapan:", transcribed, height=100, disabled=True)
                        if transcribed.strip():
                            items = [{"text": transcribed, "source": f"Pita Audio: {audio_file.name}", "url": ""}]
                            st.session_state["raw_triage_items"] = items
                            st.session_state["raw_triage_texts"] = [transcribed]
                            st.session_state["auto_radar_text"] = transcribed
                    except Exception as e:
                        st.error(f"Mesin sadap rusak seketika: {e}")

    # pasang pemicu The Sweep kilat jika data terhimpun massal
    if st.session_state.get("raw_triage_texts") and not st.session_state.get("auto_radar_text"):
        n_texts = len(st.session_state["raw_triage_texts"])
        st.caption(f"📋 **{n_texts}** teks masuk zona karantina pratinjau kilat (triage).")
        if st.button("⚡ TEKAN TOMBOL TRIAGE KILAT", type="primary", use_container_width=True):
            with st.spinner(f"Mesin sapu jagat (The Sweep) beroperasi menelaah {n_texts} teks..."):
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
                            st.error(f"Kendala blok teks ke-{i}: {resp.text}")
                    st.session_state["triage_results"] = all_briefs
                    st.success("Triage beres tak bersisa!")
                except Exception as e:
                    st.error(f"Pelayan api tumbang: {e}")

    # ─── asisten adk pendamping ruang siber ──────────────────────────────────
    # chat UI integration with session memory
    st.markdown("<hr style='border-color:#2b1744; margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Pendamping Digital (ADK Agent)")

    # memori otak agen dalam sisi antarmuka
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    # kotak ngobrol lentur
    chat_html = '<div class="agent-chat-box" id="agent-chat-box">'
    if not st.session_state["agent_messages"]:
        chat_html += '<div style="color:#6b5b7b; font-size:0.8rem; text-align:center; padding:1rem;">Ketuk pintu pikiran AI di sini.<br/>Contoh: "Jelaskan data TINGGI hari ini" · "Buatkan ringkasan data kotor"</div>'
    else:
        for msg in st.session_state["agent_messages"]:
            if msg["role"] == "user":
                chat_html += f'<div class="agent-msg agent-msg-user"><div class="agent-msg-label">🔍 Anda (Operator)</div>{msg["content"]}</div>'
            else:
                content = msg["content"].replace("\n", "<br/>")
                chat_html += f'<div class="agent-msg agent-msg-bot"><div class="agent-msg-label">🛡️ Pendamping</div>{content}</div>'
    chat_html += '</div>'
    chat_html += '<script>var el=document.getElementById("agent-chat-box");if(el)el.scrollTop=el.scrollHeight;</script>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # pemantul input tulisan operator ke fungsi pemanggil
    def _on_agent_submit():
        msg = st.session_state.get("_agent_input", "").strip()
        if msg:
            st.session_state["agent_messages"].append({"role": "user", "content": msg})
            st.session_state["_agent_pending"] = msg
            st.session_state["_agent_input"] = ""

    st.text_input("💬", placeholder="Lemparkan komando/tanya...", key="_agent_input",
                  on_change=_on_agent_submit, label_visibility="collapsed")

    # pemrosesan tulisan yang menunda tayang (pending load spinner)
    if st.session_state.get("_agent_pending"):
        pending = st.session_state.pop("_agent_pending")
        # penanda mikir gaib ukuran liliput
        status_placeholder = st.empty()
        status_placeholder.markdown('<div style="color:#7c3aed; font-size:0.75rem;">⏳ memproses serangkaian simulasi di dalam otak silikon...</div>', unsafe_allow_html=True)
        try:
            from src.agents.intel_agent import run_agent
            reply = run_agent(pending)
            st.session_state["agent_messages"].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state["agent_messages"].append({"role": "assistant", "content": f"❌ Sumbu konslet: {e}"})
        status_placeholder.empty()
        st.rerun()

    # tombol perusak memori agen
    if st.session_state["agent_messages"]:
        if st.button("🗑️ Sapu ingatan bot", key="clear_chat", type="tertiary"):
            st.session_state["agent_messages"] = []
            try:
                from src.agents.intel_agent import reset_session
                reset_session()
            except Exception:
                pass
            st.rerun()

    # ─── papan kendali mesin bahasa lokal (side bottom tier) ─────────────────
    st.markdown("<hr style='border-color:#2b1744; margin-top:1.5rem;'>", unsafe_allow_html=True)
    
    with st.expander("📊 Ruang Mesin SetFit", expanded=False):
        report_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "reports", "setfit_training_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f: report = json.load(f)
            st.metric("Skor Kesatuan F1", f"{report['results']['weighted_f1']:.4f}")
            st.metric("Ketepatan Target TINGGI", f"{report['results']['tinggi_precision']:.4f}")
            st.metric("Daya Telan Target TINGGI", f"{report['results']['tinggi_recall']:.4f}")
        else:
            st.caption("Masih bersih, belom ada ijazah pelatihan tercetak.")
        
        st.markdown("<hr style='border-color:#2b1744; margin:0.5rem 0;'>", unsafe_allow_html=True)
        
        # alat penghitung tabungan perbaikkan data (koreksi analis)
        try:
            from src.db import get_connection as _gc
            _cdb = _gc()
            fb_count = _cdb.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            st.caption(f"💬 Tangkapan Umpan Balik: **{fb_count}** · Ambang Syarat Rekalibrasi: **50**")
            can_retrain = fb_count >= 50
        except Exception:
            st.caption("💬 Tangkapan Mati: Jalur penghubung basisdata (DuckDB) tiada.")
            can_retrain = False
        
        st.button("🔄 Pecut Latih Ulang Algoritma", disabled=not can_retrain, 
                   help="Tombol hanya menyala sesudah akumulasi masukan analis tembus batas (50 butir).", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA (KONTEN INTI KANAN)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛡️ Pusat Komando Cartensz</h1>
    <p style="margin:0; color:#e9d5ff; font-family:JetBrains Mono;">Dashboard Operasi Keamanan Intelijen Taktis</p>
</div>
""", unsafe_allow_html=True)

# ─── jejak harian rekaman ─────────────────────────────────────────
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

with col1: st.markdown(f'<div class="metric-card"><h3>{total_today}</h3><p>Tebaran Hari Ini</p></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><h3 class="label-tinggi">{tinggi_today}</h3><p>🔴 KELAS TINGGI</p></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><h3 class="label-waspada">{waspada_today}</h3><p>🟡 MASA WASPADA</p></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card"><h3 class="label-aman">{aman_today}</h3><p>🟢 STATUS AMAN</p></div>', unsafe_allow_html=True)

# ─── sekilas pandang borongan (nongol sehabis penyapuan sweep) ──────────────────
if "triage_results" in st.session_state and st.session_state["triage_results"]:
    results = st.session_state["triage_results"]
    n_total = len(results)
    n_tinggi = sum(1 for r in results if r["label"] == "TINGGI")
    n_waspada = sum(1 for r in results if r["label"] == "WASPADA")
    n_aman = sum(1 for r in results if r["label"] == "AMAN")
    avg_entropy = sum(r.get("entropy", 0) for r in results) / max(n_total, 1)
    
    # pemetaan muasal temuan
    items = st.session_state.get("raw_triage_items", [])
    source_counts = {}
    for it in items:
        src = it.get("source", "Sarang Gelap") if isinstance(it, dict) else "Sarang Gelap"
        source_counts[src] = source_counts.get(src, 0) + 1
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    summary_parts = [
        f"Kupas Tuntas Atas **{n_total}** Pucuk Wacana:",
        f"🔴 **{n_tinggi}** Ancaman Tajam · 🟡 **{n_waspada}** Zona Ambigu · 🟢 **{n_aman}** Lolos Keamanan",
        f"Level Keraguan Rata-Rata Mesin: **{avg_entropy:.4f}** · Lini Depan Pembawa: {', '.join([f'{s[0]} ({s[1]})' for s in top_sources])}",
    ]
    
    if n_tinggi > 0:
        pct = n_tinggi / n_total * 100
        summary_parts.append(f"⚠️ Peringatan: **{pct:.1f}%** jaring menenggak ancaman riil tinggi. Tukik ke dalam baris laci klasifikasi terlampir di bawah.")
    
    st.markdown(
        f'<div style="background:#1c0d28; border:1px solid #3b1f5b; border-radius:10px; padding:1rem; margin:1rem 0;">'
        f'<strong style="color:#d8b4fe;">📋 Ulasan Singkat Operasi Kompartemen Jaring Lebar (The Sweep)</strong><br>'
        f'<span style="color:#e9d5ff; font-size:0.9rem;">{"<br>".join(summary_parts)}</span>'
        f'</div>', unsafe_allow_html=True
    )

# ─── seksi pencitraan grafis ──────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 📈 Detak Laju Gejolak (Spam 7 Hari Mengudara)")
    if not trend_data.empty:
        chart_data = trend_data.pivot(index='dt', columns='predicted_label', values='cnt').fillna(0)
        for c in ['AMAN', 'WASPADA', 'TINGGI']:
            if c not in chart_data.columns: chart_data[c] = 0
        st.area_chart(chart_data[['TINGGI', 'WASPADA', 'AMAN']], height=220, color=["#ef4444", "#f59e0b", "#10b981"])
    else:
        st.info("Catatan cuaca historis murni bersih tanpa kabut.")

with chart_col2:
    st.markdown("#### 🌌 Radar Pembaca Pikiran Semantik NusaBERT (PCA)")
    if "triage_results" in st.session_state and st.session_state["triage_results"]:
        results = st.session_state["triage_results"]
        # pantau ada tidaknya sumsum kedalaman ruang vektor embedding
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
                x=alt.X('PC1:Q', title=f'Dimensi Ekstraksi 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'),
                y=alt.Y('PC2:Q', title=f'Dimensi Ekstraksi 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'),
                color=alt.Color('Label:N', scale=color_scale, legend=None),
                tooltip=['Label']
            ).properties(height=220).interactive()
            st.altair_chart(scatter, use_container_width=True)
            st.caption(f"Hasil intipan ruang matriks vektor lokal 768 lapis. Penjelasan pola variabilitas disumbang sebesar {sum(pca.explained_variance_ratio_)*100:.1f}%")
        else:
            import altair as alt
            # Fallback: sandaran manual via korelasi rasa bingung mesin per kata
            df_pca = pd.DataFrame({
                'Tingkat Kebingungan Sintetik': [r.get("entropy", 0.1) for r in results],
                'Rentang Nafas Kalimat': [len(r.get("text", "")) for r in results],
                'Label': [r["label"] for r in results]
            })
            color_scale = alt.Scale(domain=['AMAN', 'WASPADA', 'TINGGI'], range=['#10b981', '#f59e0b', '#ef4444'])
            scatter = alt.Chart(df_pca).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X('Tingkat Kebingungan Sintetik:Q'),
                y=alt.Y('Rentang Nafas Kalimat:Q'),
                color=alt.Color('Label:N', scale=color_scale, legend=None),
                tooltip=['Label', 'Tingkat Kebingungan Sintetik']
            ).properties(height=220).interactive()
            st.altair_chart(scatter, use_container_width=True)
            st.caption("⚠️ Prosedur Alternatif: Kalkulasi Entropi versus Rasio Pengukuran Deret Kalimat (pasokan vektor embedding api mati/belum dirender).")
    else:
        st.info("Bentangan galaksi semantik radar muncul kelar tombol Triage ditarik.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# AUTO-RADAR: Fokus Sorot Satu Kasus (Memanggil Bintang Utama LLM)
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.get("auto_radar_text"):
    single_text = st.session_state["auto_radar_text"]
    st.markdown("## 🔍 Radar Satuan Bintang Lima (Penetrasi Panggilan LLM Utuh)")
    st.info(f"**Objek Pembongkaran:** _{single_text}_")
    
    with st.spinner("🚀 Melakukan pengawinan tebakan algoritma pelat merah SetFit dan akal buatan Gemini 3 Flash..."):
        try:
            response = requests.post(f"{API_URL}/analyze", json={"text": single_text}, timeout=60)
            response.raise_for_status()
            brief = response.json().get("brief")
            
            if brief:
                render_deep_analysis(brief)
                
                # penyuapan pakan data umpan balik (active training loop feeder)
                st.markdown("---")
                st.markdown("### 💬 Koreksi Kebijaksanaan Ahli Sandi Lapangan")
                fb1, fb2, fb3 = st.columns([1, 2, 1])
                with fb1:
                    corrected = st.selectbox("Keputusan Vonis:", ["(Ketepatan Memuaskan)", "AMAN", "WASPADA", "TINGGI"], key="fb_single")
                with fb2:
                    notes = st.text_input("Goresan Tinta (catatan):", key="notes_single")
                with fb3:
                    st.write(""); st.write("")
                    if st.button("Catat ke Pita Rekam Duk", key="save_single"):
                        if corrected != "(Ketepatan Memuaskan)":
                            th = hashlib.sha256(single_text.encode()).hexdigest()[:16]
                            requests.post(f"{API_URL}/feedback", json={
                                "text_hash": th, "original_label": brief["classification"]["label"],
                                "corrected_label": corrected, "notes": notes
                            })
                            st.success("Tembusan tersampaikan!")
        except Exception as e:
            st.error(f"Gugur berkalang tanah saat di jalan asinkronus: {e}")
    
    # pembersihan panggung radar untuk lakon berkutnya
    if st.button("🔄 Ulang dan Lepas Target Pandang", use_container_width=True):
        del st.session_state["auto_radar_text"]
        if "raw_triage_texts" in st.session_state:
            del st.session_state["raw_triage_texts"]
        if "raw_triage_items" in st.session_state:
            del st.session_state["raw_triage_items"]
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# TRIAGE TABLE (Peti Es Pengawetan Baris Berjejer The Sweep — Gratisan LLM)
# ═══════════════════════════════════════════════════════════════════════
elif "triage_results" in st.session_state and st.session_state["triage_results"]:
    st.markdown("### 🚦 Lemari Arsip Triage Kilat (The Sweep Sapu Bersih Tanpa LLM)")
    st.caption("Pilih paksa satu baris untuk menyeretnya masuk ke siksaan lampu sorot The Radar dan LLM.")
    
    hide_aman = st.toggle("👁️ Singkirkan Bangkai Kertas AMAN", value=True)
    
    items = st.session_state.get("raw_triage_items", [])
    
    df_rows = []
    for i, r in enumerate(st.session_state["triage_results"]):
        if hide_aman and r["label"] == "AMAN":
            continue
        # lacak benang muasal barang
        source = items[i]["source"] if i < len(items) else "—"
        url = items[i].get("url", "") if i < len(items) else ""
        sigs = ", ".join([s.get("type", "?") for s in r.get("signal_highlights", [])])
        df_rows.append({
            "No_Identitas": i,
            "Kesimpulan Label": r["label"],
            "Indikator Kepastian": r.get("confidence", ""),
            "Misteri (Entropy)": round(r.get("entropy", 0), 4),
            "Kedipan Sinyal": sigs if sigs else "—",
            "Lubang Asal": source,
            "Untaian Teks": r.get("text", "")[:100] + "..."
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
            st.markdown("## 🔍 Panggung The Radar Utama — Sorotan Penetrasi")
            
            src_display = f"**Telur Cikal Bakal Data:** {sel_source}"
            if sel_url:
                src_display += f" — [Buka tautan jejak]({sel_url})"
            st.markdown(src_display)
            st.info(f"**Surat Objek Incaran:** _{sel_text}_")
            
            with st.spinner("🚀 Memacu pemanggilan Direktur Akal Silikon (Gemini 3 Flash)..."):
                try:
                    response = requests.post(f"{API_URL}/analyze", json={"text": sel_text}, timeout=60)
                    response.raise_for_status()
                    brief = response.json().get("brief")
                    
                    if brief:
                        render_deep_analysis(brief)
                        
                        # kotak koreksi dari mandor
                        st.markdown("---")
                        st.markdown("### 💬 Ketetapan Revisi Mandor Pengawas")
                        fb1, fb2, fb3 = st.columns([1, 2, 1])
                        with fb1:
                            corrected = st.selectbox("Stempel Penindakan:", ["(Presisi Aman)", "AMAN", "WASPADA", "TINGGI"], key="fb_batch")
                        with fb2:
                            notes = st.text_input("Goresan Petah Umpan Balik Log:", key="notes_batch")
                        with fb3:
                            st.write(""); st.write("")
                            if st.button("Masuk Laci Keras", key="save_batch"):
                                if corrected != "(Presisi Aman)":
                                    th = hashlib.sha256(sel_text.encode()).hexdigest()[:16]
                                    requests.post(f"{API_URL}/feedback", json={
                                        "text_hash": th, "original_label": brief["classification"]["label"],
                                        "corrected_label": corrected, "notes": notes
                                    })
                                    st.success("Terdokumentasi abadi!")
                except Exception as e:
                    st.error(f"Gugur koneksi saat melakukan pengeboran lapisan deep analysis: {e}")
    else:
        st.info("✨ Hamparan bersih, seluruh wacana tervalidasi AMAN jaya berlapis es. Tarik sakelar saring jika rindu kekacauan file aman.")
else:
    st.info("👈 Setor setumpuk data mentah di bilah kelongsoran sisi kiri untuk mulai memijarkan api kompor panggangan.")
