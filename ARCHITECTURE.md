# Project Cartensz — Technical Architecture & Deep-Dive

> **Threat Narrative Classifier for PT Gemilang Satria Perkasa**
> Candidate: Yose M. Giyay | AI/ML Engineer — Seleksi Teknikal Tahap II

---

## 1. Problem Statement

PT Gemilang Satria Perkasa processes thousands of Indonesian-language texts daily from news, social media, and community forums. The core challenge is **linguistic nuance**: Indonesian threat narratives rarely use explicit violent language. They operate through:

- **Euphemism**: "sapu bersih" (sweep clean) → ethnic cleansing
- **Code-switching**: Arabic-Indonesian mixing signals extremist rhetoric
- **Community codes**: "mujahid", "istisyhad" — innocuous in religious context, lethal in tactical context
- **Metaphor/framing**: "sudah waktunya bangkit" (it's time to rise) — mobilization or motivation?

A keyword-matching system would fail catastrophically. We need a system where **every prediction is explainable** and **uncertainty is reported honestly**, so analysts can trust the output even on ambiguous cases.

---

## 2. System Architecture

### 2.1 File Structure

```
gsp-eval/
├── api/
│   └── main.py                 # FastAPI REST server
├── data/
│   ├── cartensz.duckdb          # Embedded analytical database
│   ├── curated/
│   │   └── setfit_balanced_600.csv  # Balanced training dataset
│   └── setfit_model/
│       └── nusabert_base_ft/    # Trained SetFit checkpoint
├── scripts/
│   ├── collector.py             # OSINT scraper (Reddit, RSS, YouTube)
│   ├── daily_scrape.py          # Automated daily OSINT + triage (cron target)
│   ├── evaluate.py              # Model evaluation (confusion matrix, F1)
│   ├── evaluate_asr.py          # ASR WER evaluation
│   └── generate_briefs.py       # Batch Intelligence Brief generator
├── src/
│   ├── config.py                # Settings (API keys, model names, paths)
│   ├── db.py                    # DuckDB connection + schema + CRUD
│   ├── lexicon.py               # Euphemism lexicon + regex patterns
│   ├── llm_client.py            # LiteLLM wrapper with caching + key rotation
│   ├── models.py                # Pydantic data models
│   ├── data_pipeline.py         # Dataset loading/processing
│   ├── data_synthesizer.py      # Synthetic TINGGI sample generation via Gemini
│   └── agents/
│       ├── preprocessor.py      # Stage 1: Text normalization
│       ├── signal_extractor.py  # Stage 2: Threat signal detection
│       ├── brief_writer.py      # Stage 3+4: SetFit classifier + LLM brief writer
│       ├── orchestrator.py      # Pipeline coordinator (Radar + Sweep)
│       └── intel_agent.py       # ADK conversational agent
├── src/asr/
│   └── transcriber.py           # Qwen3-ASR-0.6B speech-to-text + WER
├── src/ml/
│   ├── train_setfit.py          # SetFit training script
│   └── train_nusabert.py        # Legacy NusaBERT fine-tuning (superseded)
├── ui/
│   └── app.py                   # Streamlit: Project Cartensz
├── samples/                     # 5 sample Intelligence Briefs (M5 requirement)
├── notebooks/                   # Jupyter notebooks for data exploration
│   ├── 01_evaluation.ipynb      # Basic eval report display
│   └── 02_deep_analysis.ipynb   # Comprehensive quantitative analysis + tokenizer comparison
├── .env                         # GEMINI_API_KEY, YOUTUBE_API_KEY
├── Dockerfile                   # Python 3.12-slim, uv, ffmpeg
├── docker-compose.yml           # 3 services: api, ui, scraper (6AM WIB cron)
├── pyproject.toml               # uv project configuration
└── requirements.txt             # pip-compatible dependencies
```

### 2.2 Two Operating Modes

The system has two distinct analysis modes that share the same pipeline stages:

| Mode | Name | LLM Calls | Use Case |
|---|---|---|---|
| **Deep Analysis** | The Radar | 1 | Single-text investigation with full Intelligence Brief |
| **Batch Triage** | The Sweep | 0 | Screen hundreds of texts instantly, surface anomalies |

Both modes run through the same Preprocessor → Signal Extractor → SetFit Classifier stages. The Radar adds a final LLM call (Gemini 3 Flash) to produce the Intelligence Brief with Chain-of-Thought reasoning.

### 2.3 Data Flow

```
                    ┌──────────────────────────────────┐
                    │         ENTRY POINTS              │
                    │  Streamlit UI  │  FastAPI  │  ADK  │
                    └────────┬───────────┬────────┬─────┘
                             │           │        │
              ┌──────────────┘           │        └──────────────┐
              ▼                          ▼                       ▼
     orchestrator.py            orchestrator.py          intel_agent.py
     run_pipeline()             batch_classify()         deep_analyze() tool
     (The Radar)                (The Sweep)              → run_pipeline()
              │                          │
              ├──── SHARED STAGES ───────┤
              ▼                          ▼
     1. preprocessor.preprocess()   →  NormalizedText
     2. signal_extractor.extract()  →  List[ThreatSignal]
     3. brief_writer.classify_local() → {AMAN: 0.x, WASPADA: 0.x, TINGGI: 0.x}
              │                          │
              ▼ (Radar only)             ▼ (Sweep stops here)
     4. brief_writer.generate_brief()   → classify_local_with_attention()
        (1 Gemini call)                   + encode_texts() for PCA
              │                          │
              ▼                          ▼
     IntelligenceBrief              Triage dict (label, probs, signals)
              │                          │
              └──────────┬───────────────┘
                         ▼
                ┌─────────────────┐
                │  DuckDB Logger   │
                │  analysis_logs   │  ← pipeline_mode: 'RADAR' or 'SWEEP'
                │  feedback_logs   │  ← analyst corrections
                └─────────────────┘
```

---

## 3. Pipeline Stages (Deep-Dive)

### Stage 1: Preprocessor (`preprocessor.py`)

Normalizes raw Indonesian text through 8 sequential steps:

1. **URL extraction & stripping** — preserves URLs for provenance
2. **Emoji → text conversion** — 14 threat-relevant mappings (🔥→`[api]`, 💣→`[bom]`, 🔫→`[senjata]`)
3. **Slang normalization** — 35+ Indonesian slang→formal mappings (`gw`→`saya`, `gak`→`tidak`, `anjir`→`anjing`)
4. **Mention/hashtag cleanup** — strips `@mentions`, preserves `#hashtag` text
5. **Whitespace normalization**
6. **Language mix detection** — estimates Arabic/Latin/other script ratios from the *original* text
7. **Token counting** — whitespace-based approximation
8. **SHA-256 hashing** — content hash for caching and audit trail

**Output**: `NormalizedText` Pydantic model with all metadata.

### Stage 2: Signal Extractor (`signal_extractor.py` + `lexicon.py`)

Runs 4 detection modules in parallel on the *original* text (preserving slang/emojis for accurate matching):

| Detector | Method | What it finds |
|---|---|---|
| **Euphemism matcher** | Lexicon lookup (22 entries) | "sapu bersih", "istisyhad", "kafir harbi", etc. |
| **Call-to-action detector** | 11 regex patterns | "ayo kita", "serang!", "turun ke jalan" |
| **Temporal urgency** | 7 regex patterns | "malam ini", "segera", "besok pagi" |
| **Code-switching analyzer** | Arabic script ratio | Flags texts >5% Arabic content |

Each signal is typed (`EUPHEMISM`, `CALL_TO_ACTION`, `TEMPORAL`, `OTHER`) with a `significance` rating (`HIGH`, `MEDIUM`, `LOW`) and `context_explanation`. Signals are deduplicated before returning.

**Cost**: 0 LLM calls — pure Python regex and dictionary lookup.

### Stage 3: Local Expert Classifier (`brief_writer.py`)

The SetFit model (NusaBERT backbone) provides three capabilities:

**Classification** (`classify_local`):
- Input: normalized text
- Output: `{AMAN: 0.72, WASPADA: 0.21, TINGGI: 0.07}`
- Derived metrics:
  - **Shannon Entropy**: `H = -Σ p·log₂(p)` — higher = more uncertain
  - **Conformal Prediction Set**: Adaptive Prediction Sets (APS) with α=0.1 — accumulates classes by descending probability until 90% coverage. Set size >1 = honest ambiguity.
  - **Confidence level**: HIGH (max prob ≥0.75), MEDIUM (≥0.50), LOW (<0.50)

**Attention extraction** (`classify_local_with_attention`):
- Runs the NusaBERT forward pass with `output_attentions=True`
- Extracts the last transformer layer's attention matrix
- Averages across all attention heads
- Takes the CLS token's attention to all other tokens (row 0)
- Returns per-token importance scores normalized to `[0, 1]`
- This explains *which words the model focused on* for classification

**Embedding extraction** (`encode_texts`):
- Uses the SentenceTransformer body to produce 768-dimensional embeddings
- Used for the PCA Semantic Threat Map in the UI
- Batch-encoded for efficiency

**Model loading**: Lazy-loaded via `_get_setfit_model()`. Includes a monkey-patch for the `transformers`/`setfit` version clash (see Section 6).

### Stage 4: Fused Brief Writer (`brief_writer.py` — `generate_brief`)

In a **single Gemini 3 Flash API call**, the LLM:

1. Reviews the SetFit model's probability distribution
2. Reviews all extracted threat signals with their context
3. Performs Chain-of-Thought reasoning about the classification
4. Acts as a **safety net** — can override the local model's decision if signals warrant it
5. Generates a structured Intelligence Brief with all 7 sections (matching the assignment template)

The LLM prompt is fused — it contains the local model's probabilities, all signal details, and the original text. The LLM must respond in structured JSON that maps to the `IntelligenceBrief` Pydantic model, which validates:
- `ambiguity_notes` must be ≥10 characters (system that never reports ambiguity = overclaiming)
- Probabilities must sum to 1.0

### Pipeline Coordinator (`orchestrator.py`)

This is the glue that chains Stages 1–4 together:

| Function | Mode | What it does |
|---|---|---|
| `run_pipeline(text)` | The Radar | Runs all 4 stages → returns `IntelligenceBrief`, logs to DuckDB as `pipeline_mode='RADAR'` |
| `batch_classify(texts)` | The Sweep | Runs Stages 1–3 only (no LLM), extracts attention + embeddings, logs as `'SWEEP'` |
| `analyze(text)` | Sync wrapper | `asyncio.run(run_pipeline(text))` for non-async contexts |

---

## 4. Model Training (SetFit)

### 4.1 The Imbalance Crisis

Our initial dataset mirrored real-world distribution: **92.5% AMAN, 7.2% WASPADA, 0.3% TINGGI**. Standard NusaBERT fine-tuning with 104x inverse class-weighted loss achieved 0.90 Weighted F1 but **0.0 TINGGI Precision/Recall** — the model learned to predict AMAN for everything.

### 4.2 The SetFit Pivot

SetFit (Sentence Transformer Fine-Tuning) uses **contrastive learning** — it learns a similarity space where texts of the same class cluster together. This approach:
- Needs very few labeled examples per class
- Is naturally robust to class imbalance
- Produces calibrated probability distributions

### 4.3 Synthetic Data Augmentation

We used Gemini 3 Flash to generate **120 synthetic TINGGI samples** to augment the 80 real ones, creating a perfectly balanced **600-sample dataset** (200/200/200). The synthetic samples were constrained to:
- Use realistic Indonesian slang and code-switching
- Cover different threat subcategories (religious extremism, ethnic incitement, mobilization)
- Vary register (social media, forum posts, messaging)

### 4.4 Training Configuration

```python
MODEL_NAME = "LazarusNLP/all-nusabert-base-v4"  # Indonesian SentenceTransformer
EPOCHS = 3
BATCH_SIZE = 4    # Reduced from 16 to prevent CPU OOM
NUM_ITERATIONS = 3  # Contrastive pairs per sample (reduced from 20 → 5 → 3)
```

**Train/test split**: 80/20 stratified (480 train, 120 test)

### 4.5 Results

| Metric | Target | Achieved |
|---|---|---|
| **Weighted F1** | ≥ 0.70 | **0.719** ✅ |
| **TINGGI Precision** | ≥ 0.75 | **0.812** ✅ |
| TINGGI Recall | — | 0.650 |

The training report (JSON) is saved at `notebooks/reports/setfit_training_report.json` and displayed in the Streamlit sidebar.

---

## 5. Databases & Storage

### DuckDB (`src/db.py`)

Embedded analytical SQL database. Replaced SQLite (too slow for analytical queries) and ChromaDB (RAG was dropped).

**Tables:**

| Table | Primary Key | Purpose |
|---|---|---|
| `analysis_logs` | `id` (SHA-256 hash prefix) | Every text ever analyzed — label, risk, confidence, entropy, latency, pipeline mode |
| `feedback_logs` | `text_hash` | Analyst corrections for active learning (UPSERT on duplicate) |

**Key queries used by the ADK agent:**
- Daily stats: `SELECT predicted_label, COUNT(*) ... WHERE DATE(timestamp) = CURRENT_DATE GROUP BY predicted_label`
- Trend: `SELECT DATE(timestamp) as dt, predicted_label, COUNT(*) ... GROUP BY dt, predicted_label`
- Threat search: `SELECT ... WHERE predicted_label = ? AND timestamp >= ? AND input_text LIKE ?`
- Latest triage: `SELECT ... WHERE pipeline_mode = 'SWEEP' ORDER BY timestamp DESC LIMIT ?`

### LLM Response Cache (`src/llm_client.py`)

File-based JSON cache in `data/.llm_cache/`. Cache key = SHA-256 of `{model, messages, temperature}`. Ensures **production consistency** — identical inputs produce identical outputs (requirement M1).

---

## 6. Key Architectural Decisions

### Decision A: Why SetFit over Standard Fine-Tuning?

Standard NusaBERT fine-tuning with class-weighted cross-entropy failed (0.0 TINGGI recall). The minority class gradients were washed out by the 300:1 imbalance ratio. SetFit's contrastive learning approach is fundamentally different — it learns a metric space, not a decision boundary, making it inherently robust to imbalance with as few as 8 examples per class.

### Decision B: Why No RAG?

We initially implemented ChromaDB vector retrieval (still visible as `SimilarExample` in the models). We dropped it because:
- The task is **classification**, not information retrieval
- RAG adds 2-3 seconds of latency for marginal classification improvement
- Context window pollution from retrieved examples hurts more than helps
- The Signal Extractor's deterministic lexicons are more reliable than fuzzy retrieval

### Decision C: Fused Pipeline over Multi-Agent Debate

We initially designed an adversarial system: `ClassifierAgent` → `CriticAgent` → `BriefWriterAgent` (3 LLM calls). We fused these into a single LLM call where:
- The model receives the local classifier's probabilities and all signals
- It must write Chain-of-Thought reasoning *before* issuing the final label
- This achieves the same adversarial self-correction in O(1) LLM latency

### Decision D: DuckDB over SQLite/ChromaDB

- SQLite: Too slow for analytical aggregation queries (GROUP BY, window functions) on 10K+ rows
- ChromaDB: Only needed for RAG, which we dropped
- DuckDB: Columnar analytical engine, embedded, zero config, fast aggregations

### Decision E: LiteLLM Abstraction Layer

All LLM calls go through `llm_client.py` which wraps LiteLLM. This gives us:
- **Provider agnostic**: Switch from Gemini to GPT-4 by changing one string
- **API key rotation**: Round-robin through GEMINI_API_KEY, GEMINI_API_KEY_2, etc.
- **Response caching**: File-based SHA-256 cache for production consistency
- **Rate limit resilience**: Auto-retry with next key on 429 errors

---

## 7. REST API

### Endpoints (`api/main.py`)

| Endpoint | Method | Input | Output | LLM Calls |
|---|---|---|---|---|
| `/analyze` | POST | `{text: str}` | `IntelligenceBrief` | 1 |
| `/batch` | POST | `{texts: [str]}` (max 20) | `[{label, probs, signals}]` | 0 |
| `/feedback` | POST | `{text_hash, original_label, corrected_label}` | `{success}` | 0 |
| `/retrain` | POST | — | Starts background SetFit training | 0 |
| `/health` | GET | — | `{status: "ok"}` | 0 |
| `/transcribe` | POST | Audio file (multipart) | Transcription + optional `IntelligenceBrief` | 0-1 |

The API uses FastAPI with Pydantic validation. The `/batch` endpoint calls `batch_classify()` (The Sweep), while `/analyze` calls `run_pipeline()` (The Radar).

---

## 8. OSINT Collector (`scripts/collector.py`)

### Multi-Source Scraping

| Source | Method | Fallback |
|---|---|---|
| **Reddit** | `old.reddit.com` JSON with browser UA | RSS feed via `feedparser` |
| **RSS News** | `feedparser` (Detik News, Tempo Nasional, Detik Finance) | — |
| **YouTube** | YouTube Data API v3 (comments on Indonesian political videos) | — |

Reddit scraping required significant engineering due to Cloudflare bot detection. The dual-layer fallback (JSON → RSS) with realistic User-Agent headers ensures at least one method always works.

Every scraped text carries its provenance (`source: "Reddit"`, `source: "RSS: Detik News"`) through the entire pipeline, displayed in the triage table for full traceability.

---

## 9. ADK Conversational Agent (`intel_agent.py`)

### Architecture

- **Framework**: Google ADK with Gemini 3 Flash function-calling
- **Personality**: No-BS professional intel analyst — terse, authoritative Bahasa Indonesia
- **Memory**: `InMemorySessionService` provides multi-turn conversation context (the agent remembers what was discussed earlier in the same session)

### 5 Tools

| Tool | What it queries | Cost |
|---|---|---|
| `search_threats(label, days_back, limit, keyword)` | DuckDB `analysis_logs` with flexible filters | $0 |
| `get_daily_stats()` | Today's AMAN/WASPADA/TINGGI distribution + feedback count | $0 |
| `get_trend(days)` | Multi-day aggregated threat trends | $0 |
| `get_latest_triage(limit)` | Most recent batch triage results from The Sweep | $0 |
| `deep_analyze(text)` | Runs full Radar pipeline on any text | 1 LLM call |

### Key Engineering: Async in Streamlit

Streamlit's `ScriptRunner.scriptThread` has no asyncio event loop. ADK's `Runner.run_async()` requires one. Solution: `run_agent()` always creates a fresh `asyncio.new_event_loop()` in a `ThreadPoolExecutor` background thread.

### Environment Variable Mapping

ADK's underlying `google-genai` SDK checks `GOOGLE_API_KEY`, not `GEMINI_API_KEY`. We auto-map at module load: `os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]`.

---

## 10. Streamlit UI (`ui/app.py`)

### Sidebar Layout (top → bottom)

1. **Header**: "🛡️ Cartensz — Intelligence Command Center"
2. **About**: Pipeline architecture explainer
3. **Data Input**: Tabbed for OSINT Scraper / Manual Paste / File Upload / **🎤 Audio (ASR)**
4. **Inline Triage Button**: Shows `📋 N teks siap untuk triage.` + button (only when texts loaded)
5. **Intel Agent Chat**: Scrollable 350px container, callback-based input, subtle loading indicator
6. **Model & Retraining**: Expander with F1/Precision metrics and retrain button

### Main Content

1. **Metric cards**: Total texts, TINGGI count, WASPADA count, AMAN count
2. **Batch Executive Summary**: Key stats from the latest triage run
3. **Threat Trend Chart** (7 days) + **PCA Semantic Map** (NusaBERT 768d → 2D)
4. **Triage Results Table**: Interactive `st.dataframe` with filters (hide AMAN toggle)
5. **Deep Analysis Panel** (The Radar): Activated by clicking any row — shows full Intelligence Brief with signals, CoT, feedback buttons

### PCA Semantic Map

The Peta Semantik Ancaman visualizes NusaBERT's 768-dimensional embeddings reduced to 2D via PCA:
- Each dot = one analyzed text, colored by label (🟢 AMAN, 🟡 WASPADA, 🔴 TINGGI)
- **Tight clusters of same color** → semantically similar threats (possible coordinated narratives)
- **Mixed-color overlap zones** → ambiguous content where the model is uncertain
- The explained variance percentage indicates how much information is preserved in the 2D projection

---

## 10.5. Automatic Speech Recognition (`src/asr/transcriber.py`)

### Model: Qwen3-ASR-0.6B

- **Release**: January 28, 2026 (Alibaba Qwen team)
- **Parameters**: 600M (smallest in the Qwen3-ASR family)
- **Languages**: 30+ including Indonesian (auto-detection)
- **Backend**: HuggingFace Transformers via `qwen-asr` package

### Pipeline

```
Audio File (wav/mp3/ogg/flac/m4a)
    │
    ▼
Qwen3-ASR-0.6B (GPU / bfloat16)
    │
    ├─→ Transcribed Text (Indonesian)
    │
    └─→ (Optional) run_pipeline() → IntelligenceBrief
```

### Functions

| Function | Purpose |
|---|---|
| `transcribe(audio_path, language)` | Transcribe a file → `{text, language}` |
| `transcribe_bytes(audio_bytes, filename)` | Transcribe from bytes (file upload) |
| `transcribe_and_analyze(audio_path)` | ASR → Threat Pipeline (1 LLM call) |
| `compute_wer(reference, hypothesis)` | WER/MER/WIL/WIP metrics via `jiwer` |

### WER Evaluation

Run `scripts/evaluate_asr.py` to evaluate on a reference test set (data/asr_test/references.json) or a synthetic demo. Report saved to `notebooks/reports/asr_wer_report.json`.

---

## 11. Issues Faced & Solutions

### The `transformers`/`setfit` Version Clash

**Problem**: `setfit 1.1.3` imports `default_logdir` from `transformers.training_args`, which was removed in `transformers >= 4.45`.

**Solution**: Surgical monkey-patch before any `setfit` import:
```python
import transformers.training_args as _ta
if not hasattr(_ta, "default_logdir"):
    def _default_logdir(): return str(Path("runs") / f"{datetime.now():%b%d_%H-%M-%S}_{socket.gethostname()}")
    _ta.default_logdir = _default_logdir
```

### OOM During SetFit Training

**Problem**: `NUM_ITERATIONS=20` (contrastive pairs per sample) caused memory starvation on standard hardware.

**Solution**: Reduced to `BATCH_SIZE=4`, `NUM_ITERATIONS=3`. Trading some embedding boundary sharpness for practical trainability, while still clearing all metric thresholds.

### Reddit Scraper Blocking

**Problem**: Reddit blocks bot User-Agents with Cloudflare captcha, returning HTML instead of JSON.

**Solution**: Dual-layer fallback: (1) `old.reddit.com` JSON with realistic browser UA, (2) RSS feed via `feedparser`. At least one always works.

### Streamlit Event Loop Error

**Problem**: `asyncio.get_event_loop()` fails in Streamlit's `ScriptRunner.scriptThread` with "no current event loop".

**Solution**: `run_agent()` always creates `asyncio.new_event_loop()` in a `ThreadPoolExecutor` background thread.

### Chat Rerun Loop

**Problem**: `st.text_input` retains its value across `st.rerun()`, causing infinite submission loops.

**Solution**: Switched to `on_change` callback pattern — the callback stores the message in `_agent_pending` and clears the input, then the main script processes pending messages on the next render cycle.

---

## 12. Technology Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.13, managed by `uv` |
| Web UI | Streamlit 1.55.0 |
| REST API | FastAPI + Uvicorn |
| LLM | Gemini 3 Flash (`gemini-3-flash-preview`) via LiteLLM |
| Agent Framework | Google ADK 1.26.0 |
| Local ML | SetFit on NusaBERT (`LazarusNLP/all-nusabert-base-v4`) |
| Database | DuckDB (embedded, columnar) |
| Indonesian NLP | Sastrawi (stemmer), custom slang dictionary |
| Scraping | requests + feedparser + YouTube Data API v3 |
| Data Validation | Pydantic v2 |
| ASR | Qwen3-ASR-0.6B via `qwen-asr` |
| WER Evaluation | `jiwer` |

---

## 13. How to Run

```bash
# 1. Install dependencies
uv sync

# 2. Set environment variables
cp .env.example .env
# Edit .env: set GEMINI_API_KEY and YOUTUBE_API_KEY

# 3. Start API server (terminal 1)
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 4. Start Streamlit UI (terminal 2)
uv run streamlit run ui/app.py

# 5. (Optional) Run model training
uv run python -m src.ml.train_setfit

# 6. (Optional) Run evaluation
uv run python scripts/evaluate.py
```

---

## 14. Docker Deployment

### 14.1 Architecture

```
docker-compose.yml
├── api (cartensz-api)      → FastAPI on :8000
├── ui (cartensz-ui)        → Streamlit on :8501, depends on api
└── scraper (cartensz-scraper) → cron: 6 AM WIB (23:00 UTC) daily OSINT
```

All 3 services share a `./data` volume mount for persistent DuckDB and model data.

### 14.2 Quick Start (Docker)

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Manual scrape trigger
docker exec cartensz-scraper python scripts/daily_scrape.py
```

### 14.3 Scheduled Scraper

The `scraper` service installs `cron` and registers a crontab entry:
```
0 23 * * *  →  6:00 AM WIB (UTC+7)
```

This calls `scripts/daily_scrape.py` which:
1. Scrapes all OSINT sources (Reddit, RSS, YouTube)
2. Sends texts to `/batch` endpoint in chunks of 20
3. Results are persisted to DuckDB automatically
4. Flags any TINGGI detections in the log

---

## 15. Potential Improvements

1. **Persistent Agent Memory**: Store chat history in DuckDB instead of `InMemorySessionService` so conversations survive server restarts.
2. **~~Custom Tokenizer Evaluation~~** — ✅ Covered in `02_deep_analysis.ipynb` Section 4
3. **Telegram Scraper**: Integrate Telethon for public channel monitoring.
4. **Active Learning Loop**: When feedback count ≥ 50, automatically trigger SetFit retraining with corrected labels mixed into the training set.
5. **Streaming Agent Responses**: Use Streamlit's `st.write_stream` for real-time agent output instead of waiting for full response.
6. **Confidence Calibration**: Post-hoc Platt scaling on the SetFit probabilities using the analyst feedback data.
7. **Real ASR Test Set**: Curate Indonesian audio test set with ground-truth transcriptions for WER benchmarking.
