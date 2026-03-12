from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

class NormalizedText(BaseModel):
    original_text: str
    normalized_text: str
    token_count: int
    content_hash: str # sha-256 untuk cache
    language_mix_ratios: dict[str, float] = Field(description="contoh {'id': 0.8, 'ar': 0.1, 'en': 0.1}")
    has_emoji: bool
    stripped_urls: List[str]

class SimilarExample(BaseModel):
    text: str
    label: Literal["AMAN", "WASPADA", "TINGGI"]
    relevance_score: float = Field(ge=0.0, le=1.0)
    source: str

class ClassificationResult(BaseModel):
    label: Literal["AMAN", "WASPADA", "TINGGI"]
    probabilities: dict[str, float] = Field(description="probabilitas setiap kelas")
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    entropy: float = Field(default=0.0, description="entropi shannon dari probabilitas. lebih tinggi berarti lebih ambigu.")
    prediction_set: List[str] = Field(default_factory=list, description="himpunan prediksi konformal. banyak label menandakan ambiguitas murni.")
    conformal_set_size: int = Field(default=1, description="ukuran himpunan prediksi. 1=pasti, 2+=ambigu.")
    reasoning: str = Field(description="alur pemikiran terstruktur untuk hasil klasifikasi ini")

    @model_validator(mode='after')
    def check_probs_sum(self):
        total = sum(self.probabilities.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"total probabilitas harus 1.0, angka saat ini {total}")
        return self

class CritiqueResult(BaseModel):
    counter_argument: str = Field(description="bermain peran menentang klasifikasi awal")
    counter_strength: float = Field(ge=0.0, le=1.0, description="seberapa kuat argumen bantahan ini?")
    revised_label: Optional[Literal["AMAN", "WASPADA", "TINGGI"]] = None
    confidence_impact: Literal["UPGRADE", "DOWNGRADE", "KEEP"]

class ThreatSignal(BaseModel):
    signal_type: Literal["ENTITY", "EUPHEMISM", "TEMPORAL", "CALL_TO_ACTION", "OTHER"]
    extracted_text: str
    significance: Literal["HIGH", "MEDIUM", "LOW"]
    context_explanation: str

class IntelligenceBrief(BaseModel):
    document_id: str
    summary_narrative: str = Field(description="ringkasan eksekutif 2-3 kalimat")
    classification: ClassificationResult
    signals_detected: List[ThreatSignal]
    ambiguity_notes: str = Field(description="wajib diisi. terangkan dengan jelas bagian yang belum pasti.")
    risk_score: int = Field(ge=0, le=100)
    recommendation: Literal["ARCHIVE", "MONITOR", "ESCALATE"]
    
    @model_validator(mode='after')
    def check_ambiguity(self):
        if not self.ambiguity_notes or len(self.ambiguity_notes.strip()) < 10:
            raise ValueError("catatan ambiguitas harus mendeskripsikan ketidakpastian.")
        return self
