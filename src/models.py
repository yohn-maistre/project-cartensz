from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

class NormalizedText(BaseModel):
    original_text: str
    normalized_text: str
    token_count: int
    content_hash: str # sha-256 buat cache
    language_mix_ratios: dict[str, float] = Field(description="misal: {'id': 0.8, 'ar': 0.1, 'en': 0.1}")
    has_emoji: bool
    stripped_urls: List[str]

class ClassificationResult(BaseModel):
    label: Literal["AMAN", "WASPADA", "TINGGI"]
    probabilities: dict[str, float] = Field(description="peluang untuk tiap kelas")
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    entropy: float = Field(default=0.0, description="entropi shannon dr peluang. makin tinggi = makin ambigu")
    prediction_set: List[str] = Field(default_factory=list, description="himpunan prediksi konformal. dobel label = jujur ambigu")
    conformal_set_size: int = Field(default=1, description="ukuran himpunan prediksi. 1=yakin, 2+=rancu")
    reasoning: str = Field(description="alasan tersetruktur dr ai (cot)")

    @model_validator(mode='after')
    def check_probs_sum(self):
        total = sum(self.probabilities.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"total peluang kudu 1.0, dapetnya {total}")
        return self

class ThreatSignal(BaseModel):
    signal_type: Literal["ENTITY", "EUPHEMISM", "TEMPORAL", "CALL_TO_ACTION", "OTHER"]
    extracted_text: str
    significance: Literal["HIGH", "MEDIUM", "LOW"]
    context_explanation: str

class IntelligenceBrief(BaseModel):
    document_id: str
    summary_narrative: str = Field(description="ringkasan singkat 2-3 kalimat")
    classification: ClassificationResult
    signals_detected: List[ThreatSignal]
    ambiguity_notes: str = Field(description="NGGAK BOLEH KOSONG. sebutin bagian mana yg bikin ragu")
    risk_score: int = Field(ge=0, le=100)
    recommendation: Literal["ARCHIVE", "MONITOR", "ESCALATE"]
    
    @model_validator(mode='after')
    def check_ambiguity(self):
        if not self.ambiguity_notes or len(self.ambiguity_notes.strip()) < 10:
            raise ValueError("ambiguity_notes kudu nyebutin alasan ketidakpastian yg jelas.")
        return self
