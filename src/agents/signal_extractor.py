"""
SignalExtractorAgent - Mengekstrak sinyal ancaman dari teks berbahasa Indonesia.
Menggunakan: pencocokan kamus eufemisme, pola urgensi waktu, dan deteksi 
panggilan aksi (call-to-action). 100% Python murni, tanpa panggilan LLM.
"""
import re
from src.models import NormalizedText, ThreatSignal
from src.lexicon import (
    EUPHEMISM_LEXICON,
    CALL_TO_ACTION_PATTERNS,
    TEMPORAL_URGENCY_PATTERNS,
)


def _find_euphemisms(text: str) -> list[ThreatSignal]:
    """periksa kehadiran kata bersandi."""
    signals = []
    text_lower = text.lower()
    for term, info in EUPHEMISM_LEXICON.items():
        if term.lower() in text_lower:
            # kalkulasi persentase krisis
            threat = info["threat_level"]
            if threat == "HIGH":
                sig = "HIGH"
            elif threat == "MEDIUM":
                sig = "MEDIUM"
            else:
                sig = "LOW"

            signals.append(ThreatSignal(
                signal_type="EUPHEMISM",
                extracted_text=term,
                significance=sig,
                context_explanation=f"{info['meaning']} - {info['note']}",
            ))
    return signals


def _find_call_to_action(text: str) -> list[ThreatSignal]:
    """cari intrik seruan penggalangan massa."""
    signals = []
    text_lower = text.lower()
    for pattern in CALL_TO_ACTION_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            # potong teks sekitar penemuan pola sandi
            match_obj = re.search(pattern, text_lower)
            if match_obj:
                start = max(0, match_obj.start() - 20)
                end = min(len(text_lower), match_obj.end() + 40)
                context = text[start:end].strip()
                signals.append(ThreatSignal(
                    signal_type="CALL_TO_ACTION",
                    extracted_text=context,
                    significance="MEDIUM",
                    context_explanation=f"ditemukan propaganda lapangan berformat: '{pattern.strip()}'",
                ))
    return signals


def _find_temporal_urgency(text: str) -> list[ThreatSignal]:
    """temukan penanda batas waktu operasional."""
    signals = []
    text_lower = text.lower()
    for pattern in TEMPORAL_URGENCY_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            start = max(0, match.start() - 15)
            end = min(len(text_lower), match.end() + 30)
            context = text[start:end].strip()
            signals.append(ThreatSignal(
                signal_type="TEMPORAL",
                extracted_text=match.group(),
                significance="LOW",
                context_explanation=f"ditemukan urgensi jadwal berdekatan: '{context}'",
            ))
    return signals


def _detect_code_switching_signals(normalized: NormalizedText) -> list[ThreatSignal]:
    """observasi pertukaran bahasa arab-latin sebagai tanda indikasi."""
    signals = []
    ar_ratio = normalized.language_mix_ratios.get("ar", 0.0)
    if ar_ratio > 0.05:  # pantau bila bahasa arab melebihi ambang
        significance = "HIGH" if ar_ratio > 0.2 else "MEDIUM" if ar_ratio > 0.1 else "LOW"
        signals.append(ThreatSignal(
            signal_type="OTHER",
            extracted_text=f"rasio kehadiran alfabet arab: {ar_ratio:.1%}",
            significance=significance,
            context_explanation=(
                f"menemukan komposisi tulisan timur tengah hingga {ar_ratio:.1%}. "
                "transliterasi arab-indonesia dapat digunakan dalam ekstremisme, "
                "padahal tak jarang berupa ujaran spiritual normatif. perlu peninjauan ulas."
            ),
        ))
    return signals


def extract_signals(original_text: str, normalized: NormalizedText) -> list[ThreatSignal]:
    """
    terapkan rute ekstraksi sinyal utuh memakai bahasa dasar python saja.
    
    skema rute:
    1. pengenalan eufemisme
    2. lacak ajakan bersatu
    3. tangkap desakan waktu
    4. bedah pertukaran alih bahasa
    """
    signals: list[ThreatSignal] = []
    
    # sisir menggunakan wacana primer yang memuat kata berantakan dan ekspresi
    signals.extend(_find_euphemisms(original_text))
    signals.extend(_find_call_to_action(original_text))
    signals.extend(_find_temporal_urgency(original_text))
    signals.extend(_detect_code_switching_signals(normalized))
    
    # filter ganda sinyal ekstrakan
    seen = set()
    unique_signals = []
    for s in signals:
        key = (s.signal_type, s.extracted_text)
        if key not in seen:
            seen.add(key)
            unique_signals.append(s)
    
    return unique_signals
