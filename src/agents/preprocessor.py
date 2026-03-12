"""
Agen Preprocessor - Normalisasi teks untuk konten berbahasa Indonesia.
Menangani: normalisasi bahasa tidak baku, deteksi percampuran kode bahasa, 
konversi emoji, penghapusan tautan, dan pencarian basis kalimat (stemming).
"""
import hashlib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from src.models import NormalizedText

# inisialisi sistem pemecah kata
_stemmer_factory = StemmerFactory()
_stemmer = _stemmer_factory.create_stemmer()

# pemetaan kata lisan ke ragam formal
SLANG_DICT: dict[str, str] = {
    "gw": "saya", "gue": "saya", "gua": "saya",
    "lu": "kamu", "lo": "kamu", "elu": "kamu",
    "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "udah": "sudah", "udh": "sudah",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena",
    "bgt": "banget", "bkn": "bukan", "blm": "belum",
    "org": "orang", "hrs": "harus", "bs": "bisa",
    "sm": "sama", "dr": "dari", "tp": "tapi",
    "emg": "memang", "emang": "memang",
    "kyk": "kayak", "kayak": "seperti",
    "dmn": "dimana", "gmn": "bagaimana", "gmna": "bagaimana",
    "bnyk": "banyak", "skrg": "sekarang", "trs": "terus",
    "lgsg": "langsung", "blg": "bilang", "ngmg": "ngomong",
    "wkwk": "", "wkwkwk": "", "haha": "", "hehe": "",
    "anjir": "anjing", "anjay": "anjing",
    "bokap": "ayah", "nyokap": "ibu",
}

# ubah lambang visual ke deskripsi tulisan
EMOJI_MAP: dict[str, str] = {
    "🔥": "[api]", "💀": "[tengkorak]", "⚔️": "[pedang]",
    "💣": "[bom]", "🔫": "[senjata]", "🗡️": "[pisau]",
    "✊": "[tinju]", "👊": "[pukul]", "😡": "[marah]",
    "🤬": "[marah]", "💪": "[kuat]", "⚠️": "[peringatan]",
    "🚨": "[darurat]", "☠️": "[bahaya]",
}

# penyaringan aksara arab
ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
# penyaringan abjad latin
LATIN_PATTERN = re.compile(r'[a-zA-Z]+')
# penyaringan tautan
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
# penyaringan rujukan user
MENTION_PATTERN = re.compile(r'@\w+')
# penyaringan tag
HASHTAG_PATTERN = re.compile(r'#(\w+)')


def _detect_language_mix(text: str) -> dict[str, float]:
    """taksir porsi teks berbahasa arab, latin, maupun sandi angka."""
    arabic_chars = len(ARABIC_PATTERN.findall(text))
    latin_chars = len(LATIN_PATTERN.findall(text))
    total = arabic_chars + latin_chars + 1  # cegah pembagian nol
    
    arabic_ratio = arabic_chars / total
    latin_ratio = latin_chars / total
    
    return {
        "id": round(latin_ratio, 3),  # estimasi: tulisan latin dominan
        "ar": round(arabic_ratio, 3),
        "other": round(1.0 - latin_ratio - arabic_ratio, 3),
    }


def _normalize_slang(text: str) -> str:
    """modifikasi wacana informal menjadi baku."""
    words = text.split()
    normalized = []
    for word in words:
        lower = word.lower()
        if lower in SLANG_DICT:
            replacement = SLANG_DICT[lower]
            if replacement:  # hindari modifikasi kosong
                normalized.append(replacement)
        else:
            normalized.append(word)
    return " ".join(normalized)


def _convert_emojis(text: str) -> tuple[str, bool]:
    """kembalikan teks berisi pertanda dari stiker atau gambar."""
    had_emoji = False
    for emoji, replacement in EMOJI_MAP.items():
        if emoji in text:
            had_emoji = True
            text = text.replace(emoji, f" {replacement} ")
    return text, had_emoji


def _extract_and_strip_urls(text: str) -> tuple[str, list[str]]:
    """kumpulkan dan hilangkan alamat web."""
    urls = URL_PATTERN.findall(text)
    cleaned = URL_PATTERN.sub("", text)
    return cleaned, urls


def preprocess(raw_text: str) -> NormalizedText:
    """
    pipeline pra-pemrosesan:
    1. eliminasi tautan
    2. konversi lambang gambar
    3. format tata bahasa
    4. singkirkan tanda pengguna
    5. pelacakan bahasa asing
    6. buat pelacak identitas (hash)
    """
    # 1. tautan
    text, urls = _extract_and_strip_urls(raw_text)
    
    # 2. emoji dan gambar
    text, has_emoji = _convert_emojis(text)
    
    # 3. ejaan
    text = _normalize_slang(text)
    
    # 4. identitas sosial (mention)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    
    # 5. reduksi jarak jeda (spasi ganda)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. deteksi kode bahasa berdasarkan teks awalan
    lang_mix = _detect_language_mix(raw_text)
    
    # 7. kalkulasi perhitungan kata minimum
    token_count = len(text.split())
    
    # 8. penomoran rekam jejak
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    
    return NormalizedText(
        original_text=raw_text,
        normalized_text=text,
        token_count=token_count,
        content_hash=content_hash,
        language_mix_ratios=lang_mix,
        has_emoji=has_emoji,
        stripped_urls=urls,
    )
