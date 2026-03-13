"""
Kamus eufemisme dan bahasa terselubung Indonesia.
Memetakan kata-kata permukaan yang terlihat polos ke makna ancaman yang sebenarnya.
Digunakan oleh SignalExtractorAgent untuk mendeteksi bahasa ancaman yang disamarkan.
"""

# pemetaan eufemisme: kata -> { arti, tingkat ancaman, penjelasan }
EUPHEMISM_LEXICON: dict[str, dict] = {
    # eufemisme berorientasi radikal
    "jihad": {"meaning": "perjuangan", "threat_level": "CONTEXT_DEPENDENT", "note": "makna bervariasi bergantung situasi lapangan"},
    "istisyhad": {"meaning": "operasi bunuh diri", "threat_level": "HIGH", "note": "istilah pejuang martir sayap kanan"},
    "thogut": {"meaning": "pemerintah lalim", "threat_level": "MEDIUM", "note": "pelabelan antipati pihak penguasa"},
    "kafir harbi": {"meaning": "target perlawanan", "threat_level": "HIGH", "note": "konotasi dehumanisasi fatal"},
    "hijrah": {"meaning": "pindah aliansi", "threat_level": "CONTEXT_DEPENDENT", "note": "penanda reorganisasi atau perpindahan tempat"},
    "amaliyah": {"meaning": "kegiatan lapangan", "threat_level": "MEDIUM", "note": "sinyal ancaman mobilisasi massa"},
    
    # diksi ancaman sipil
    "bakar": {"meaning": "hasutan provokasi", "threat_level": "MEDIUM", "note": "seruan kerusuhan fisik"},
    "gebuk": {"meaning": "hantaman fisik", "threat_level": "MEDIUM", "note": "perintah ancaman penganiayaan"},
    "sapu bersih": {"meaning": "eliminasi sepihak", "threat_level": "HIGH", "note": "retorika intoleransi dan pembersihan"},
    "bersihkan": {"meaning": "pembersihan kelompok", "threat_level": "CONTEXT_DEPENDENT", "note": "sinyal pengusiran atau pembasmian"},
    "ganyang": {"meaning": "binasakan", "threat_level": "HIGH", "note": "pembawaan nada konfrontasi keras"},
    "usir": {"meaning": "usir paksa", "threat_level": "MEDIUM", "note": "pengusiran terorganisir kelompok minoritas"},
    
    # terminologi sapaan tertutup
    "akhi": {"meaning": "saudara kawan", "threat_level": "LOW", "note": "sapaan biasa namun sering menjadi penanda isolasi"},
    "ikhwan": {"meaning": "asosiasi internal", "threat_level": "CONTEXT_DEPENDENT", "note": "indikasi faksionalisme atau organisasi ekstrem"},
    "anshor": {"meaning": "penolong pihak luar", "threat_level": "CONTEXT_DEPENDENT", "note": "bermakna ganda sesuai situasi lapangan"},
    "mujahid": {"meaning": "pejuang tempur", "threat_level": "MEDIUM", "note": "identifikasi personel garis geras"},
    
    # penunjukan tenggat waktu
    "malam ini": {"meaning": "tenggat malam", "threat_level": "LOW", "note": "lampu kuning bila dirangkaikan instruksi operasional"},
    "segera": {"meaning": "gerak cepat", "threat_level": "LOW", "note": "urgensi eskalasi singkat"},
    "besok pagi": {"meaning": "inisiasi pagi", "threat_level": "LOW", "note": "penanda batas waktu gerakan"},
    "sudah waktunya": {"meaning": "waktu pelaksanaan", "threat_level": "MEDIUM", "note": "sinyal persiapan mobilisasi matang"},
    
    # formasi ideologi permusuhan
    "dizalimi": {"meaning": "perasaan dirugikan", "threat_level": "MEDIUM", "note": "justifikasi untuk retaliasi agresif"},
    "bangkit": {"meaning": "eskalasi", "threat_level": "CONTEXT_DEPENDENT", "note": "propaganda mobilisasi reaktif"},
    "perlawanan": {"meaning": "aksi balasan", "threat_level": "CONTEXT_DEPENDENT", "note": "ajakan memberontak terbuka"},
    "musuh": {"meaning": "target operasi", "threat_level": "MEDIUM", "note": "penanda oposisi tidak kompromi"},
    "pengkhianat": {"meaning": "desertir", "threat_level": "MEDIUM", "note": "label bagi informan atau pihak kompromis"},
}

# rumus deteksi seruan gerakan
CALL_TO_ACTION_PATTERNS = [
    r"ayo\s+(kita\s+)?",
    r"mari\s+(kita\s+)?",
    r"wajib\s+",
    r"harus\s+",
    r"saatnya\s+",
    r"jangan\s+diam\s+saja",
    r"bergerak\s*!",
    r"lawan\s*!",
    r"habisi\s*!",
    r"serang\s*!",
    r"turun\s+ke\s+jalan",
]

# rumus deteksi garis waktu darurat
TEMPORAL_URGENCY_PATTERNS = [
    r"malam\s+ini",
    r"besok\s+(pagi|siang|malam)",
    r"segera",
    r"sekarang\s+juga",
    r"detik\s+ini",
    r"sudah\s+waktunya",
    r"tidak\s+bisa\s+ditunda",
]
