# Project Cartensz

Analisis ancaman untuk teks bahasa Indonesia. Sistem ini mengklasifikasikan teks ke dalam tingkat risiko (Aman/Waspada/Tinggi) dan mengekstrak sinyal ancaman.

## Instalasi

```bash
git clone https://github.com/yosegiyay/gsp-threat-classifier.git
cd gsp-threat-classifier
pip install -r requirements.txt
cp .env.example .env
```

## Fitur

- **Preproses Teks**: Agen khusus untuk membersihkan teks mentah. Dapat membuang URL, mengubah emoji ancaman menjadi penanda teks (misalnya 💣 menjadi [bom]), dan membakukan bahasa gaul Indonesia agar lebih mudah dianalisis.
- **Kamus Eufemisme**: Dilengkapi dengan kamus kode, bahasa gaul ancaman, dan pola kalimat yang menunjukkan urgensi untuk mendeteksi ancaman terselubung.
- **Ekstraktor Sinyal**: Mengekstrak sinyal ancaman menggunakan pola RegEx murni tanpa LLM untuk menjamin kecepatan proses.
- **Klien LLM Pintar**: Menggunakan `litellm` sebagai jembatan ke model bahasa (Gemini/Ollama) dengan fitur rotasi API Key otomatis jika terkena *rate-limit* dan penyimpanan (*caching*) respons bawaan agar lebih hemat kuota.
- **Pencatat Analisis DuckDB**: Menyimpan sistem umpan balik pengguna dan riwayat identifikasi ke dalam sistem basis data analitik supersingkat (DuckDB).
- **Brief Writer Agent**: Agen hulu yang mengintegrasikan klasifikasi LLM dengan hasil machine learning kecil lokal menjadi laporan singkat intelijen (Intelligence Briefing) berstruktur yang aman.
