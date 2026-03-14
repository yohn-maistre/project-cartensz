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
- **Orkestrator ADK**: Menyatukan semua tahapan agen di atas menjadi sarana alur kerja hulu-ke-hilir untuk analisis intelijen dengan dua mode kecepatan.
- **SetFit NusaBERT**: Algoritma model lokal yang bisa dilatih ulang dari basis model bahasa Indonesia yang mumpuni. Terdapat program pensintesis data buatan (Gemini 3 Flash) untuk mengatasi kelas TINGGI yang sangat sedikit muncul di alam liar.
- **API Backend FastAPI**: Menyediakan titik akhir antarmuka aplikasi terprogram (API) untuk dapat dihubungkan ke dasbor operasi maupun layanan pelaporan (*collector*).
- **Streamlit Command Center**: Antarmuka visual yang menampilkan data analisis ke dalam struktur ringkasan terperinci (*Intelligence Brief*), pemetaan visual (*PCA*), asisten *chat* fungsional (ADK Agent), dan formulir pelaporan cepat atas anomali identifikasi.
- **Docker Ready**: Proyek ini dapat langsung diluncurkan menggunakan Docker Compose untuk menyatukan API Backend dan UI Dasbor di bawah satu atap server.
- **Pengepul OSINT**: Modul penarik wacana publik (scrapers) secara mandiri dari Reddit, saluran berita RSS, dan kolom komentar video YouTube. Teks hasil operasi pengerukan dikemas dan dikirim ke sistem secara reguler.

