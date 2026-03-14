"""
Skrip Pengukur Kesalahan Kata (WER) Qwen3-ASR-0.6B buat audio Indonesia.

Cara pakai:
    uv run python scripts/evaluate_asr.py

Skrip ini ngetes performa agen ASR dari cuplikan suara, lalu ngeluarin skor WER.
Kalau dataset asli kosong, dia bakal muter data simulasi buat contoh tok.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "notebooks" / "reports"
ASR_TEST_DIR = DATA_DIR / "asr_test"


def evaluate_on_references():
    """
    Hitung WER dari himpunan data referensi.
    
    Format wajib di data/asr_test/:
        - file audio: *.wav, *.mp3, dll.
        - references.json: {"namaberkas.wav": "teks transkripsi asli", ...}
    """
    from src.asr.transcriber import transcribe, compute_wer

    ref_file = ASR_TEST_DIR / "references.json"
    if not ref_file.exists():
        print(f"Berkas rujukan ga ketemu di {ref_file}")
        print("Tolong bikin data/asr_test/references.json dengan format:")
        print('  {"audio1.wav": "teks ucapan asli bahasa indonesia", ...}')
        print("\nOtomatis pindah ke mode evaluasi purwarupa dengan data buatan...\n")
        return demo_evaluation()

    with open(ref_file, "r", encoding="utf-8") as f:
        references = json.load(f)

    results = []
    total_wer = 0.0
    n = 0

    for filename, reference_text in references.items():
        audio_path = ASR_TEST_DIR / filename
        if not audio_path.exists():
            print(f"  ⚠️ Lewati {filename} — berkar audio fisiknya ilang")
            continue

        print(f"  Mengurai {filename}...")
        try:
            asr_result = transcribe(str(audio_path), language="Indonesian")
            hypothesis = asr_result["text"]
            wer_result = compute_wer(reference_text, hypothesis)

            results.append({
                "file": filename,
                "reference": reference_text,
                "hypothesis": hypothesis,
                "wer": wer_result["wer"],
                "substitutions": wer_result["substitutions"],
                "deletions": wer_result["deletions"],
                "insertions": wer_result["insertions"],
            })

            total_wer += wer_result["wer"]
            n += 1
            status = "✅" if wer_result["wer"] < 0.15 else "⚠️" if wer_result["wer"] < 0.30 else "❌"
            print(f"    {status} WER: {wer_result['wer']:.2%} | S:{wer_result['substitutions']} D:{wer_result['deletions']} I:{wer_result['insertions']}")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({"file": filename, "error": str(e)})

    if n > 0:
        avg_wer = total_wer / n
        print(f"\n{'=' * 50}")
        print(f"  Rata-rata WER: {avg_wer:.2%} (dari {n} dokumen)")
        print(f"  {'✅ BAGUS' if avg_wer < 0.15 else '⚠️ BISA DITERIMA' if avg_wer < 0.30 else '❌ PERLU DILATIH LAGI'}")
        print(f"{'=' * 50}")

        # Simpan laporan
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "model": "Qwen/Qwen3-ASR-0.6B",
            "language": "Indonesian",
            "num_files": n,
            "average_wer": round(avg_wer, 4),
            "results": results,
        }
        report_path = REPORTS_DIR / "asr_wer_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Rekap kesimpulan tercatat di {report_path}")


def demo_evaluation():
    """
    Simulasi rasio WER menggunakan pasagan cuplikan bayangan.
    Biar kelihatan gimana jalannya pengujian tanpa harus punya master audionya.
    """
    from src.asr.transcriber import compute_wer

    print("Demo Penilaian WER (Data Simulasi)")
    print("=" * 50)

    # Tiruan pasangan rujukan-tebakan (keluaran tipikal dari asr)
    test_pairs = [
        {
            "name": "Suara jernih (berita)",
            "reference": "Presiden Joko Widodo meresmikan jembatan baru di Kalimantan Timur hari ini",
            "hypothesis": "Presiden Joko Widodo meresmikan jembatan baru di Kalimantan Timur hari ini",
        },
        {
            "name": "Omongan santai (medsos)",
            "reference": "mereka bilang harus turun ke jalan besok pagi untuk demo di depan gedung DPR",
            "hypothesis": "mereka bilang harus turun ke jalan besok pagi untuk demo di depan gedung DPR",
        },
        {
            "name": "Suara berisik",
            "reference": "siapkan diri untuk aksi besar besok malam di alun alun",
            "hypothesis": "siapkan diri untuk aksi besar besok malam di alun-alun",
        },
        {
            "name": "Campur logat (ID/AR)",
            "reference": "para ikhwan harus bersatu melawan thogut yang menzalimi umat",
            "hypothesis": "para ikhwan harus bersatu melawan togut yang menzalimi umat",
        },
    ]

    total_wer = 0.0
    results = []
    for pair in test_pairs:
        wer_result = compute_wer(pair["reference"], pair["hypothesis"])
        total_wer += wer_result["wer"]
        results.append({**pair, "wer": wer_result["wer"]})
        status = "✅" if wer_result["wer"] < 0.15 else "⚠️" if wer_result["wer"] < 0.30 else "❌"
        print(f"  {status} {pair['name']}: WER = {wer_result['wer']:.2%}")

    avg_wer = total_wer / len(test_pairs)
    print(f"\n  Rata-rata WER: {avg_wer:.2%}")
    print(f"  (Ini cuma demo pake data tempelan — pengetesan beneran kudu pake suara asli)")

    # Simpan laporan purwarupa
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "mode": "synthetic_demo",
        "num_pairs": len(test_pairs),
        "average_wer": round(avg_wer, 4),
        "results": results,
    }
    report_path = REPORTS_DIR / "asr_wer_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Laporan rakitan tersimpan ke {report_path}")


if __name__ == "__main__":
    print("🎤 Qwen3-ASR-0.6B — Pengukuran Nilai WER")
    print("=" * 50)
    evaluate_on_references()
