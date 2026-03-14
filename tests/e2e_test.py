import asyncio
import json
import time
from src.agents.orchestrator import run_pipeline, batch_classify

async def test_radar():
    """Uji The Radar — analisis mendalam dengan 1 kali pemanfaatan otak LLM per teks."""
    test_texts = [
        # AMAN - Aman
        ("Kemacetan terpantau di Jalan Sudirman pagi ini. Banyak pekerja kantor yang terlambat karena penutupan jalan sementara.", "AMAN"),
        ("Resep soto betawi enak dan mudah dibuat di rumah. Bahan-bahannya simpel banget.", "AMAN"),
        # WASPADA - Abu-abu / Hati-hati
        ("Mereka dari barat mulai ikut campur urusan kita. Jangan biarkan mereka mengambil alih tanah leluhur. Mari bersiap sebelum terlambat.", "WASPADA"),
        # TINGGI - Ancaman Keras
        ("Pemerintahan thogut ini harus segera dihancurkan. Siapkan amaliyah malam ini di alun-alun kota, bawa 'kembang gula' untuk perayaan.", "TINGGI"),
        ("Kafir-kafir itu semakin berani. Darah mereka halal. Tunggu aba-aba selanjutnya untuk aksi besar.", "TINGGI"),
    ]

    print("=" * 60)
    print("🎯 THE RADAR — Analisis Ekstensif Terpusat (1 keping LLM per data)")
    print("=" * 60)
    
    for i, (text, expected) in enumerate(test_texts, 1):
        print(f"\n{'─' * 60}")
        print(f"KASUS UJI {i} (Ekspektasi: {expected}):")
        print(f"Teks: \"{text[:80]}...\"")
        print(f"{'─' * 60}")
        
        start = time.time()
        try:
            brief = await run_pipeline(text)
            elapsed = time.time() - start
            match = "✅" if brief.classification.label == expected else "⚠️"
            
            print(f"{match} Label Akhir: {brief.classification.label} (Pembanding: {expected})")
            print(f"  Keyakinan (Confidence): {brief.classification.confidence}")
            print(f"  Tingkat Bias (Entropy): {brief.classification.entropy:.4f}")
            print(f"  Set Konformal: {brief.classification.prediction_set}")
            print(f"  Skor Bahaya: {brief.risk_score}/100")
            print(f"  Tindakan: {brief.recommendation}")
            print(f"  Narasi: {brief.summary_narrative[:150]}...")
            print(f"  Anomali: {brief.ambiguity_notes[:150]}...")
            if brief.signals_detected:
                print(f"  Sinyal Terdeteksi ({len(brief.signals_detected)}):")
                for s in brief.signals_detected:
                    print(f"    - [{s.signal_type}] \"{s.extracted_text}\" ({s.significance})")
            print(f"  ⏱️ {elapsed:.1f} detik")
        except Exception as e:
            print(f"❌ Gugur Sistem: {e.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()


async def test_sweep():
    """Uji The Sweep — klasifikasi massal super kilat tanpa keterlibatan otak LLM."""
    texts = [
        "Cuaca cerah hari ini di Jakarta.",
        "Siapkan senjata untuk jihad melawan musuh Allah.",
        "Demo buruh menuntut kenaikan UMR besok.",
        "Kita harus bersatu melawan thogut, bawa peralatan malam ini.",
        "Ibu membuat kue untuk arisan RT.",
    ]

    print("\n\n" + "=" * 60)
    print("🔍 THE SWEEP — Triage Sapu Jagat (0 keping LLM)")
    print("=" * 60)
    
    start = time.time()
    results = await batch_classify(texts)
    elapsed = time.time() - start
    
    for i, r in enumerate(results, 1):
        icon = {"AMAN": "🟢", "WASPADA": "🟡", "TINGGI": "🔴"}.get(r["label"], "⚪")
        print(f"\n{icon} [{r['label']}] (conf={r['confidence']}, entropy={r['entropy']:.3f})")
        print(f"  Teks  : \"{r['text'][:80]}\"")
        print(f"  Konformal: {r['prediction_set']}")
        if r["signals"]:
            print(f"  Sinyal Terbaca: {r['signal_count']}")
            for s in r["signals"][:3]:
                print(f"    - [{s['type']}] \"{s['text']}\" ({s['significance']})")
    
    print(f"\n⏱️ Operasi massal untuk {len(texts)} rekaman tuntas dlm {elapsed:.2f} detik (tanpa biaya LLM)")


async def main():
    print("🚀 Pijakan Cartensz — Proses Automasi Pra-Peluncuran (E2E)\n")
    await test_radar()
    await test_sweep()
    print("\n\n✅ Seluruh simulasi tertutup dilalui dengan aman!")


if __name__ == "__main__":
    asyncio.run(main())
