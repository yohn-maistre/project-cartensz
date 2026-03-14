"""
Daily Scrape & Triage — Automated OSINT collection for Project Cartensz.

Designed to run as a cron job (6 AM WIB / 23:00 UTC daily).
Scrapes all OSINT sources → sends to /batch endpoint for triage.
"""
import sys
import os
import time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_URL = os.getenv("API_URL", "http://api:8000")  # Docker service name


def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting daily OSINT scrape...")

    from scripts.collector import (
        scrape_reddit, scrape_rss, scrape_youtube,
        REDDIT_TARGETS, RSS_TARGETS, YOUTUBE_TARGETS,
    )

    all_items = []

    # Reddit
    try:
        for t in REDDIT_TARGETS:
            all_items.extend(scrape_reddit(t, debug=False))
        print(f"  Reddit: {len(all_items)} items")
    except Exception as e:
        print(f"  Reddit failed: {e}")

    # RSS
    rss_before = len(all_items)
    try:
        for t in RSS_TARGETS:
            all_items.extend(scrape_rss(t, debug=False))
        print(f"  RSS: {len(all_items) - rss_before} items")
    except Exception as e:
        print(f"  RSS failed: {e}")

    # YouTube
    yt_before = len(all_items)
    try:
        for t in YOUTUBE_TARGETS:
            all_items.extend(scrape_youtube(t, debug=False))
        print(f"  YouTube: {len(all_items) - yt_before} items")
    except Exception as e:
        print(f"  YouTube failed: {e}")

    # Normalize
    texts = []
    for it in all_items:
        if isinstance(it, dict) and it.get("text", "").strip():
            texts.append(it["text"])

    print(f"  Total texts collected: {len(texts)}")

    if not texts:
        print("  No texts to triage. Done.")
        return

    # Send to /batch endpoint in chunks of 20
    chunk_size = 20
    total_processed = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        try:
            resp = requests.post(f"{API_URL}/batch", json={"texts": chunk}, timeout=120)
            if resp.status_code == 200:
                briefs = resp.json().get("briefs", [])
                total_processed += len(briefs)
                # Count threats
                tinggi = sum(1 for b in briefs if b.get("label") == "TINGGI")
                if tinggi:
                    print(f"  ⚠️ Chunk {i//chunk_size + 1}: {tinggi} TINGGI detected!")
            else:
                print(f"  Chunk {i//chunk_size + 1} failed: {resp.status_code}")
        except Exception as e:
            print(f"  Chunk {i//chunk_size + 1} error: {e}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done. Processed {total_processed}/{len(texts)} texts.")


if __name__ == "__main__":
    main()
