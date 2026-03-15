"""
Fase 8: Skrip Pengumpul OSINT (Ringan)
==============================================
Mengumpulkan data dari forum awam dan berita nasional:
  - Reddit JSON API (tanpa token API, via JSON)
  - Saluran RSS Berita Nasional (Detik, Tempo, Kompas)
  - Komentar YouTube Terkait Politik

Teks yang dikumpulkan akan digunakan untuk Cartensz Batch Triage
(model lokal SetFit untuk klasifikasi cepat tanpa biaya LLM).

Penggunaan:
    uv run python scripts/collector.py
    uv run python scripts/collector.py --dry-run   # uji coba tanpa mengirim data ke API
    uv run python scripts/collector.py --debug     # mode cetak detail

Disiapkan sebagai cronjob harian:
    0 0 * * * cd /path/to/gsp-eval && uv run python scripts/collector.py
"""
import asyncio
import os
import sys
import logging
import argparse
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

# Tambahkan struktur utama proyek ke dalam path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Konfigurasi sistem logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agen_pengepul_osint")

# Mengimpor feedparser untuk membaca saluran RSS
try:
    import feedparser
except ImportError:
    feedparser = None
    logger.warning("Modul feedparser tidak ditemukan. Sumber RSS akan dilewati. Gunakan: uv pip install feedparser")


# ─── Konfigurasi Target OSINT ────────────────────────────────────────────────────

# Gunakan header browser umum untuk melewati blokir Cloudflare
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

# Target Reddit: Menggunakan old.reddit.com untuk meminimalisasi pemblokiran
# Fallback RSS digunakan untuk memastikan sistem tahan terhadap galat JSON
REDDIT_TARGETS = [
    {
        "name": "r/indonesia (new)",
        "subreddit": "indonesia",
        "sort": "new",
        "limit": 50,
    },
    {
        "name": "r/indonesia (hot)",
        "subreddit": "indonesia",
        "sort": "hot",
        "limit": 25,
    },
]

# Target saluran RSS berita nasional
RSS_TARGETS = [
    {
        "name": "Detik News",
        "url": "https://news.detik.com/berita/rss",
    },
    {
        "name": "Tempo Nasional",
        "url": "http://rss.tempo.co/nasional",
    },
    {
        "name": "Detik Finance",
        "url": "https://finance.detik.com/rss",
    },
]

# Target video YouTube (komentar bertema politik)
YOUTUBE_TARGETS = [
    {
        "name": "YouTube: Diskusi Politik Terkini",
        "search_query": "politik indonesia terkini",
        "max_videos": 3,
        "max_comments_per_video": 20,
    },
]


# ─── Fungsi Pengumpul Data ─────────────────────────────────────────────────────────

def _reddit_json(subreddit: str, sort: str, limit: int, debug: bool = False) -> list[dict]:
    """Mengambil postingan Reddit via JSON API di old.reddit.com."""
    url = f"https://old.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8",
    }
    
    resp = requests.get(url, headers=headers, timeout=15, verify=False)
    resp.raise_for_status()
    
    content_type = resp.headers.get("content-type", "")
    if "json" not in content_type and "javascript" not in content_type:
        raise ValueError(f"Format data tidak dikenali. Diharapkan JSON (didapatkan: {content_type})")
    
    data = resp.json()
    texts = []
    children = data.get("data", {}).get("children", [])
    
    for post in children:
        pd = post.get("data", {})
        title = pd.get("title", "").strip()
        selftext = pd.get("selftext", "").strip()
        combined = f"{title}. {selftext}" if selftext else title
        
        if combined and len(combined) > 15:
            permalink = pd.get("permalink", "")
            post_url = f"https://reddit.com{permalink}" if permalink else url
            texts.append({"text": combined, "source": "Reddit", "url": post_url})
    
    return texts


def _reddit_rss(subreddit: str, sort: str, limit: int, debug: bool = False) -> list[dict]:
    """Mengambil postingan Reddit sebagai cadangan menggunakan saluran RSS."""
    if not feedparser:
        return []
    
    import re
    url = f"https://old.reddit.com/r/{subreddit}/{sort}/.rss?limit={limit}"
    
    feed = feedparser.parse(url, agent=HEADERS["User-Agent"])
    
    if feed.bozo and not feed.entries:
        raise ValueError(f"Gagal memilah saluran RSS: {feed.bozo_exception}")
    
    texts = []
    for entry in feed.entries:
        title = entry.get("title", "").strip()
        # Membersihkan tag HTML dari konten RSS
        content_parts = entry.get("content", [])
        if isinstance(content_parts, list) and content_parts:
            content_html = content_parts[0].get("value", "") if isinstance(content_parts[0], dict) else ""
        else:
            content_html = entry.get("summary", "")
        content_clean = re.sub(r'<[^>]+>', '', str(content_html)).strip()
        
        combined = f"{title}. {content_clean}" if content_clean and len(content_clean) > 10 else title
        link = entry.get("link", f"https://reddit.com/r/{subreddit}")
        
        if combined and len(combined) > 15:
            texts.append({"text": combined[:500], "source": "Reddit", "url": link})
    
    return texts


def scrape_reddit(target: dict, debug: bool = False) -> list[dict]:
    """Mengumpulkan postingan dari Reddit. Prioritaskan JSON, cadangkan ke RSS."""
    name = target["name"]
    sub = target["subreddit"]
    sort = target["sort"]
    limit = target.get("limit", 25)
    logger.info(f"🌐 Menelusuri Reddit: {name}")

    # Percobaan pertama: JSON endpoint (old.reddit.com)
    try:
        texts = _reddit_json(sub, sort, limit, debug)
        if texts:
            logger.info(f"  ✅ Berhasil mengunduh {len(texts)} data dari {name} (JSON)")
            if debug:
                for t in texts[:3]:
                    logger.debug(f"    -> {t['text'][:100]}...")
            time.sleep(2)
            return texts
        else:
            logger.warning(f"  ⚠️ Tidak mendapatkan data JSON untuk {name}. Beralih ke RSS...")
    except Exception as e:
        logger.warning(f"  ⚠️ Akses JSON untuk {name} gagal: {e}. Beralih ke RSS...")

    # Percobaan kedua: RSS endpoint
    try:
        texts = _reddit_rss(sub, sort, limit, debug)
        if texts:
            logger.info(f"  ✅ Berhasil mengunduh {len(texts)} data dari {name} (RSS cadangan)")
            if debug:
                for t in texts[:3]:
                    logger.debug(f"    -> {t['text'][:100]}...")
            time.sleep(1)
            return texts
        else:
            logger.warning(f"  ⚠️ Data RSS juga kosong untuk {name}.")
            return []
    except Exception as e:
        logger.error(f"  ❌ Gagal memproses JSON dan RSS pada {name}: {e}")
        return []


def scrape_rss(target: dict, debug: bool = False) -> list[dict]:
    """Mengekstraksi artikel berita dari saluran RSS."""
    if not feedparser:
        logger.warning(f"  ⚠️ Mengabaikan target RSS {target['name']} (modul feedparser tidak ada)")
        return []

    url = target["url"]
    name = target["name"]
    logger.info(f"📰 Membaca RSS: {name}")

    try:
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            logger.error(f"  ❌ Gagal memroses saluran RSS {name}: {feed.bozo_exception}")
            return []

        texts = []
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()

            import re
            summary_clean = re.sub(r'<[^>]+>', '', summary)

            combined = f"{title}. {summary_clean}" if summary_clean else title
            entry_url = entry.get("link", url)

            if combined and len(combined) > 20:
                texts.append({"text": combined, "source": f"RSS: {name}", "url": entry_url})

        logger.info(f"  ✅ Berhasil mendapatkan {len(texts)} artikel dari {name}")

        if debug and texts:
            for t in texts[:3]:
                logger.debug(f"    -> {t['text'][:100]}...")

        return texts

    except Exception as e:
        logger.error(f"  ❌ Mengalami kendala pada sumber RSS {name}: {e}.")
        return []


def scrape_youtube(target: dict, debug: bool = False) -> list[dict]:
    """
    Mengambil komentar YouTube publik menggunakan YouTube Data API v3.
    Membutuhkan Environment Variable YOUTUBE_API_KEY.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.warning(f"  ⚠️ Operasi YouTube diabaikan karena YOUTUBE_API_KEY tidak dikonfigurasi.")
        return []

    name = target["name"]
    query = target["search_query"]
    max_videos = target.get("max_videos", 3)
    max_comments = target.get("max_comments_per_video", 20)
    logger.info(f"🎬 Menelusuri YouTube: {name} (Kueri pencarian: '{query}')")

    texts = []

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 1. Mencari video berdasarkan kueri
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "relevanceLanguage": "id",
            "maxResults": max_videos,
            "order": "date",
            "key": api_key,
        }
        search_resp = requests.get(search_url, params=search_params, timeout=15, verify=False)
        search_resp.raise_for_status()
        videos = search_resp.json().get("items", [])
        
        logger.info(f"  📹 Menemukan {len(videos)} video untuk diproses.")

        # 2. Menarik komentar dari masing-masing video
        for video in videos:
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            video_url = f"https://youtube.com/watch?v={video_id}"
            
            if debug:
                logger.debug(f"    📹 Membaca video: {video_title}")

            comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
            comments_params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": max_comments,
                "order": "relevance",
                "key": api_key,
            }

            try:
                comments_resp = requests.get(comments_url, params=comments_params, timeout=15, verify=False)
                comments_resp.raise_for_status()
                threads = comments_resp.json().get("items", [])

                for thread in threads:
                    comment = thread["snippet"]["topLevelComment"]["snippet"]
                    comment_text = comment.get("textDisplay", "").strip()
                    
                    # Hilangkan tag HTML bawaan YouTube API
                    import re
                    comment_clean = re.sub(r'<[^>]+>', '', comment_text)
                    
                    if comment_clean and len(comment_clean) > 15:
                        texts.append({"text": comment_clean, "source": "YouTube", "url": video_url})

            except Exception as e:
                logger.warning(f"    ⚠️ Kolom komentar dinonaktifkan atau gagal ditarik untuk {video_id}: {e}")
                continue

            time.sleep(0.5)

        logger.info(f"  ✅ Ekstraksi selesai. Terkumpul {len(texts)} komentar dari {name}")

        if debug and texts:
            for t in texts[:3]:
                logger.debug(f"    -> {t['text'][:100]}...")

        return texts

    except Exception as e:
        logger.error(f"  ❌ Mengalami kegagalan sistem pada YouTube {name}: {e}")
        return []


# ─── Eksekusi Utama ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Script Ekstraktor OSINT untuk Project Cartensz")
    parser.add_argument("--debug", action="store_true", help="Mengaktifkan logging rinci")
    parser.add_argument("--dry-run", action="store_true", help="Uji coba ekstraksi tanpa mengirim data API")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("🛡️ Fase 8 Ekstraksi OSINT — Project Cartensz")
    logger.info(f"   Waktu eksekusi:  {datetime.now().isoformat()}")
    logger.info(f"   Target Reddit:    {len(REDDIT_TARGETS)}")
    logger.info(f"   Target RSS:       {len(RSS_TARGETS)}")
    logger.info(f"   Target YouTube:   {len(YOUTUBE_TARGETS)}")
    logger.info("=" * 60)

    all_items = []

    # 1. Scraping Reddit
    for target in REDDIT_TARGETS:
        items = scrape_reddit(target, debug=args.debug)
        all_items.extend(items)

    # 2. Parsing RSS News
    for target in RSS_TARGETS:
        items = scrape_rss(target, debug=args.debug)
        all_items.extend(items)

    # 3. Scraping YouTube Comments
    for target in YOUTUBE_TARGETS:
        items = scrape_youtube(target, debug=args.debug)
        all_items.extend(items)

    all_texts = [item["text"] for item in all_items]
    logger.info(f"\n📊 Total keseluruhan data teks diekstrak: {len(all_texts)}")

    if not all_texts:
        logger.warning("Tidak memperoleh data dari semua sasaran yang ada.")
        return

    logger.info("--- Cuplikan Data (5 baris pertama) ---")
    for i, t in enumerate(all_texts[:5], 1):
        logger.info(f"  [{i}] {t[:120]}...")

    if args.dry_run:
        logger.info("\n🏁 Eksekusi percobaan selesai (Dry Run). Melewati tahapan Triage API.")
        return

    # ─── Mengirim hasil pengumpulan ke Batch Triage (Tanpa LLM) ───
    api_url = os.getenv("API_URL", "http://localhost:8000")

    logger.info(f"\n⚡ Mentransfer {len(all_texts)} teks ke Cartensz API (Mode Triage)...")
    try:
        response = requests.post(
            f"{api_url}/batch",
            json={"texts": all_texts},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.error(f"Permintaan API Batch Triage ditolak dengan galat: {data.get('errors')}")
            return

        briefs = data.get("briefs", [])
        labels = [b["label"] for b in briefs]
        stats = {
            "total": len(briefs),
            "AMAN": labels.count("AMAN"),
            "WASPADA": labels.count("WASPADA"),
            "TINGGI": labels.count("TINGGI"),
        }

        logger.info("=" * 60)
        logger.info("🛡️ LAPORAN TRIAGE OTOMATIS SELESAI")
        logger.info(f"   Total Analisis: {stats['total']}")
        logger.info(f"   🟢 Kategori AMAN:    {stats['AMAN']}")
        logger.info(f"   🟡 Kategori WASPADA: {stats['WASPADA']}")
        logger.info(f"   🔴 Kategori TINGGI:  {stats['TINGGI']}")
        logger.info("=" * 60)

        # Soroti peringatan kritis
        tinggi_texts = [b for b in briefs if b["label"] == "TINGGI"]
        if tinggi_texts:
            logger.warning(f"\n🚨 Peringatan: Ditemukan {len(tinggi_texts)} data masuk dalam kategori ancaman TINGGI.")
            for t in tinggi_texts[:5]:
                logger.warning(f"   -> {t['text'][:100]}...")

    except Exception as e:
        logger.error(f"Gagal menghubungi server Triage API: {e}. (Akan diabaikan hingga siklus berikutnya)")

if __name__ == "__main__":
    main()
