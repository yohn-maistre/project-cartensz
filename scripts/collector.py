"""
Fase 8: Skrip Pengepul OSINT (Ringan)
==============================================
Menyapu bersih forum publik dan berita lokal menggunakan:
  - Reddit JSON API (tanpa token, cukup lewat jalur .json)
  - Saluran RSS Berita Nasional (Detik, Tempo, Kompas)
  - Komentar YouTube Politik

Tangkapan teks akan dituang paksa ke dalam Cartensz Batch Triage
(model lolal SetFit kecepatan tinggi, 0 tarikan LLM) guna labelisasi spontan.

Pemakaian:
    uv run python scripts/collector.py
    uv run python scripts/collector.py --dry-run   # uji jalan tanpa kirim api
    uv run python scripts/collector.py --debug     # mode cerewet

Disiapkan mantap sebagai dewan penjaga malam (cronjob):
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

# selipkan pondasi induk proyek ke path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# stel radio pancar log
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agen_pengepul_osint")

# coba bongkar feedparser buat baca rss; pasang pengingat jika bolong
try:
    import feedparser
except ImportError:
    feedparser = None
    logger.warning("pembaca feedparser mogok. sumber rss bakal diabaikan. Pasang: uv pip install feedparser")


# ─── konfigurasi peta target ────────────────────────────────────────────────────

# pakai tampang browser nyata biar nembus gerbang cloudflare reddit
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

# sasaran reddit — jalur tikus old.reddit.com (jarang kena blok) dengan jaring json + rss cadangan
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

# corong berita rss republik indonesia
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

# sumur video komentar youtube (kanal bahasan politik indo)
YOUTUBE_TARGETS = [
    {
        "name": "YouTube: Gonjang Ganjing Politik",
        "search_query": "politik indonesia terkini",
        "max_videos": 3,
        "max_comments_per_video": 20,
    },
]


# ─── alat keruk mesin ─────────────────────────────────────────────────────────

def _reddit_json(subreddit: str, sort: str, limit: int, debug: bool = False) -> list[dict]:
    """Coba gigit lewat pintu json old.reddit.com. Balikin array kosong kalo gagal."""
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
        raise ValueError(f"Ditolak gara-gara bukan file JSON ({content_type})")
    
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
    """Bantalan cadangan: keruk muatan postingan lewat suapan rss (feedparser). Pantang mundur."""
    if not feedparser:
        return []
    
    import re
    url = f"https://old.reddit.com/r/{subreddit}/{sort}/.rss?limit={limit}"
    
    feed = feedparser.parse(url, agent=HEADERS["User-Agent"])
    
    if feed.bozo and not feed.entries:
        raise ValueError(f"Alat pendedah rss eror parah: {feed.bozo_exception}")
    
    texts = []
    for entry in feed.entries:
        title = entry.get("title", "").strip()
        # buntelan rss aslinya html murni, cukur abis
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
    """Mengorek timbunan postingan di Reddit. Hantam pakai JSON dulu, kalau njeblok ganti RSS."""
    name = target["name"]
    sub = target["subreddit"]
    sort = target["sort"]
    limit = target.get("limit", 25)
    logger.info(f"🌐 Nyelam parit Reddit: {name}")

    # manuver 1: jembatan json (old.reddit.com)
    try:
        texts = _reddit_json(sub, sort, limit, debug)
        if texts:
            logger.info(f"  ✅ Sanggup mengait {len(texts)} bangkai postingan dari {name} (JSON)")
            if debug:
                for t in texts[:3]:
                    logger.debug(f"    → {t['text'][:100]}...")
            time.sleep(2)
            return texts
        else:
            logger.warning(f"  ⚠️ Tarikan JSON kopong untuk lokasi {name}, pindah gigi ke RSS cadangan...")
    except Exception as e:
        logger.warning(f"  ⚠️ Mesin JSON mogok depan gang {name}: {e}. Pindah gigi ke RSS...")

    # manuver 2: jalur darurat rss
    try:
        texts = _reddit_rss(sub, sort, limit, debug)
        if texts:
            logger.info(f"  ✅ Sanggup mengait {len(texts)} bangkai postingan {name} (RSS cadangan)")
            if debug:
                for t in texts[:3]:
                    logger.debug(f"    → {t['text'][:100]}...")
            time.sleep(1)
            return texts
        else:
            logger.warning(f"  ⚠️ Aliran RSS juga buntu melompong {name}.")
            return []
    except Exception as e:
        logger.error(f"  ❌ Gawat, dua mesin (JSON + RSS) berantakan di wilayah {name}: {e}")
        return []


def scrape_rss(target: dict, debug: bool = False) -> list[dict]:
    """Membedah buntelan feed RSS lalu melunturkan balasan berupa deret kamus."""
    if not feedparser:
        logger.warning(f"  ⚠️ Lompati sasaran RSS {target['name']} (modul feedparser raib)")
        return []

    url = target["url"]
    name = target["name"]
    logger.info(f"📰 Buka gulungan RSS: {name}")

    try:
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            logger.error(f"  ❌ Gagal total mendedah RSS {name}: {feed.bozo_exception}")
            return []

        texts = []
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()

            # kikis kotoran tag html bawaan rss (standar)
            import re
            summary_clean = re.sub(r'<[^>]+>', '', summary)

            combined = f"{title}. {summary_clean}" if summary_clean else title
            entry_url = entry.get("link", url)

            if combined and len(combined) > 20:
                texts.append({"text": combined, "source": f"RSS: {name}", "url": entry_url})

        logger.info(f"  ✅ Panen tarikan {len(texts)} warta artikel {name}")

        if debug and texts:
            for t in texts[:3]:
                logger.debug(f"    → {t['text'][:100]}...")

        return texts

    except Exception as e:
        logger.error(f"  ❌ Kandas sedot saluran RSS {name}: {e}")
        return []


def scrape_youtube(target: dict, debug: bool = False) -> list[dict]:
    """
    Nyerok lumbung ocehan bawahan YouTube di lapak politik pakai jalur resmi v3.
    Butuh umpan variabel YOUTUBE_API_KEY (gratisan dapet 10k koin/hari).
    
    Ritual: Geledah video → Bongkar benang komentar → Peras sari teksnya.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.warning(f"  ⚠️ Cium bau anyir YouTube dilewatkan (kunci gembok YOUTUBE_API_KEY luntang lantung)")
        return []

    name = target["name"]
    query = target["search_query"]
    max_videos = target.get("max_videos", 3)
    max_comments = target.get("max_comments_per_video", 20)
    logger.info(f"🎬 Nyebar jala di laut YouTube: {name} (kata sakti: '{query}')")

    texts = []

    try:
        # acuhkan omelan kehati-hatian gembok ganda ssl lokal
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Babak 1: Sorot gulungan pita video lokal anyar
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
        
        logger.info(f"  📹 Temukan {len(videos)} mangsa video")

        # Babak 2: Perah sumsum komentar di tiap badan video
        for video in videos:
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            video_url = f"https://youtube.com/watch?v={video_id}"
            
            if debug:
                logger.debug(f"    📹 Incaran gambar obah: {video_title}")

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
                    
                    # sapu bersih noda markup html
                    import re
                    comment_clean = re.sub(r'<[^>]+>', '', comment_text)
                    
                    if comment_clean and len(comment_clean) > 15:
                        texts.append({"text": comment_clean, "source": "YouTube", "url": video_url})

            except Exception as e:
                logger.warning(f"    ⚠️ Layanan ngoceh ditutup/mampet untuk video urutan {video_id}: {e}")
                continue

            time.sleep(0.5)  # kalem dikit patuhi batas dewa api

        logger.info(f"  ✅ Bongkar muat {len(texts)} celotehan terekstrak dari {name}")

        if debug and texts:
            for t in texts[:3]:
                logger.debug(f"    → {t['text'][:100]}...")

        return texts

    except Exception as e:
        logger.error(f"  ❌ Hancur berantakan pas nyedot YouTube {name}: {e}")
        return []


# ─── dapur pusaka ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mesin Penadah Gelap OSINT buat Cartensz")
    parser.add_argument("--debug", action="store_true", help="Nyalakan senter silau")
    parser.add_argument("--dry-run", action="store_true", help="Gali tanahnya aja, gak usah lempar parsel")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("🛡️  Fase 8 Alat Pengepul OSINT — Project Cartensz")
    logger.info(f"   Jam dinding:         {datetime.now().isoformat()}")
    logger.info(f"   Jumlah lobang Reddit: {len(REDDIT_TARGETS)}")
    logger.info(f"   Pipa corong RSS:      {len(RSS_TARGETS)}")
    logger.info(f"   Jaring bule YouTube:  {len(YOUTUBE_TARGETS)}")
    logger.info("=" * 60)

    all_items = []  # wadah asbak diktori teks mentah bersumber

    # 1. Obrak-abrik Reddit
    for target in REDDIT_TARGETS:
        items = scrape_reddit(target, debug=args.debug)
        all_items.extend(items)

    # 2. Bekam Siaran RSS
    for target in RSS_TARGETS:
        items = scrape_rss(target, debug=args.debug)
        all_items.extend(items)

    # 3. Kuras Sumur Lendir Kolom YouTube
    for target in YOUTUBE_TARGETS:
        items = scrape_youtube(target, debug=args.debug)
        all_items.extend(items)

    all_texts = [item["text"] for item in all_items]
    logger.info(f"\n📊 Total gerbong bawaan mentah kerukan: {len(all_texts)}")

    if not all_texts:
        logger.warning("Nol hasil tangkapan. Pulang kampung tangan kosong mending gih.")
        return

    # icip dulu dikit
    logger.info("--- Sedotan Uji (5 porsi awal) ---")
    for i, t in enumerate(all_texts[:5], 1):
        logger.info(f"  [{i}] {t[:120]}...")

    if args.dry_run:
        logger.info("\n🏁 Sandiwara lari kering beres. Skip ngelempar kurir API.")
        return

    # ─── Salurkan pipa corong langsung ke meja periksa Batch Triage kilat (0-Bakar LLM) ───
    api_url = os.getenv("API_URL", "http://localhost:8000")

    logger.info(f"\n⚡ Nembak {len(all_texts)} selongsong peluru teks ke mulut bot Cartensz Triage Kilat...")
    try:
        response = requests.post(
            f"{api_url}/batch",
            json={"texts": all_texts},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.error(f"Sistem gerbang Batch keno eror parah: {data.get('errors')}")
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
        logger.info("🛡️  PENGAYAKAN KILAT RAMPUNG")
        logger.info(f"   Jumlah Jaring: {stats['total']}")
        logger.info(f"   🟢 Golongan AMAN:    {stats['AMAN']}")
        logger.info(f"   🟡 Kelas WASPADA: {stats['WASPADA']}")
        logger.info(f"   🔴 Zona TINGGI:  {stats['TINGGI']}")
        logger.info("=" * 60)

        # Sorot merah data menyimpang beringas
        tinggi_texts = [b for b in briefs if b["label"] == "TINGGI"]
        if tinggi_texts:
            logger.warning(f"\n🚨 Waspada, {len(tinggi_texts)} biji ampas terdeteksi di KUADRAN BERBAHAYA!")
            for t in tinggi_texts[:5]:
                logger.warning(f"   → {t['text'][:100]}...")

    except Exception as e:
        logger.error(f"Ditolak mentah-mentah gerbang meja Triage API: {e}")


if __name__ == "__main__":
    main()
