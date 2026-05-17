"""
src/youtube_ingest.py
Pengambilan komentar dari YouTube Data API v3 dengan pagination.
Mendukung ingestion satu video maupun batch multi-video.
"""

import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from pymongo.collection import Collection

from src.utils import now_iso, safe_int, setup_logger
from src.mongo_utils import insert_many_safe
from src import ingestion_tracker as tracker

logger = setup_logger("youtube_ingest")


class YouTubeAPIError(Exception):
    pass


class QuotaExceededError(YouTubeAPIError):
    pass


def build_request_params(
    video_id: str,
    api_key: str,
    page_token: Optional[str] = None,
    max_results: int = 100,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "key": api_key,
        "videoId": video_id,
        "part": "snippet",
        "order": "relevance",
        "maxResults": max_results,
        "textFormat": "plainText",
    }
    if page_token:
        params["pageToken"] = page_token
    return params


def parse_comment_item(
    item: Dict[str, Any],
    video_id: str,
    source_url: str,
    source_title: Optional[str] = None,
) -> Dict[str, Any]:

    thread_snippet = item.get("snippet", {})

    top_comment = thread_snippet.get(
        "topLevelComment",
        {}
    )

    snippet = top_comment.get("snippet", {})

    return {

        # ==================================================
        # IDENTITAS
        # ==================================================
        "comment_id": top_comment.get("id", ""),
        "video_id": video_id,

        # ==================================================
        # TEKS
        # ==================================================
        "text": snippet.get("textDisplay", ""),
        "text_original": snippet.get("textOriginal", ""),

        # ==================================================
        # AUTHOR
        # ==================================================
        "author_display_name": snippet.get(
            "authorDisplayName"
        ),

        "author_profile_image": snippet.get(
            "authorProfileImageUrl"
        ),

        "author_channel_url": snippet.get(
            "authorChannelUrl"
        ),

        "author_channel_id": (
            snippet.get("authorChannelId", {})
            .get("value")
        ),

        # ==================================================
        # ENGAGEMENT
        # ==================================================
        "like_count": safe_int(
            snippet.get("likeCount", 0)
        ),

        "total_reply_count": safe_int(
            thread_snippet.get("totalReplyCount", 0)
        ),

        # ==================================================
        # WAKTU
        # ==================================================
        "published_at": snippet.get(
            "publishedAt"
        ),

        "updated_at": snippet.get(
            "updatedAt"
        ),

        # ==================================================
        # VIDEO
        # ==================================================
        "source_url": source_url,
        "source_title": source_title,

        "video_owner_channel_id": snippet.get(
            "videoOwnerChannelId"
        ),

        # ==================================================
        # METADATA API
        # ==================================================
        "can_rate": snippet.get("canRate"),

        "viewer_rating": snippet.get(
            "viewerRating"
        ),

        "etag": item.get("etag"),

        "fetched_at": now_iso(),

        # ==================================================
        # RAW JSON BACKUP
        # ==================================================
        "raw_json": item,
    }


def fetch_comments_page(
    endpoint: str,
    params: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Ambil satu halaman komentar dari API.

    Raises:
        QuotaExceededError: jika quota API habis.
        YouTubeAPIError: untuk error API lainnya.
    """
    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise YouTubeAPIError(f"Gagal terhubung ke API: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise YouTubeAPIError(f"Request timeout: {exc}") from exc

    if response.status_code == 403:
        body = response.json()
        errors = body.get("error", {}).get("errors", [])
        reasons = [e.get("reason", "") for e in errors]
        if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
            raise QuotaExceededError("Quota YouTube API telah habis.")
        raise YouTubeAPIError(f"HTTP 403: {body}")

    if response.status_code != 200:
        raise YouTubeAPIError(
            f"HTTP {response.status_code}: {response.text[:200]}"
        )
    return response.json()


def collect_comments(
    video_id: str,
    api_key: str,
    endpoint: str,
    source_url: str,
    source_title: Optional[str] = None,
    target_count: int = 10000,
    max_results_per_page: int = 100,
    delay_seconds: float = 0.5,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yang mengumpulkan komentar YouTube satu video secara bertahap.

    Yields:
        Dict komentar yang sudah diparsing.
    """
    if not api_key:
        raise YouTubeAPIError("API key YouTube tidak ditemukan di konfigurasi.")

    collected = 0
    page_token: Optional[str] = None
    page_num = 0

    while collected < target_count:
        page_num += 1
        params = build_request_params(
            video_id=video_id,
            api_key=api_key,
            page_token=page_token,
            max_results=min(max_results_per_page, target_count - collected),
        )

        try:
            data = fetch_comments_page(endpoint, params)
        except QuotaExceededError:
            logger.error("Quota API habis di halaman %d.", page_num)
            raise  # propagate agar batch handler bisa menghentikan loop
        except YouTubeAPIError as exc:
            logger.error("Error API halaman %d: %s", page_num, exc)
            break

        items = data.get("items", [])
        if not items:
            logger.info("Tidak ada komentar lagi di halaman %d.", page_num)
            break

        for item in items:
            comment = parse_comment_item(
                item, video_id, source_url, source_title
            )
            if comment["text"].strip():
                yield comment
                collected += 1
                if collected >= target_count:
                    break

        logger.info(
            "Halaman %d: %d komentar dikumpulkan (total: %d).",
            page_num, len(items), collected,
        )

        page_token = data.get("nextPageToken")
        if not page_token:
            logger.info("Tidak ada halaman berikutnya.")
            break

        time.sleep(delay_seconds)

    logger.info("Total komentar dikumpulkan untuk %s: %d", video_id, collected)


def get_video_title(video_id: str, api_key: str, base_url: str) -> Optional[str]:
    """Ambil judul video YouTube menggunakan endpoint videos."""
    try:
        url = f"{base_url}/videos"
        params = {"key": api_key, "id": video_id, "part": "snippet"}
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        items = data.get("items", [])
        if items:
            return items[0].get("snippet", {}).get("title")
    except Exception as exc:
        logger.warning("Gagal ambil judul video %s: %s", video_id, exc)
    return None


def ingest_single_video(
    video_id: str,
    video_url: str,
    api_key: str,
    endpoint: str,
    base_url: str,
    collection_raw: Collection,
    collection_jobs: Collection,
    target_count: int = 10000,
    max_results_per_page: int = 100,
    flush_every: int = 500,
    delay_seconds: float = 0.5,
) -> Tuple[bool, int]:
    """
    Orkestrasi ingestion satu video: ambil komentar, simpan ke MongoDB,
    dan update status job.

    Args:
        video_id: ID video YouTube.
        video_url: URL asli video.
        api_key: YouTube API key.
        endpoint: URL endpoint commentThreads.
        base_url: Base URL YouTube API.
        collection_raw: Collection MongoDB untuk raw comments.
        collection_jobs: Collection MongoDB untuk ingestion_jobs.
        target_count: Target jumlah komentar.
        max_results_per_page: Jumlah per halaman API.
        flush_every: Simpan ke MongoDB setiap N komentar.
        delay_seconds: Jeda antar halaman.

    Returns:
        Tuple (success: bool, comment_count: int).
    """
    # Ambil judul video
    title = get_video_title(video_id, api_key, base_url)
    tracker.update_title(collection_jobs, video_id, title)

    logger.info("Memulai ingestion: %s (%s)", video_id, title or "tanpa judul")
    tracker.mark_running(collection_jobs, video_id)

    buffer: List[Dict[str, Any]] = []
    total_inserted = 0

    try:
        gen = collect_comments(
            video_id=video_id,
            api_key=api_key,
            endpoint=endpoint,
            source_url=video_url,
            source_title=title,
            target_count=target_count,
            max_results_per_page=max_results_per_page,
            delay_seconds=delay_seconds,
        )

        for comment in gen:
            buffer.append(comment)
            if len(buffer) >= flush_every:
                inserted = insert_many_safe(collection_raw, buffer)
                total_inserted += inserted
                buffer.clear()

        # Flush sisa buffer
        if buffer:
            inserted = insert_many_safe(collection_raw, buffer)
            total_inserted += inserted

        tracker.mark_completed(collection_jobs, video_id, total_inserted)
        return True, total_inserted

    except QuotaExceededError as exc:
        # Quota habis: tandai failed dan re-raise agar loop batch berhenti
        tracker.mark_failed(collection_jobs, video_id, str(exc))
        raise

    except Exception as exc:
        tracker.mark_failed(collection_jobs, video_id, str(exc))
        logger.error("Gagal memproses %s: %s", video_id, exc)
        return False, total_inserted


def ingest_all_videos(
    video_entries: List[Dict[str, str]],
    api_key: str,
    endpoint: str,
    base_url: str,
    collection_raw: Collection,
    collection_jobs: Collection,
    target_count: int = 10000,
    max_results_per_page: int = 100,
    flush_every: int = 500,
    delay_seconds: float = 0.5,
    inter_video_delay: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Jalankan ingestion untuk semua video dalam daftar secara berurutan.
    Video yang sudah 'completed' dilewati otomatis.
    Jika quota habis, seluruh proses dihentikan untuk hari ini.

    Args:
        video_entries: List dict dengan key 'video_id', 'video_url'.
        inter_video_delay: Jeda antar video (detik) untuk menghindari rate limit.

    Returns:
        List hasil per video: [{video_id, status, comment_count, error}].
    """
    results = []

    for i, entry in enumerate(video_entries):
        video_id = entry["video_id"]
        video_url = entry["video_url"]

        # Cek apakah sudah completed
        if tracker.is_already_completed(collection_jobs, video_id):
            logger.info("[%d/%d] SKIP %s: sudah completed.",
                        i + 1, len(video_entries), video_id)
            results.append({
                "video_id": video_id,
                "status": "skipped_completed",
                "comment_count": None,
                "error": None,
            })
            continue

        logger.info("[%d/%d] Memproses %s...", i + 1, len(video_entries), video_id)

        try:
            success, count = ingest_single_video(
                video_id=video_id,
                video_url=video_url,
                api_key=api_key,
                endpoint=endpoint,
                base_url=base_url,
                collection_raw=collection_raw,
                collection_jobs=collection_jobs,
                target_count=target_count,
                max_results_per_page=max_results_per_page,
                flush_every=flush_every,
                delay_seconds=delay_seconds,
            )
            results.append({
                "video_id": video_id,
                "status": "completed" if success else "failed",
                "comment_count": count,
                "error": None,
            })
        except QuotaExceededError as exc:
            logger.error("Quota API habis. Menghentikan semua ingestion.")
            results.append({
                "video_id": video_id,
                "status": "quota_exceeded",
                "comment_count": 0,
                "error": str(exc),
            })
            break  # Hentikan seluruh batch

        # Jeda antar video
        if i < len(video_entries) - 1:
            logger.info("Jeda %s detik sebelum video berikutnya...", inter_video_delay)
            time.sleep(inter_video_delay)

    return results
