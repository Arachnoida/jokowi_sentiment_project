"""
src/ingestion_tracker.py
Pelacak status ingestion per video YouTube di MongoDB.

Setiap video memiliki satu dokumen "job" di collection ingestion_jobs dengan
status: pending -> running -> completed / failed.
Memungkinkan ingestion di-resume tanpa mengulang video yang sudah selesai.
"""

from typing import Any, Dict, List, Optional
from pymongo.collection import Collection

from src.utils import now_iso, setup_logger

logger = setup_logger("ingestion_tracker")

# Nilai status yang valid
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def register_jobs(
    collection: Collection,
    video_entries: List[Dict[str, str]],
    overwrite_failed: bool = True,
) -> None:
    """
    Daftarkan daftar video sebagai job di MongoDB.
    Video yang sudah 'completed' tidak diubah.
    Video yang 'failed' di-reset ke 'pending' jika overwrite_failed=True.

    Args:
        collection: Collection MongoDB untuk ingestion_jobs.
        video_entries: List dict dengan key 'video_id', 'video_url'.
        overwrite_failed: Jika True, video gagal akan di-reset ke pending.
    """
    for entry in video_entries:
        video_id = entry["video_id"]
        existing = collection.find_one({"video_id": video_id})

        if existing:
            if existing["status"] == STATUS_COMPLETED:
                logger.info(f"[SKIP] {video_id}: sudah completed.")
                continue
            if existing["status"] == STATUS_FAILED and overwrite_failed:
                collection.update_one(
                    {"video_id": video_id},
                    {"$set": {
                        "status": STATUS_PENDING,
                        "error_message": None,
                        "started_at": None,
                        "completed_at": None,
                        "comment_count": 0,
                    }},
                )
                logger.info(f"[RESET] {video_id}: dari failed ke pending.")
            # Status running atau pending yang sudah ada: biarkan
        else:
            collection.insert_one({
                "video_id": video_id,
                "video_url": entry.get("video_url", ""),
                "video_title": entry.get("video_title"),
                "status": STATUS_PENDING,
                "comment_count": 0,
                "started_at": None,
                "completed_at": None,
                "error_message": None,
            })
            logger.info(f"[REGISTER] {video_id}: job baru didaftarkan.")


def mark_running(collection: Collection, video_id: str) -> None:
    """Tandai job sebagai sedang berjalan."""
    collection.update_one(
        {"video_id": video_id},
        {"$set": {"status": STATUS_RUNNING, "started_at": now_iso()}},
    )


def mark_completed(
    collection: Collection, video_id: str, comment_count: int
) -> None:
    """Tandai job sebagai berhasil selesai."""
    collection.update_one(
        {"video_id": video_id},
        {"$set": {
            "status": STATUS_COMPLETED,
            "completed_at": now_iso(),
            "comment_count": comment_count,
            "error_message": None,
        }},
    )
    logger.info(f"[COMPLETED] {video_id}: {comment_count} komentar.")


def mark_failed(
    collection: Collection, video_id: str, error_message: str
) -> None:
    """Tandai job sebagai gagal dengan pesan error."""
    collection.update_one(
        {"video_id": video_id},
        {"$set": {
            "status": STATUS_FAILED,
            "completed_at": now_iso(),
            "error_message": str(error_message),
        }},
    )
    logger.warning(f"[FAILED] {video_id}: {error_message}")


def update_title(
    collection: Collection, video_id: str, title: Optional[str]
) -> None:
    """Perbarui judul video di dokumen job."""
    if title:
        collection.update_one(
            {"video_id": video_id},
            {"$set": {"video_title": title}},
        )


def get_all_jobs(collection: Collection) -> List[Dict[str, Any]]:
    """Kembalikan semua job, diurutkan berdasarkan video_id."""
    return list(collection.find({}, {"_id": 0}).sort("video_id", 1))


def get_pending_jobs(collection: Collection) -> List[Dict[str, Any]]:
    """Kembalikan job yang masih pending (belum diproses)."""
    return list(
        collection.find({"status": STATUS_PENDING}, {"_id": 0})
    )


def get_jobs_summary(collection: Collection) -> Dict[str, int]:
    """Kembalikan ringkasan jumlah job per status."""
    pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    result = list(collection.aggregate(pipeline))
    summary = {STATUS_PENDING: 0, STATUS_RUNNING: 0,
               STATUS_COMPLETED: 0, STATUS_FAILED: 0}
    for item in result:
        summary[item["_id"]] = item["count"]
    return summary


def is_already_completed(collection: Collection, video_id: str) -> bool:
    """Cek apakah video sudah berhasil diproses sebelumnya."""
    doc = collection.find_one(
        {"video_id": video_id, "status": STATUS_COMPLETED}
    )
    return doc is not None
