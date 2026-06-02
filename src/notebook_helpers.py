"""
src/notebook_helpers.py
Helper untuk notebook Jupyter: preview, sanity check, display multi-video.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from src.utils import setup_logger, extract_and_validate_urls

logger = setup_logger("notebook_helpers")


def display_sample(
    records: List[Dict[str, Any]], n: int = 5, label: str = "Sample"
) -> None:
    """Tampilkan sample records dalam format yang mudah dibaca."""
    print(f"\n{'='*60}")
    print(f"  {label} ({min(n, len(records))} dari {len(records)} record)")
    print(f"{'='*60}")
    for i, rec in enumerate(records[:n]):
        print(f"\n[{i+1}] {json.dumps(rec, ensure_ascii=False, indent=2, default=str)}")
    print(f"{'='*60}\n")


def display_video_url_validation(
    valid: List[Tuple[str, str]],
    invalid: List[str],
) -> None:
    """Tampilkan hasil validasi daftar URL video."""
    print(f"\n{'='*60}")
    print(f"  VALIDASI URL VIDEO")
    print(f"{'='*60}")
    print(f"  Valid   : {len(valid)} URL")
    print(f"  Invalid : {len(invalid)} URL")
    print(f"{'='*60}")

    if valid:
        print("\n[VALID]")
        for i, (url, vid) in enumerate(valid):
            print(f"  [{i+1:02d}] {vid}  <-  {url}")

    if invalid:
        print("\n[TIDAK VALID]")
        for url in invalid:
            print(f"  [X] {url}")

    print()


def display_jobs_status(jobs: List[Dict[str, Any]]) -> None:
    """Tampilkan status semua job ingestion dalam bentuk tabel ringkas."""
    STATUS_ICON = {
        "pending": "[ ]",
        "running": "[~]",
        "completed": "[V]",
        "failed": "[X]",
    }
    print(f"\n{'='*75}")
    print(f"  STATUS INGESTION ({len(jobs)} video)")
    print(f"{'='*75}")
    print(f"  {'#':>3}  {'ICON':<5}  {'VIDEO ID':<13}  {'KOMENTAR':>10}  JUDUL")
    print(f"  {'-'*70}")
    for i, job in enumerate(jobs):
        icon = STATUS_ICON.get(job.get("status", ""), "[?]")
        vid = job.get("video_id", "")[:13]
        count = job.get("comment_count", 0) or 0
        title = (job.get("video_title") or "")[:35]
        print(f"  {i+1:>3}  {icon:<5}  {vid:<13}  {count:>10,}  {title}")
    print(f"{'='*75}\n")


def display_jobs_summary(summary: Dict[str, int]) -> None:
    """Tampilkan ringkasan jumlah job per status."""
    total = sum(summary.values())
    print(f"\n  Ringkasan Job:")
    print(f"    Pending   : {summary.get('pending', 0):>5}")
    print(f"    Running   : {summary.get('running', 0):>5}")
    print(f"    Completed : {summary.get('completed', 0):>5}")
    print(f"    Failed    : {summary.get('failed', 0):>5}")
    print(f"    Total     : {total:>5}\n")


def display_ingestion_results(results: List[Dict[str, Any]]) -> None:
    """Tampilkan hasil akhir setelah satu sesi ingestion batch selesai."""
    print(f"\n{'='*60}")
    print(f"  HASIL INGESTION BATCH")
    print(f"{'='*60}")
    for r in results:
        vid = r.get("video_id", "")
        status = r.get("status", "")
        count = r.get("comment_count") or 0
        err = r.get("error")
        line = f"  {vid}  {status:<20}  {count:>8,} komentar"
        if err:
            line += f"  ERR: {str(err)[:40]}"
        print(line)
    print(f"{'='*60}\n")


def sanity_check_env(config) -> bool:
    """Periksa variabel environment yang diperlukan."""
    issues = []
    if not config.youtube.API_KEY:
        issues.append("YOUTUBE_API_KEY belum di-set di .env")
    if not config.mongo.URI:
        issues.append("MONGO_URI belum di-set di .env")
    if not config.mongo.DB_NAME:
        issues.append("MONGO_DB_NAME belum di-set di .env")

    if issues:
        print("\n[PERINGATAN] Konfigurasi tidak lengkap:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\n[OK] Konfigurasi environment valid.")
    print(f"  MongoDB  : {config.mongo.URI} / {config.mongo.DB_NAME}")
    key_preview = config.youtube.API_KEY
    print(f"  API Key  : {'*' * 10}{key_preview[-4:] if key_preview else 'tidak ada'}")
    print(f"  Target   : {config.youtube.TARGET_COMMENT_COUNT} komentar per video")
    return True


def check_mongo_connection(mongo_uri: str) -> bool:
    """Uji koneksi MongoDB."""
    try:
        from src.mongo_utils import get_client
        client = get_client(mongo_uri)
        client.close()
        print(f"\n[OK] Koneksi MongoDB berhasil: {mongo_uri}")
        return True
    except Exception as exc:
        print(f"\n[GAGAL] Koneksi MongoDB gagal: {exc}")
        return False


def print_preprocessing_comparison(
    original: str, text_svm: str, text_bert: str
) -> None:
    """Tampilkan perbandingan preprocessing dua jalur."""
    print("\n" + "="*70)
    print("PERBANDINGAN PREPROCESSING")
    print("="*70)
    print(f"ORIGINAL : {original}")
    print(f"SVM Path : {text_svm}")
    print(f"BERT Path: {text_bert}")
    print("="*70 + "\n")
