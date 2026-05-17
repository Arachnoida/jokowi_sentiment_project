"""
src/utils.py
Helper umum: logging, validasi URL YouTube, path utilities.
"""

import re
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple


def setup_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Inisialisasi logger dengan output ke konsol dan file (opsional)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


_YT_PATTERNS = [
    r"(?:youtube\.com/watch\?(?:[^&]*&)*v=)([a-zA-Z0-9_-]{11})",
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})",
]


def extract_video_id(url: str) -> Optional[str]:
    """
    Ekstrak videoId dari berbagai format URL YouTube.

    Returns:
        videoId (11 karakter) atau None jika tidak ditemukan.

    Examples:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not url or not isinstance(url, str):
        return None
    for pattern in _YT_PATTERNS:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


def validate_video_id(video_id: Optional[str]) -> bool:
    """Validasi format videoId YouTube (11 karakter alfanumerik + _ + -)."""
    if not video_id:
        return False
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_id))


def extract_and_validate_urls(
    urls: List[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Proses daftar URL: ekstrak videoId dan pisahkan yang valid dari yang tidak.

    Args:
        urls: Daftar URL YouTube mentah.

    Returns:
        Tuple (valid_list, invalid_list) di mana valid_list berisi
        pasangan (url, video_id) dan invalid_list berisi URL yang gagal.
    """
    valid: List[Tuple[str, str]] = []
    invalid: List[str] = []
    seen_ids: set = set()

    for url in urls:
        vid = extract_video_id(url)
        if not validate_video_id(vid):
            invalid.append(url)
        elif vid in seen_ids:
            # URL duplikat (videoId sudah ada)
            invalid.append(f"DUPLIKAT: {url}")
        else:
            seen_ids.add(vid)
            valid.append((url, vid))

    return valid, invalid


def now_iso() -> str:
    """Kembalikan timestamp UTC sekarang dalam format ISO 8601."""
    return datetime.utcnow().isoformat()


def safe_int(value, default: int = 0) -> int:
    """Konversi value ke int dengan fallback default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
