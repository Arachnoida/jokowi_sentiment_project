"""
src/mongo_utils.py
Koneksi MongoDB, operasi insert, read, dan update untuk pipeline.
"""

from typing import Any, Dict, List, Optional

import certifi
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, ConnectionFailure

from src.utils import setup_logger

logger = setup_logger("mongo_utils")

# Cache koneksi agar tidak membuat MongoClient baru terus-menerus
_CLIENT_CACHE: Dict[str, MongoClient] = {}


def get_client(uri: str) -> MongoClient:
    """
    Buat atau ambil koneksi MongoDB dari cache.

    Raises:
        ConnectionFailure: jika tidak dapat terhubung ke MongoDB.
    """
    if uri in _CLIENT_CACHE:
        return _CLIENT_CACHE[uri]

    try:
        client: MongoClient = MongoClient(
            uri,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=30000,
            maxPoolSize=10,
        )

        # Test koneksi
        client.admin.command("ping")

        logger.info("Koneksi MongoDB berhasil.")

        _CLIENT_CACHE[uri] = client
        return client

    except ConnectionFailure as exc:
        logger.error(f"Gagal terhubung ke MongoDB: {exc}")
        raise


def get_collection(
    uri: str,
    db_name: str,
    collection_name: str,
) -> Collection:
    """
    Kembalikan objek Collection MongoDB.
    """
    client = get_client(uri)
    return client[db_name][collection_name]


def insert_many_safe(
    collection: Collection,
    documents: List[Dict[str, Any]],
    batch_size: int = 500,
) -> int:
    """
    Insert banyak dokumen dengan batching,
    skip duplikat berdasarkan _id.

    Returns:
        Jumlah dokumen yang berhasil diinsert.
    """
    if not documents:
        return 0

    total_inserted = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        try:
            result = collection.insert_many(batch, ordered=False)
            total_inserted += len(result.inserted_ids)

        except BulkWriteError as bwe:
            inserted = bwe.details.get("nInserted", 0)
            total_inserted += inserted
            logger.warning(
                f"BulkWriteError batch {i // batch_size + 1}: "
                f"{inserted} berhasil, ada duplikat yang di-skip."
            )

    return total_inserted


def setup_index(
    collection: Collection,
    field: str,
    unique: bool = True,
) -> None:
    """
    Buat index pada field tertentu.
    """
    collection.create_index(
        [(field, pymongo.ASCENDING)],
        unique=unique,
    )

    logger.info(
        f"Index dibuat pada field '{field}' "
        f"(unique={unique})"
    )


def count_documents(
    collection: Collection,
    query: Optional[Dict] = None,
) -> int:
    """
    Hitung jumlah dokumen dalam collection.
    """
    return collection.count_documents(query or {})


def find_all_as_dicts(
    collection: Collection,
    query: Optional[Dict] = None,
    projection: Optional[Dict] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """
    Baca dokumen dari collection
    dan kembalikan sebagai list dict.
    """
    proj = projection or {"_id": 0}

    cursor = collection.find(query or {}, proj)

    if limit > 0:
        cursor = cursor.limit(limit)

    return list(cursor)


def upsert_document(
    collection: Collection,
    filter_query: Dict[str, Any],
    update_data: Dict[str, Any],
) -> None:
    """
    Update dokumen jika ada,
    insert jika tidak ada.
    """
    collection.update_one(
        filter_query,
        {"$set": update_data},
        upsert=True,
    )


def close_client(uri: str) -> None:
    """
    Tutup koneksi MongoClient yang tersimpan di cache.
    """
    client = _CLIENT_CACHE.pop(uri, None)
    if client is not None:
        client.close()
        logger.info("Koneksi MongoDB ditutup.")