import os
import sqlite3
import uuid
from datetime import datetime

from src.config import HISTORY_DB_PATH, HISTORY_IMAGE_DIR


def _connect():
    connection = sqlite3.connect(HISTORY_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_history_storage():
    os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
    os.makedirs(HISTORY_IMAGE_DIR, exist_ok=True)

    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_image_name TEXT,
                predicted_label TEXT NOT NULL,
                raw_label TEXT,
                confidence REAL,
                confidence_text TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )
        connection.commit()


def _build_image_filename():
    return f"{uuid.uuid4().hex}.jpg"


def save_history_record(image, original_filename, result_payload):
    ensure_history_storage()

    stored_image_name = _build_image_filename()
    stored_image_path = os.path.join(HISTORY_IMAGE_DIR, stored_image_name)
    image.save(stored_image_path, format="JPEG", quality=92)

    predicted_label = result_payload.get("prediction") or result_payload.get("class") or "Unknown"
    confidence_value = result_payload.get("confidence")
    confidence_numeric = confidence_value if isinstance(confidence_value, (int, float)) else None
    confidence_text = (
        f"{confidence_value * 100:.2f}%"
        if isinstance(confidence_value, (int, float))
        else str(confidence_value or "0%")
    )
    status = result_payload.get("status", "success")
    message = result_payload.get("message", "Prediction completed.")

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO detection_history (
                created_at,
                original_filename,
                stored_image_name,
                predicted_label,
                raw_label,
                confidence,
                confidence_text,
                status,
                message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                original_filename,
                stored_image_name,
                predicted_label,
                result_payload.get("raw_label"),
                confidence_numeric,
                confidence_text,
                status,
                message,
            ),
        )
        connection.commit()
        record_id = cursor.lastrowid

    return get_history_record(record_id)


def _row_to_record(row):
    if row is None:
        return None

    record = dict(row)
    record["image_url"] = (
        f"/history/{record['id']}/image" if record.get("stored_image_name") else None
    )
    return record


def list_history_records(limit=20):
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, original_filename, stored_image_name, predicted_label,
                   raw_label, confidence, confidence_text, status, message
            FROM detection_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [_row_to_record(row) for row in rows]


def get_history_record(record_id):
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, created_at, original_filename, stored_image_name, predicted_label,
                   raw_label, confidence, confidence_text, status, message
            FROM detection_history
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    return _row_to_record(row)


def get_history_image_path(record_id):
    record = get_history_record(record_id)
    if not record or not record.get("stored_image_name"):
        return None

    image_path = os.path.join(HISTORY_IMAGE_DIR, record["stored_image_name"])
    if not os.path.exists(image_path):
        return None

    return image_path


def delete_history_record(record_id):
    record = get_history_record(record_id)
    if record is None:
        return False

    image_path = get_history_image_path(record_id)
    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    with _connect() as connection:
        connection.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
        connection.commit()

    return True
