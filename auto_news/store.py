import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


class DocumentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self):
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_type TEXT NOT NULL,
                document_key TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_type, document_key)
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_type_updated
            ON documents(document_type, updated_at)
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                task_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                available_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(task_type, task_key)
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_status_available
            ON tasks(status, available_at)
            """
        )
        self.connection.commit()

    def upsert(self, document_type, document_key, content):
        self.connection.execute(
            """
            INSERT INTO documents (document_type, document_key, content_json)
            VALUES (?, ?, ?)
            ON CONFLICT(document_type, document_key) DO UPDATE SET
                content_json = excluded.content_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (document_type, document_key, json.dumps(content, ensure_ascii=False)),
        )
        self.connection.commit()

    def get(self, document_type, document_key):
        row = self.connection.execute(
            """
            SELECT content_json
            FROM documents
            WHERE document_type = ? AND document_key = ?
            """,
            (document_type, document_key),
        ).fetchone()

        if not row:
            return None

        return json.loads(row["content_json"])

    def list_by_type(self, document_type):
        rows = self.connection.execute(
            """
            SELECT document_key, content_json
            FROM documents
            WHERE document_type = ?
            ORDER BY updated_at DESC
            """,
            (document_type,),
        ).fetchall()

        return [
            {
                "document_key": row["document_key"],
                "content": json.loads(row["content_json"]),
            }
            for row in rows
        ]

    def delete_document(self, document_type, document_key):
        self.connection.execute(
            """
            DELETE FROM documents
            WHERE document_type = ? AND document_key = ?
            """,
            (document_type, document_key),
        )
        self.connection.commit()

    def enqueue_task(self, task_type, task_key, payload=None, max_attempts=3):
        payload = payload or {}
        self.connection.execute(
            """
            INSERT INTO tasks (task_type, task_key, payload_json, max_attempts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_type, task_key) DO UPDATE SET
                payload_json = excluded.payload_json,
                status = 'pending',
                attempts = 0,
                last_error = NULL,
                available_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                task_type,
                task_key,
                json.dumps(payload, ensure_ascii=False),
                max_attempts,
            ),
        )
        self.connection.commit()

    def claim_task(self):
        row = self.connection.execute(
            """
            SELECT *
            FROM tasks
            WHERE status = 'pending' AND available_at <= CURRENT_TIMESTAMP
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()

        if not row:
            return None

        self.connection.execute(
            """
            UPDATE tasks
            SET status = 'running',
                attempts = attempts + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (row["id"],),
        )
        self.connection.commit()

        return {
            "id": row["id"],
            "task_type": row["task_type"],
            "task_key": row["task_key"],
            "payload": json.loads(row["payload_json"]),
            "attempts": row["attempts"] + 1,
            "max_attempts": row["max_attempts"],
        }

    def list_tasks(self, task_type=None):
        if task_type:
            rows = self.connection.execute(
                """
                SELECT *
                FROM tasks
                WHERE task_type = ?
                ORDER BY created_at ASC
                """,
                (task_type,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT *
                FROM tasks
                ORDER BY created_at ASC
                """
            ).fetchall()

        return [
            {
                "id": row["id"],
                "task_type": row["task_type"],
                "task_key": row["task_key"],
                "payload": json.loads(row["payload_json"]),
                "status": row["status"],
                "attempts": row["attempts"],
                "max_attempts": row["max_attempts"],
                "available_at": row["available_at"],
                "last_error": row["last_error"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def delete_task(self, task_id):
        self.connection.execute(
            """
            DELETE FROM tasks
            WHERE id = ?
            """,
            (task_id,),
        )
        self.connection.commit()

    def complete_task(self, task_id):
        self.connection.execute(
            """
            UPDATE tasks
            SET status = 'done',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (task_id,),
        )
        self.connection.commit()

    def fail_task(self, task, error):
        next_status = (
            "failed" if task["attempts"] >= task["max_attempts"] else "pending"
        )
        delay_seconds = min(60 * task["attempts"], 300)
        self.connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                last_error = ?,
                available_at = datetime(CURRENT_TIMESTAMP, ?),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (next_status, str(error), f"+{delay_seconds} seconds", task["id"]),
        )
        self.connection.commit()

    def task_counts(self):
        rows = self.connection.execute(
            """
            SELECT status, count(*) AS count
            FROM tasks
            GROUP BY status
            ORDER BY status
            """
        ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def close(self):
        self.connection.close()


def stable_json_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sentence_set(text):
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return set()

    return {
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    }


def diff_article_text(previous_text, current_text):
    previous_sentences = sentence_set(previous_text)
    current_sentences = sentence_set(current_text)
    added = sorted(current_sentences - previous_sentences)
    removed = sorted(previous_sentences - current_sentences)

    return {
        "added": added,
        "removed": removed,
        "added_count": len(added),
        "removed_count": len(removed),
    }


def article_revision_key(url, content_hash):
    return f"{url}#{content_hash}"


def store_article_revision(store, article_document):
    url = article_document["url"]
    current_hash = text_hash(article_document.get("content", ""))
    previous_article = store.get("article", url)
    previous_hash = previous_article.get("content_hash") if previous_article else None
    previous_content = previous_article.get("content", "") if previous_article else ""

    article_document["content_hash"] = current_hash
    article_document["changed_since_last_seen"] = (
        previous_hash is not None and previous_hash != current_hash
    )
    article_document["previous_content_hash"] = previous_hash

    if previous_hash and previous_hash != current_hash:
        article_document["content_diff"] = diff_article_text(
            previous_content,
            article_document.get("content", ""),
        )

    revision_document = {
        **article_document,
        "seen_at": datetime.now().isoformat(timespec="seconds"),
    }
    store.upsert(
        "article_revision",
        article_revision_key(url, current_hash),
        revision_document,
    )
    store.upsert("article", url, article_document)
    return article_document


def topic_input_hash(articles_content):
    inputs = [
        {
            "provider": article.get("provider"),
            "url": article.get("url"),
            "content_hash": article.get("content_hash")
            or text_hash(article.get("content", "")),
        }
        for article in sorted(articles_content, key=lambda item: item.get("url", ""))
    ]
    return stable_json_hash(inputs)


def get_cached_topic_report(store, topic, articles_content):
    input_hash = topic_input_hash(articles_content)
    cache_key = f"{topic}#{input_hash}"
    cached_report = store.get("topic_report_cache", cache_key)

    if cached_report:
        print(f"Using cached topic report for {topic}")
        report = cached_report.get("report", {})
        report["topic"] = topic
        report["input_hash"] = input_hash
        return report

    return None


def store_topic_report_cache(store, topic, articles_content, report):
    input_hash = topic_input_hash(articles_content)
    cache_key = f"{topic}#{input_hash}"
    report["input_hash"] = input_hash
    store.upsert(
        "topic_report_cache",
        cache_key,
        {
            "topic": topic,
            "input_hash": input_hash,
            "article_inputs": [
                {
                    "provider": article.get("provider"),
                    "url": article.get("url"),
                    "content_hash": article.get("content_hash"),
                }
                for article in articles_content
            ],
            "report": report,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return report
