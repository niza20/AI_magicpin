"""SQLite store for /demo chats, so a reload (or a server restart) never loses a conversation.

One row per chat: how it was started (scenario / custom form), the transcript as shown in the UI, and the owner's
inputs. The reply engine is deterministic, so a chat whose in-memory state is gone is rebuilt by replaying its inputs.

Only the demo uses this. The judge-facing /v1 API stays in memory by design (testing brief §11: no persistence after
the test, wiped on /v1/teardown). Path: env DEMO_DB (default data/demo_chats.db). On Render, point DEMO_DB at a
persistent disk (e.g. /var/data/demo_chats.db) — the free plan's filesystem is reset on every restart/deploy.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

_PATH = os.environ.get("DEMO_DB") or str(Path(__file__).resolve().parent / "data" / "demo_chats.db")
_LOCK = threading.Lock()
_CONN: Optional[sqlite3.Connection] = None
MAX_CHATS_PER_OWNER = 100
MAX_ROWS = 20_000
_OWNER_RX = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")


def valid_owner(owner) -> Optional[str]:
    """Owner = a random id the browser keeps in localStorage; chats are only listed for that browser."""
    return owner if isinstance(owner, str) and _OWNER_RX.match(owner) else None


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        Path(_PATH).parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(_PATH, check_same_thread=False)
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.execute("""CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
            title TEXT, subtitle TEXT, start TEXT NOT NULL, log TEXT NOT NULL, ended INTEGER DEFAULT 0)""")
        _CONN.execute("CREATE INDEX IF NOT EXISTS chats_owner ON chats(owner, updated DESC)")
        _CONN.commit()
    return _CONN


def create(chat_id: str, owner: Optional[str], title: str, subtitle: str, start: dict, first: dict) -> None:
    if not owner:
        return
    now = time.time()
    with _LOCK:
        c = _conn()
        c.execute("INSERT OR REPLACE INTO chats(id, owner, created, updated, title, subtitle, start, log) VALUES (?,?,?,?,?,?,?,?)",
                  (chat_id, owner, now, now, title[:120], subtitle[:160], json.dumps(start, ensure_ascii=False),
                   json.dumps([{"role": "vera", "r": first, "ts": now}], ensure_ascii=False)))
        # keep the store bounded: oldest chats of this owner, then oldest overall
        c.execute("DELETE FROM chats WHERE owner=? AND id NOT IN (SELECT id FROM chats WHERE owner=? ORDER BY updated DESC LIMIT ?)",
                  (owner, owner, MAX_CHATS_PER_OWNER))
        c.execute("DELETE FROM chats WHERE id NOT IN (SELECT id FROM chats ORDER BY updated DESC LIMIT ?)", (MAX_ROWS,))
        c.commit()


def append(chat_id: str, entries: list[dict], ended: bool = False) -> None:
    with _LOCK:
        c = _conn()
        row = c.execute("SELECT log FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not row:
            return
        now = time.time()
        log = json.loads(row[0]) + [{**e, "ts": e.get("ts", now)} for e in entries]
        c.execute("UPDATE chats SET log=?, updated=?, ended=? WHERE id=?",
                  (json.dumps(log, ensure_ascii=False), now, int(ended), chat_id))
        c.commit()


def get(chat_id: str) -> Optional[dict]:
    with _LOCK:
        row = _conn().execute("SELECT id, owner, created, updated, title, subtitle, start, log, ended FROM chats WHERE id=?",
                              (chat_id,)).fetchone()
    if not row:
        return None
    return {"id": row[0], "owner": row[1], "created": row[2], "updated": row[3], "title": row[4], "subtitle": row[5],
            "start": json.loads(row[6]), "log": json.loads(row[7]), "ended": bool(row[8])}


def list_for(owner: str, limit: int = 50) -> list[dict]:
    with _LOCK:
        rows = _conn().execute("SELECT id, title, subtitle, updated, log, ended FROM chats WHERE owner=? ORDER BY updated DESC LIMIT ?",
                               (owner, limit)).fetchall()
    out = []
    for cid, title, sub, upd, log, ended in rows:
        entries = json.loads(log)
        last = next((e for e in reversed(entries) if e.get("role") in ("vera", "me") and (e.get("text") or (e.get("r") or {}).get("body"))), {})
        text = last.get("text") or (last.get("r") or {}).get("body") or ""
        out.append({"id": cid, "title": title, "subtitle": sub, "updated": upd, "ended": bool(ended),
                    "last": ("You: " if last.get("role") == "me" else "") + text.replace("\n", " ")[:90],
                    "turns": sum(1 for e in entries if e.get("role") == "me")})
    return out


def delete(chat_id: str, owner: str) -> bool:
    with _LOCK:
        c = _conn()
        n = c.execute("DELETE FROM chats WHERE id=? AND owner=?", (chat_id, owner)).rowcount
        c.commit()
    return n > 0


def reset_for_tests(path: str) -> None:
    global _PATH, _CONN
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
        _PATH, _CONN = path, None


def import_chat(chat_id: str, owner: str, title: str, subtitle: str, start: dict, log: list, ended: bool = False) -> bool:
    """Re-create a chat from the browser's own copy (after the server lost its disk). Never overwrites another owner's chat."""
    with _LOCK:
        c = _conn()
        row = c.execute("SELECT owner FROM chats WHERE id=?", (chat_id,)).fetchone()
        if row:
            return row[0] == owner
        ts = [e.get("ts") for e in log if isinstance(e, dict) and isinstance(e.get("ts"), (int, float))]
        created, updated = (min(ts), max(ts)) if ts else (time.time(), time.time())
        c.execute("INSERT INTO chats(id, owner, created, updated, title, subtitle, start, log, ended) VALUES (?,?,?,?,?,?,?,?,?)",
                  (chat_id, owner, created, updated, str(title)[:120], str(subtitle)[:160], json.dumps(start, ensure_ascii=False),
                   json.dumps(log, ensure_ascii=False), int(bool(ended))))
        c.commit()
    return True
