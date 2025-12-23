"""
Chat history persistence layer.
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import sqlite3
from pathlib import Path

from .settings import CHAT_HISTORY_DB_PATH

# Using SQLite for simplicity.
DB_PATH = str(CHAT_HISTORY_DB_PATH)

def _ensure_db_path():
    """Ensure the database directory exists before creating/accessing DB."""
    db_file = Path(DB_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    """Initialize database schema. Creates DB and tables if they don't exist."""
    _ensure_db_path()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist (use html_answers as canonical column)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL UNIQUE,
        messages JSON NOT NULL,
        html_answers TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Ensure index exists
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_user_session 
    ON chat_sessions(session_id);
    """)

    # Migration: if an older column `html_answer` exists but `html_answers` doesn't,
    # add `html_answers` and migrate existing values into JSON list format.
    cursor.execute("PRAGMA table_info('chat_sessions')")
    cols = [row[1] for row in cursor.fetchall()]
    if 'html_answers' not in cols:
        # Add the new column
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN html_answers TEXT;")
        conn.commit()
        # If there is an old `html_answer` column, migrate its contents
        if 'html_answer' in cols:
            cursor.execute("SELECT session_id, html_answer FROM chat_sessions WHERE html_answer IS NOT NULL")
            rows = cursor.fetchall()
            for sid, old_html in rows:
                try:
                    # Wrap existing string into a JSON list
                    new_val = json.dumps([old_html])
                except Exception:
                    new_val = json.dumps([str(old_html)])
                cursor.execute(
                    "UPDATE chat_sessions SET html_answers = ? WHERE session_id = ?",
                    (new_val, sid),
                )
            conn.commit()
    
    conn.commit()
    conn.close()

def save_chat_history(session_id: str, messages: List[Dict[str, Any]], html_answer: Optional[str] = None) -> bool:
    """
    Save or update chat history for a session.
    
    - Creates DB file and schema if they don't exist
    - Creates/updates the session record with the given messages and optional html_answer
    
    Parameters:
        session_id: Unique identifier for the chat session
        messages: List of message dicts with {"role": "user"|"assistant", "content": "..."}
        html_answer: Optional HTML formatted answer to store separately
    
    Returns:
        True if save was successful
    """
    _ensure_db_path()
    init_db()

    # Normalize session_id: enforce a non-empty string so DB NOT NULL constraint isn't violated.
    if not session_id:
        session_id = "default_session"
    else:
        # strip whitespace
        session_id = str(session_id).strip() or "default_session"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Read existing html_answers (if any)
    cursor.execute("SELECT html_answers FROM chat_sessions WHERE session_id = ?", (session_id,))
    existing = cursor.fetchone()

    if existing and existing[0]:
        try:
            html_list = json.loads(existing[0])
            if not isinstance(html_list, list):
                html_list = [html_list]
        except Exception:
            html_list = [existing[0]]
    else:
        html_list = []

    if html_answer:
        html_list.append(html_answer)

    html_answers_json = json.dumps(html_list) if html_list else None

    # Upsert record
    if existing:
        cursor.execute(
            "UPDATE chat_sessions SET messages = ?, html_answers = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (json.dumps(messages), html_answers_json, session_id),
        )
    else:
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, messages, html_answers, created_at, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (session_id, json.dumps(messages), html_answers_json),
        )

    conn.commit()
    conn.close()
    return True

def get_user_session(session_id: str,) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch chat history for user session.
    
    Returns:
        - List of message dicts if history exists
        - Empty list [] if session_id doesn't exist or DB doesn't exist yet
    """
    # Normalize session_id to avoid None/empty values
    if not session_id:
        session_id = "default_session"
    else:
        session_id = str(session_id).strip() or "default_session"
    
    _ensure_db_path()
    
    # If DB file doesn't exist yet, return empty list (init will create it on save)
    if not Path(DB_PATH).exists():
        return []
    
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT messages FROM chat_sessions
    WHERE session_id = ?
    """, (session_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    # Return messages if found, otherwise empty list
    return json.loads(result[0]) if result else []

def get_user_session_with_html(session_id: str) -> tuple:
    """
    Fetch chat history AND html_answer for user session.

    Returns:
        - Tuple of (messages, html_answer_str) if history exists
        - Tuple of ([], None) if session_id doesn't exist or DB doesn't exist yet
    """
    _ensure_db_path()

    # If DB file doesn't exist yet, return empty values
    if not Path(DB_PATH).exists():
        return [], []

    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT messages, html_answers FROM chat_sessions WHERE session_id = ?", (session_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        messages = json.loads(result[0]) if result[0] else []
        if result[1]:
            try:
                html_answers = json.loads(result[1])
                if not isinstance(html_answers, list):
                    html_answers = [html_answers]
            except Exception:
                html_answers = [result[1]]
        else:
            html_answers = []
        return messages, html_answers

    return [], []