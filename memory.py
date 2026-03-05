#!/usr/bin/env python3
"""
memory.py - Persistent Memory for RAG Agent

SQLite-based conversation memory with entity tracking.
Provides long-term learning across runs.
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# Memory database path
MEMORY_DB_PATH = Path(os.getenv("MEMORY_DB_PATH", "./memory.db")).resolve()


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_memory_db():
    """Initialize the memory database schema."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Entity memory table (tracks mentioned topics/facts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT NOT NULL,
                description TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                mention_count INTEGER DEFAULT 1
            )
        """)
        
        # Session metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                summary TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        
        conn.commit()
    
    print(f"✓ Memory database initialized at {MEMORY_DB_PATH}")


class ConversationMemory:
    """Manages conversation history and entity memory."""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize memory with optional session ID."""
        init_memory_db()
        self.session_id = session_id or self._generate_session_id()
        self._ensure_session()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _ensure_session(self):
        """Ensure session exists in database."""
        now = datetime.now().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO sessions (session_id, created_at, last_active)
                   VALUES (?, ?, ?)""",
                (self.session_id, now, now)
            )
            conn.commit()
    
    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Add a message to conversation history."""
        now = datetime.now().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO conversations (session_id, role, content, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.session_id, role, content, now, json.dumps(metadata) if metadata else None)
            )
            cursor.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (now, self.session_id)
            )
            conn.commit()
    
    def get_history(self, limit: int = 20) -> list[dict]:
        """Get recent conversation history."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT role, content, timestamp, metadata FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (self.session_id, limit)
            )
            rows = cursor.fetchall()
            return [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                }
                for row in reversed(rows)
            ]
    
    def add_entity(self, name: str, entity_type: str, description: Optional[str] = None):
        """Add or update an entity in memory."""
        now = datetime.now().isoformat()
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO entities (name, entity_type, description, first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       last_seen = excluded.last_seen,
                       mention_count = mention_count + 1,
                       description = COALESCE(excluded.description, description)""",
                (name.lower(), entity_type, description, now, now)
            )
            conn.commit()
    
    def get_entities(self, entity_type: Optional[str] = None) -> list[dict]:
        """Get all entities, optionally filtered by type."""
        with get_db() as conn:
            cursor = conn.cursor()
            if entity_type:
                cursor.execute(
                    "SELECT * FROM entities WHERE entity_type = ? ORDER BY mention_count DESC",
                    (entity_type,)
                )
            else:
                cursor.execute("SELECT * FROM entities ORDER BY mention_count DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def search_entities(self, query: str) -> list[dict]:
        """Search entities by name."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM entities WHERE name LIKE ? ORDER BY mention_count DESC",
                (f"%{query.lower()}%",)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_session_summary(self) -> Optional[str]:
        """Get session summary."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (self.session_id,)
            )
            row = cursor.fetchone()
            return row["summary"] if row else None
    
    def set_session_summary(self, summary: str):
        """Set session summary."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET summary = ? WHERE session_id = ?",
                (summary, self.session_id)
            )
            conn.commit()
    
    def list_sessions(self, limit: int = 10) -> list[dict]:
        """List recent sessions."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT session_id, created_at, last_active, summary
                   FROM sessions ORDER BY last_active DESC LIMIT ?""",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def clear_session(self):
        """Clear current session's conversation history."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (self.session_id,)
            )
            conn.commit()


def format_history_for_prompt(history: list[dict], max_chars: int = 2000) -> str:
    """Format conversation history for inclusion in prompt."""
    if not history:
        return ""
    
    lines = []
    char_count = 0
    
    for msg in reversed(history):
        line = f"{msg['role'].upper()}: {msg['content']}"
        if char_count + len(line) > max_chars:
            break
        lines.insert(0, line)
        char_count += len(line)
    
    if lines:
        return "Previous conversation:\n" + "\n".join(lines) + "\n\n"
    return ""


if __name__ == "__main__":
    # Demo usage
    print("Memory module demo")
    print("-" * 40)
    
    memory = ConversationMemory()
    print(f"Session ID: {memory.session_id}")
    
    # Add some messages
    memory.add_message("user", "What is machine learning?")
    memory.add_message("assistant", "Machine learning is a field of AI... [SOURCE: intro.md]")
    
    # Add entities
    memory.add_entity("machine learning", "topic", "A field of artificial intelligence")
    memory.add_entity("chromadb", "technology", "Vector database for embeddings")
    
    # Show history
    print("\nConversation history:")
    for msg in memory.get_history():
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Show entities
    print("\nEntities:")
    for entity in memory.get_entities():
        print(f"  {entity['name']} ({entity['entity_type']})")
    
    print("\n✓ Memory demo complete")
