#!/usr/bin/env python3
"""
security.py - Security Hardening for RAG Agent

Provides audit logging, rate limiting, and tool execution limits.
All security features are local-first with no external dependencies.
"""

import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from functools import wraps
from contextlib import contextmanager

# Audit log database path
AUDIT_DB_PATH = Path(os.getenv("AUDIT_DB_PATH", "./audit.db")).resolve()

# Security limits
MAX_TOOL_CALLS_PER_QUERY = int(os.getenv("MAX_TOOL_CALLS_PER_QUERY", "10"))
MAX_QUERIES_PER_MINUTE = int(os.getenv("MAX_QUERIES_PER_MINUTE", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "50000"))


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(AUDIT_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_audit_db():
    """Initialize the audit database schema."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                success INTEGER NOT NULL DEFAULT 1
            )
        """)
        
        # Rate limiting table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                key TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0,
                window_start TEXT NOT NULL
            )
        """)
        
        # Tool call tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments TEXT,
                timestamp TEXT NOT NULL,
                duration_ms INTEGER,
                success INTEGER NOT NULL DEFAULT 1,
                error TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_query ON tool_calls(query_id)")
        
        conn.commit()
    
    print(f"✓ Audit database initialized at {AUDIT_DB_PATH}")


class AuditLogger:
    """Logs all security-relevant events."""
    
    def __init__(self):
        init_audit_db()
    
    def log(
        self,
        event_type: str,
        action: str,
        details: Optional[dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
    ):
        """Log an audit event."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO audit_log 
                   (timestamp, event_type, user_id, session_id, action, details, ip_address, success)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    event_type,
                    user_id,
                    session_id,
                    action,
                    json.dumps(details) if details else None,
                    ip_address,
                    1 if success else 0,
                )
            )
            conn.commit()
    
    def log_query(self, question: str, session_id: Optional[str] = None, **kwargs):
        """Log a query event."""
        self.log("QUERY", "user_query", {"question": question[:500]}, session_id=session_id, **kwargs)
    
    def log_tool_call(self, tool_name: str, arguments: dict, query_id: str, **kwargs):
        """Log a tool call."""
        self.log("TOOL_CALL", f"call_{tool_name}", {"arguments": arguments}, **kwargs)
        
        # Also track in tool_calls table
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO tool_calls (query_id, tool_name, arguments, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (query_id, tool_name, json.dumps(arguments), datetime.now().isoformat())
            )
            conn.commit()
    
    def log_security_event(self, action: str, details: dict, success: bool = True, **kwargs):
        """Log a security-related event."""
        self.log("SECURITY", action, details, success=success, **kwargs)
    
    def get_recent_logs(self, limit: int = 100, event_type: Optional[str] = None) -> list[dict]:
        """Get recent audit logs."""
        with get_db() as conn:
            cursor = conn.cursor()
            if event_type:
                cursor.execute(
                    """SELECT * FROM audit_log 
                       WHERE event_type = ? 
                       ORDER BY timestamp DESC LIMIT ?""",
                    (event_type, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]


class RateLimiter:
    """Rate limiting for queries and tool calls."""
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        init_audit_db()
        self.audit = audit_logger or AuditLogger()
    
    def check_rate_limit(self, key: str, max_count: int, window_seconds: int = 60) -> bool:
        """Check if rate limit is exceeded. Returns True if allowed, False if blocked."""
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get current rate limit state
            cursor.execute("SELECT count, window_start FROM rate_limits WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                stored_window = datetime.fromisoformat(row["window_start"])
                if stored_window < window_start:
                    # Window expired, reset
                    cursor.execute(
                        "UPDATE rate_limits SET count = 1, window_start = ? WHERE key = ?",
                        (now.isoformat(), key)
                    )
                elif row["count"] >= max_count:
                    self.audit.log_security_event(
                        "rate_limit_exceeded",
                        {"key": key, "count": row["count"], "max": max_count},
                        success=False,
                    )
                    return False
                else:
                    cursor.execute(
                        "UPDATE rate_limits SET count = count + 1 WHERE key = ?",
                        (key,)
                    )
            else:
                cursor.execute(
                    "INSERT INTO rate_limits (key, count, window_start) VALUES (?, 1, ?)",
                    (key, now.isoformat())
                )
            
            conn.commit()
            return True
    
    def check_query_rate(self, user_id: str = "default") -> bool:
        """Check query rate limit."""
        return self.check_rate_limit(f"query:{user_id}", MAX_QUERIES_PER_MINUTE, 60)


class ToolCallLimiter:
    """Limits tool calls per query."""
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        init_audit_db()
        self.audit = audit_logger or AuditLogger()
        self._query_tool_counts: dict[str, int] = {}
    
    def check_tool_limit(self, query_id: str) -> bool:
        """Check if tool call limit is exceeded for a query."""
        current = self._query_tool_counts.get(query_id, 0)
        
        if current >= MAX_TOOL_CALLS_PER_QUERY:
            self.audit.log_security_event(
                "tool_limit_exceeded",
                {"query_id": query_id, "count": current, "max": MAX_TOOL_CALLS_PER_QUERY},
                success=False,
            )
            return False
        
        self._query_tool_counts[query_id] = current + 1
        return True
    
    def reset_query(self, query_id: str):
        """Reset tool count for a query."""
        self._query_tool_counts.pop(query_id, None)
    
    def get_tool_calls_for_query(self, query_id: str) -> list[dict]:
        """Get all tool calls for a query."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM tool_calls WHERE query_id = ? ORDER BY timestamp",
                (query_id,)
            )
            return [dict(row) for row in cursor.fetchall()]


class InputSanitizer:
    """Sanitizes user input to prevent injection attacks."""
    
    # Characters that could be dangerous in various contexts
    DANGEROUS_PATTERNS = [
        "../",  # Path traversal
        "..\\",
        "/etc/",
        "/proc/",
        "~/.ssh/",
        "${",  # Shell variable injection
        "$((",
        "`",   # Backtick command execution
        "<script>",  # XSS
        "javascript:",
        "data:",
    ]
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit = audit_logger or AuditLogger()
    
    def sanitize_query(self, query: str) -> tuple[str, bool]:
        """
        Sanitize a user query.
        Returns (sanitized_query, was_modified).
        """
        original = query
        modified = False
        
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in query.lower():
                query = query.replace(pattern, "")
                modified = True
        
        if modified:
            self.audit.log_security_event(
                "input_sanitized",
                {"original_length": len(original), "sanitized_length": len(query)},
            )
        
        return query.strip(), modified
    
    def validate_file_path(self, path: str, allowed_dirs: list[str]) -> bool:
        """Validate a file path is within allowed directories."""
        try:
            resolved = Path(path).resolve()
            
            for allowed in allowed_dirs:
                allowed_path = Path(allowed).resolve()
                if str(resolved).startswith(str(allowed_path)):
                    return True
            
            self.audit.log_security_event(
                "path_validation_failed",
                {"path": str(path), "allowed_dirs": allowed_dirs},
                success=False,
            )
            return False
            
        except Exception:
            return False
    
    def truncate_context(self, context: str) -> str:
        """Truncate context to prevent memory issues."""
        if len(context) > MAX_CONTEXT_CHARS:
            self.audit.log_security_event(
                "context_truncated",
                {"original_length": len(context), "max_length": MAX_CONTEXT_CHARS},
            )
            return context[:MAX_CONTEXT_CHARS] + "\n\n[TRUNCATED]"
        return context


# Convenience instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get shared audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_tool_call(func: Callable) -> Callable:
    """Decorator to audit tool calls."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_audit_logger()
        start = time.time()
        success = True
        error = None
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration = int((time.time() - start) * 1000)
            logger.log(
                "TOOL_CALL",
                func.__name__,
                {
                    "duration_ms": duration,
                    "success": success,
                    "error": error,
                },
                success=success,
            )
    
    return wrapper


if __name__ == "__main__":
    print("Security module demo")
    print("-" * 40)
    
    # Initialize
    audit = AuditLogger()
    rate_limiter = RateLimiter(audit)
    tool_limiter = ToolCallLimiter(audit)
    sanitizer = InputSanitizer(audit)
    
    # Test sanitization
    test_input = "What is in ../../../etc/passwd?"
    sanitized, modified = sanitizer.sanitize_query(test_input)
    print(f"Input: {test_input}")
    print(f"Sanitized: {sanitized} (modified: {modified})")
    
    # Test rate limiting
    for i in range(5):
        allowed = rate_limiter.check_query_rate("test_user")
        print(f"Query {i+1}: {'allowed' if allowed else 'blocked'}")
    
    # Show logs
    print("\nRecent audit logs:")
    for log in audit.get_recent_logs(5):
        print(f"  {log['timestamp']}: {log['action']}")
    
    print("\n✓ Security demo complete")
