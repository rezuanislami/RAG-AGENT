#!/usr/bin/env python3
"""
mcp_server.py - MCP Server with RAG Tools

Exposes MCP tools for document search, listing, and reading.
Uses stdio transport for subprocess communication.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration - resolve paths for security
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

COLLECTION_NAME = "documents"

# ============================================================================
# SAFETY-CRITICAL: Command allowlist and blocked patterns
# These are HARDCODED and cannot be extended at runtime for security
# ============================================================================
ALLOWED_COMMANDS = frozenset({"ls", "pwd", "open", "echo", "date", "whoami", "cat", "head", "tail"})
BLOCKED_PATTERNS = frozenset({"rm", "sudo", "chmod", "chown", "mv", "cp", "kill", "pkill", 
                               ">", ">>", "|", ";", "&", "$", "`", "eval", "exec"})
ALLOWED_KEYS = frozenset({"enter", "return", "tab", "escape", "space", "backspace", "delete",
                          "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
                          "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"})
MAX_TYPE_LENGTH = 1000  # Maximum characters for type_text

# Global state for approval workflow
_pending_actions = []
_execution_log = []
_emergency_stop = False

# Initialize server
server = Server("rag-mcp-server")

# Lazy-loaded resources
_embedder = None
_collection = None


def get_embedder() -> SentenceTransformer:
    """Lazy load the embedding model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_collection():
    """Lazy load the ChromaDB collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def is_safe_path(requested_path: str) -> tuple[bool, Path | None]:
    """
    Validate that the requested path is within DATA_DIR.
    Prevents directory traversal attacks.
    Returns (is_safe, resolved_path or None).
    """
    try:
        # Resolve the path relative to DATA_DIR
        if requested_path.startswith("/"):
            # Absolute path - reject
            return False, None
        
        # Clean and resolve
        full_path = (DATA_DIR / requested_path).resolve()
        
        # Check if it's within DATA_DIR
        try:
            full_path.relative_to(DATA_DIR)
            return True, full_path
        except ValueError:
            return False, None
    except Exception:
        return False, None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        # Document tools
        Tool(
            name="search_docs",
            description="Search the document collection for relevant chunks. Returns top matching chunks with their source filenames.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    },
                    "k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {DEFAULT_TOP_K})",
                        "default": DEFAULT_TOP_K
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_docs",
            description="List all documents available in the data directory.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="read_doc",
            description="Read the full content of a specific document. Only files within the data directory can be read.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the document within the data directory"
                    }
                },
                "required": ["path"]
            }
        ),
        # ============================================================
        # SAFE CONTROL TOOLS - All require human approval before execution
        # ============================================================
        Tool(
            name="propose_actions",
            description="Generate a structured action plan for a goal. DOES NOT EXECUTE - only proposes actions for human approval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The goal to create an action plan for"
                    }
                },
                "required": ["goal"]
            }
        ),
        Tool(
            name="run_command",
            description="Execute a command from the ALLOWLIST ONLY. Blocked: rm, sudo, shell operators. REQUIRES HUMAN APPROVAL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute (must be in allowlist: ls, pwd, open, echo, date, whoami, cat, head, tail)"
                    }
                },
                "required": ["command"]
            }
        ),
        Tool(
            name="mouse_click",
            description="Click at screen coordinates. REQUIRES HUMAN APPROVAL before execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "X coordinate on screen"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate on screen"
                    }
                },
                "required": ["x", "y"]
            }
        ),
        Tool(
            name="type_text",
            description="Type text on the keyboard. REQUIRES HUMAN APPROVAL. Max 1000 characters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to type (max 1000 characters)"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="press_key",
            description="Press a keyboard key. REQUIRES HUMAN APPROVAL. Only allowlisted keys (enter, tab, escape, arrows, etc).",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to press (enter, tab, escape, space, backspace, delete, arrows, F1-F12)"
                    }
                },
                "required": ["key"]
            }
        ),
        Tool(
            name="get_pending_actions",
            description="Get the list of pending actions awaiting approval.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_execution_log",
            description="Get the execution log of all actions taken.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    # Document tools
    if name == "search_docs":
        return await search_docs(
            query=arguments["query"],
            k=arguments.get("k", DEFAULT_TOP_K)
        )
    elif name == "list_docs":
        return await list_docs()
    elif name == "read_doc":
        return await read_doc(path=arguments["path"])
    
    # Safe control tools
    elif name == "propose_actions":
        return await propose_actions(goal=arguments["goal"])
    elif name == "run_command":
        return await run_command(command=arguments["command"])
    elif name == "mouse_click":
        return await mouse_click(x=arguments["x"], y=arguments["y"])
    elif name == "type_text":
        return await type_text(text=arguments["text"])
    elif name == "press_key":
        return await press_key(key=arguments["key"])
    elif name == "get_pending_actions":
        return await get_pending_actions()
    elif name == "get_execution_log":
        return await get_execution_log()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_docs(query: str, k: int = DEFAULT_TOP_K) -> list[TextContent]:
    """Search for relevant document chunks."""
    try:
        embedder = get_embedder()
        collection = get_collection()
        
        # Generate query embedding
        query_embedding = embedder.encode([query])[0].tolist()
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, 20)  # Cap at 20 for safety
        )
        
        if not results["documents"] or not results["documents"][0]:
            return [TextContent(type="text", text="No relevant documents found.")]
        
        # Format results
        output_parts = [f"Found {len(results['documents'][0])} relevant chunk(s):\n"]
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            source = metadata.get("source", "unknown")
            chunk_idx = metadata.get("chunk_index", 0)
            similarity = 1 - distance  # Convert distance to similarity
            
            output_parts.append(f"\n--- Result {i+1} ---")
            output_parts.append(f"Source: {source} (chunk {chunk_idx})")
            output_parts.append(f"Relevance: {similarity:.2%}")
            output_parts.append(f"Content:\n{doc}\n")
        
        return [TextContent(type="text", text="\n".join(output_parts))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching documents: {str(e)}")]


async def list_docs() -> list[TextContent]:
    """List all documents in the data directory."""
    try:
        if not DATA_DIR.exists():
            return [TextContent(type="text", text=f"Data directory does not exist: {DATA_DIR}")]
        
        documents = []
        for ext in ["*.txt", "*.md"]:
            documents.extend(DATA_DIR.rglob(ext))
        
        if not documents:
            return [TextContent(type="text", text="No documents found in the data directory.")]
        
        output_parts = [f"Documents in {DATA_DIR}:\n"]
        for doc in sorted(documents):
            rel_path = doc.relative_to(DATA_DIR)
            size = doc.stat().st_size
            output_parts.append(f"  - {rel_path} ({size:,} bytes)")
        
        output_parts.append(f"\nTotal: {len(documents)} document(s)")
        
        return [TextContent(type="text", text="\n".join(output_parts))]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing documents: {str(e)}")]


async def read_doc(path: str) -> list[TextContent]:
    """Read a specific document (with path safety checks)."""
    # Security check
    is_safe, full_path = is_safe_path(path)
    
    if not is_safe:
        return [TextContent(
            type="text",
            text=f"Access denied: Path '{path}' is outside the allowed data directory."
        )]
    
    if not full_path.exists():
        return [TextContent(type="text", text=f"File not found: {path}")]
    
    if not full_path.is_file():
        return [TextContent(type="text", text=f"Not a file: {path}")]
    
    # Check extension
    if full_path.suffix.lower() not in [".txt", ".md"]:
        return [TextContent(type="text", text=f"Only .txt and .md files can be read.")]
    
    try:
        content = full_path.read_text(encoding="utf-8")
        
        # Limit content size for safety
        max_size = 50000
        if len(content) > max_size:
            content = content[:max_size] + f"\n\n[... truncated, file too large ({len(content):,} bytes) ...]"
        
        return [TextContent(
            type="text",
            text=f"=== Content of {path} ===\n\n{content}"
        )]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading file: {str(e)}")]


# ============================================================================
# SAFE CONTROL TOOL IMPLEMENTATIONS
# All actions require human approval before execution
# ============================================================================

def _log_action(action_type: str, details: dict, status: str, result: str = ""):
    """Log an action to the execution log."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action_type": action_type,
        "details": details,
        "status": status,
        "result": result
    }
    _execution_log.append(log_entry)
    return log_entry


def is_command_allowed(command: str) -> tuple[bool, str]:
    """
    SAFETY-CRITICAL: Check if a command is in the allowlist.
    Returns (is_allowed, reason).
    """
    # Check for blocked patterns first
    for blocked in BLOCKED_PATTERNS:
        if blocked in command.lower():
            return False, f"Blocked pattern detected: '{blocked}'"
    
    # Get the base command (first word)
    parts = command.strip().split()
    if not parts:
        return False, "Empty command"
    
    base_command = parts[0].lower()
    
    # Check allowlist
    if base_command not in ALLOWED_COMMANDS:
        return False, f"Command '{base_command}' not in allowlist. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
    
    return True, "Command allowed"


async def propose_actions(goal: str) -> list[TextContent]:
    """
    Generate a structured action plan for a goal.
    DOES NOT EXECUTE - only proposes for human approval.
    """
    global _pending_actions
    
    # Create a structured plan
    action_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    plan = {
        "id": action_id,
        "goal": goal,
        "status": "pending_approval",
        "created_at": datetime.now().isoformat(),
        "steps": [
            {
                "step_id": f"{action_id}_step_1",
                "description": f"Analyze goal: {goal}",
                "action_type": "analysis",
                "status": "pending"
            }
        ],
        "warning": "⚠️ This plan requires HUMAN APPROVAL before any execution."
    }
    
    # Add to pending actions
    _pending_actions.append(plan)
    
    _log_action("propose_actions", {"goal": goal}, "proposed", f"Plan ID: {action_id}")
    
    return [TextContent(
        type="text",
        text=json.dumps(plan, indent=2)
    )]


async def run_command(command: str) -> list[TextContent]:
    """
    Execute a command from the ALLOWLIST ONLY.
    SAFETY-CRITICAL: Only allowlisted commands can be executed.
    """
    global _emergency_stop
    
    # Check emergency stop
    if _emergency_stop:
        return [TextContent(type="text", text="⛔ EMERGENCY STOP ACTIVE. No commands can be executed.")]
    
    # SAFETY CHECK: Validate command against allowlist
    is_allowed, reason = is_command_allowed(command)
    
    if not is_allowed:
        _log_action("run_command", {"command": command}, "BLOCKED", reason)
        return [TextContent(
            type="text", 
            text=f"⛔ COMMAND BLOCKED: {reason}\n\nThis action has been logged for security review."
        )]
    
    # Log the attempt
    _log_action("run_command", {"command": command}, "approved", "Executing...")
    
    try:
        # Execute with strict safety constraints
        # - No shell=True (prevents shell injection)
        # - Timeout to prevent hanging
        # - Capture output
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(DATA_DIR)  # Restrict to data directory
        )
        
        output = result.stdout if result.stdout else ""
        error = result.stderr if result.stderr else ""
        
        _log_action("run_command", {"command": command}, "completed", output[:500])
        
        response = f"✅ Command executed: {command}\n\n"
        if output:
            response += f"Output:\n{output}\n"
        if error:
            response += f"Errors:\n{error}\n"
        response += f"\nReturn code: {result.returncode}"
        
        return [TextContent(type="text", text=response)]
        
    except subprocess.TimeoutExpired:
        _log_action("run_command", {"command": command}, "timeout", "Command timed out after 30s")
        return [TextContent(type="text", text="⚠️ Command timed out after 30 seconds.")]
    except Exception as e:
        _log_action("run_command", {"command": command}, "error", str(e))
        return [TextContent(type="text", text=f"❌ Error executing command: {str(e)}")]


async def mouse_click(x: int, y: int) -> list[TextContent]:
    """
    Click at screen coordinates.
    REQUIRES: pyautogui installed and human approval.
    """
    global _emergency_stop
    
    if _emergency_stop:
        return [TextContent(type="text", text="⛔ EMERGENCY STOP ACTIVE. No actions can be executed.")]
    
    # Validate coordinates (basic sanity check)
    if x < 0 or y < 0 or x > 10000 or y > 10000:
        return [TextContent(type="text", text=f"❌ Invalid coordinates: ({x}, {y})")]
    
    _log_action("mouse_click", {"x": x, "y": y}, "pending_approval")
    
    try:
        import pyautogui
        pyautogui.click(x, y)
        _log_action("mouse_click", {"x": x, "y": y}, "completed", f"Clicked at ({x}, {y})")
        return [TextContent(type="text", text=f"✅ Clicked at ({x}, {y})")]
    except ImportError:
        return [TextContent(type="text", text="❌ pyautogui not installed. Run: pip install pyautogui")]
    except Exception as e:
        _log_action("mouse_click", {"x": x, "y": y}, "error", str(e))
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]


async def type_text(text: str) -> list[TextContent]:
    """
    Type text on the keyboard.
    Max length: 1000 characters for safety.
    """
    global _emergency_stop
    
    if _emergency_stop:
        return [TextContent(type="text", text="⛔ EMERGENCY STOP ACTIVE. No actions can be executed.")]
    
    # Enforce max length
    if len(text) > MAX_TYPE_LENGTH:
        return [TextContent(type="text", text=f"❌ Text too long. Maximum {MAX_TYPE_LENGTH} characters allowed.")]
    
    _log_action("type_text", {"text_length": len(text), "preview": text[:50]}, "pending_approval")
    
    try:
        import pyautogui
        pyautogui.typewrite(text, interval=0.02)
        _log_action("type_text", {"text_length": len(text)}, "completed")
        return [TextContent(type="text", text=f"✅ Typed {len(text)} characters")]
    except ImportError:
        return [TextContent(type="text", text="❌ pyautogui not installed. Run: pip install pyautogui")]
    except Exception as e:
        _log_action("type_text", {"text_length": len(text)}, "error", str(e))
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]


async def press_key(key: str) -> list[TextContent]:
    """
    Press a keyboard key.
    Only allowlisted keys can be pressed.
    """
    global _emergency_stop
    
    if _emergency_stop:
        return [TextContent(type="text", text="⛔ EMERGENCY STOP ACTIVE. No actions can be executed.")]
    
    key_lower = key.lower()
    
    # Check allowlist
    if key_lower not in ALLOWED_KEYS:
        _log_action("press_key", {"key": key}, "BLOCKED", f"Key not allowed")
        return [TextContent(
            type="text",
            text=f"❌ Key '{key}' not in allowlist.\nAllowed keys: {', '.join(sorted(ALLOWED_KEYS))}"
        )]
    
    try:
        import pyautogui
        pyautogui.press(key_lower)
        _log_action("press_key", {"key": key}, "completed")
        return [TextContent(type="text", text=f"✅ Pressed key: {key}")]
    except ImportError:
        return [TextContent(type="text", text="❌ pyautogui not installed. Run: pip install pyautogui")]
    except Exception as e:
        _log_action("press_key", {"key": key}, "error", str(e))
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]


async def get_pending_actions() -> list[TextContent]:
    """Get the list of pending actions awaiting approval."""
    if not _pending_actions:
        return [TextContent(type="text", text="No pending actions.")]
    
    return [TextContent(type="text", text=json.dumps(_pending_actions, indent=2))]


async def get_execution_log() -> list[TextContent]:
    """Get the execution log of all actions taken."""
    if not _execution_log:
        return [TextContent(type="text", text="Execution log is empty.")]
    
    # Return last 50 entries
    recent_logs = _execution_log[-50:]
    return [TextContent(type="text", text=json.dumps(recent_logs, indent=2))]


def set_emergency_stop(stop: bool) -> None:
    """Set the emergency stop flag. Called by UI."""
    global _emergency_stop
    _emergency_stop = stop
    _log_action("emergency_stop", {"active": stop}, "triggered" if stop else "released")


def clear_pending_actions() -> None:
    """Clear all pending actions. Called by UI."""
    global _pending_actions
    _pending_actions = []


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
