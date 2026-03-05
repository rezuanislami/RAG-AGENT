#!/usr/bin/env python3
"""
server.py - FastAPI Service for RAG Agent

REST API and WebSocket endpoints for the MCP RAG Agent.
Provides /query, /ingest, /health, and streaming support.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain and LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# MCP
from langchain_mcp_adapters.client import MultiServerMCPClient

# Local imports
from memory import ConversationMemory, format_history_for_prompt

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_PORT = int(os.getenv("API_PORT", "8000"))
SCRIPT_DIR = Path(__file__).parent.resolve()

# System prompt
SYSTEM_PROMPT = """
You are a retrieval-grounded AI agent.

You are given all relevant document context.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT call retrieval tools.
3. If context is insufficient, say so.
4. Cite sources using [SOURCE: filename].

Do not use prior knowledge.
"""


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    include_history: bool = True


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []
    context_chars: int = 0


class IngestRequest(BaseModel):
    file_path: Optional[str] = None
    text: Optional[str] = None
    filename: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    chromadb: bool
    mcp: bool


# Global state
mcp_client: Optional[MultiServerMCPClient] = None
llm: Optional[ChatOllama] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global mcp_client, llm
    
    print("🚀 Starting RAG Agent API...")
    
    # Initialize LLM
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )
    print(f"✓ Ollama connected ({OLLAMA_MODEL})")
    
    # Initialize MCP client
    mcp_server_path = str(SCRIPT_DIR / "mcp_server.py")
    mcp_client = MultiServerMCPClient({
        "rag-server": {
            "command": sys.executable,
            "args": [mcp_server_path],
            "transport": "stdio",
        }
    })
    print("✓ MCP client initialized")
    
    yield
    
    print("👋 Shutting down RAG Agent API...")


# Create FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="Local RAG Agent with MCP tools, memory, and grounded answers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def retrieve_context(question: str, tools) -> dict:
    """Mandatory retrieval step."""
    search_tool = next((t for t in tools if t.name == "search_docs"), None)
    if not search_tool:
        return {"context": "", "retrieved": False}
    
    try:
        results = await search_tool.ainvoke({"query": question})
        if not results:
            return {"context": "", "retrieved": False}
        
        if isinstance(results, str):
            return {"context": results, "retrieved": True}
        elif isinstance(results, list):
            combined = "\n\n".join(
                f"[SOURCE: {r.get('source', 'unknown')}]\n{r.get('content', str(r))}"
                for r in results
            )
            return {"context": combined, "retrieved": True}
        else:
            return {"context": str(results), "retrieved": True}
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        return {"context": "", "retrieved": False}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all services."""
    import urllib.request
    import urllib.error
    
    # Check Ollama
    ollama_ok = False
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5):
            ollama_ok = True
    except:
        pass
    
    # Check ChromaDB
    chroma_path = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()
    chromadb_ok = chroma_path.exists()
    
    # Check MCP
    mcp_ok = mcp_client is not None
    
    status = "healthy" if all([ollama_ok, chromadb_ok, mcp_ok]) else "degraded"
    
    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        chromadb=chromadb_ok,
        mcp=mcp_ok,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG agent."""
    if not mcp_client or not llm:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get tools
    all_tools = await mcp_client.get_tools()
    
    # Retrieval gate
    retrieval = await retrieve_context(request.question, all_tools)
    
    if not retrieval["retrieved"]:
        memory = ConversationMemory(request.session_id)
        memory.add_message("user", request.question)
        memory.add_message("assistant", "No relevant information was found in the documents.")
        return QueryResponse(
            answer="No relevant information was found in the documents.",
            session_id=memory.session_id,
            sources=[],
            context_chars=0,
        )
    
    # Initialize memory
    memory = ConversationMemory(request.session_id)
    
    # Build conversation history
    history_text = ""
    if request.include_history:
        history = memory.get_history(limit=10)
        history_text = format_history_for_prompt(history)
    
    # Prepare context message
    context_message = f"""
{history_text}Context:
{retrieval['context']}

Question:
{request.question}
"""
    
    # Get response from LLM (no tools needed - context already fetched)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context_message),
    ]
    
    response = await asyncio.to_thread(llm.invoke, messages)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # Save to memory
    memory.add_message("user", request.question)
    memory.add_message("assistant", answer, {"context_chars": len(retrieval['context'])})
    
    # Extract sources from answer
    import re
    sources = re.findall(r'\[SOURCE:\s*([^\]]+)\]', answer)
    
    return QueryResponse(
        answer=answer,
        session_id=memory.session_id,
        sources=list(set(sources)),
        context_chars=len(retrieval['context']),
    )


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """WebSocket endpoint for streaming queries."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            session_id = data.get("session_id")
            
            if not question:
                await websocket.send_json({"error": "No question provided"})
                continue
            
            # Get tools
            all_tools = await mcp_client.get_tools()
            
            # Retrieval gate
            await websocket.send_json({"status": "retrieving"})
            retrieval = await retrieve_context(question, all_tools)
            
            if not retrieval["retrieved"]:
                await websocket.send_json({
                    "status": "complete",
                    "answer": "No relevant information was found in the documents.",
                    "sources": [],
                })
                continue
            
            await websocket.send_json({
                "status": "reasoning",
                "context_chars": len(retrieval['context']),
            })
            
            # Prepare messages
            context_message = f"""
Context:
{retrieval['context']}

Question:
{question}
"""
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=context_message),
            ]
            
            # Get response
            response = await asyncio.to_thread(llm.invoke, messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Extract sources
            import re
            sources = re.findall(r'\[SOURCE:\s*([^\]]+)\]', answer)
            
            await websocket.send_json({
                "status": "complete",
                "answer": answer,
                "sources": list(set(sources)),
            })
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")


@app.get("/sessions")
async def list_sessions(limit: int = 10):
    """List recent sessions."""
    memory = ConversationMemory()
    return {"sessions": memory.list_sessions(limit)}


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50):
    """Get conversation history for a session."""
    memory = ConversationMemory(session_id)
    return {"history": memory.get_history(limit)}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's conversation history."""
    memory = ConversationMemory(session_id)
    memory.clear_session()
    return {"status": "cleared", "session_id": session_id}


def main():
    """Run the API server."""
    import uvicorn
    
    print("=" * 60)
    print("RAG Agent API Server")
    print("=" * 60)
    print(f"Port: {API_PORT}")
    print(f"Model: {OLLAMA_MODEL}")
    print("=" * 60)
    
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
