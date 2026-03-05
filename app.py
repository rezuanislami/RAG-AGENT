#!/usr/bin/env python3
"""
app.py - LangGraph Agent with MCP Tools

A ReAct-style agent that uses Ollama for LLM and MCP tools for document retrieval.
Provides grounded answers with citations.
"""

import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

# LangChain and LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# MCP
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# System prompt for the agent (context is pre-fetched before agent runs)
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


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list, add_messages]


async def retrieve_context(question: str, tools):
    """
    Mandatory retrieval step.
    Always calls search_docs before agent reasoning.
    """
    search_tool = next((t for t in tools if t.name == "search_docs"), None)
    if not search_tool:
        return {"context": "", "retrieved": False, "question": question}
    
    try:
        results = await search_tool.ainvoke({"query": question})
        
        if not results:
            return {"context": "", "retrieved": False, "question": question}
        
        # Handle different result formats
        if isinstance(results, str):
            # If results is a string, use it directly
            return {"context": results, "retrieved": True, "question": question}
        elif isinstance(results, list):
            # If results is a list of dicts with source/content
            combined = "\n\n".join(
                f"[SOURCE: {r.get('source', 'unknown')}]\n{r.get('content', str(r))}"
                for r in results
            )
            return {"context": combined, "retrieved": True, "question": question}
        else:
            return {"context": str(results), "retrieved": True, "question": question}
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        return {"context": "", "retrieved": False, "question": question}


async def run_agent(question: str):
    """Run the agent with a question."""
    
    # Initialize Ollama LLM
    print(f"Connecting to Ollama ({OLLAMA_MODEL})...")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,  # Low temperature for more consistent answers
    )
    
    # MCP server configuration
    mcp_server_path = str(SCRIPT_DIR / "mcp_server.py")
    
    # Create MCP client (new API - not a context manager)
    print("Connecting to MCP server...")
    mcp_client = MultiServerMCPClient({
        "rag-server": {
            "command": sys.executable,
            "args": [mcp_server_path],
            "transport": "stdio",
        }
    })
    
    # Get tools from MCP server (await since it's async)
    all_tools = await mcp_client.get_tools()
    print(f"Available tools: {[t.name for t in all_tools]}")
    
    # Step 1: Mandatory retrieval gate - call search_docs FIRST
    print("\n🔍 Running retrieval gate...")
    retrieval = await retrieve_context(question, all_tools)
    
    if not retrieval["retrieved"]:
        print("-" * 60)
        print("\n📝 Answer:\n")
        print("No relevant information was found in the documents.")
        return
    
    print(f"✓ Retrieved context ({len(retrieval['context'])} chars)")
    
    # Since we already have context, we don't need tools or a graph
    # Just use the LLM directly for grounded reasoning
    print("\nProcessing your question...\n")
    print("-" * 60)
    
    # Prepare context message
    context_message = f"""
Context:
{retrieval['context']}

Question:
{question}
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context_message)
    ]
    
    # Get response directly from LLM (no tools needed)
    response = llm.invoke(messages)
    final_response = response.content if hasattr(response, 'content') else str(response)
    
    print("-" * 60)
    if final_response:
        print("\n📝 Answer:\n")
        print(final_response)
    else:
        print("\nNo response generated.")


def check_prerequisites():
    """Check that required services are available."""
    import urllib.request
    import urllib.error
    
    # Check Ollama
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            pass
        print("✓ Ollama is running")
    except (urllib.error.URLError, Exception) as e:
        print(f"✗ Ollama is not running at {OLLAMA_BASE_URL}")
        print("  Please start Ollama with: ollama serve")
        print("  Then pull the model: ollama pull " + OLLAMA_MODEL)
        sys.exit(1)
    
    # Check Chroma DB
    chroma_path = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()
    if not chroma_path.exists():
        print(f"✗ ChromaDB not found at {chroma_path}")
        print("  Please run: python ingest.py")
        sys.exit(1)
    print(f"✓ ChromaDB found at {chroma_path}")
    
    # Check data directory
    data_path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    if not data_path.exists():
        print(f"✗ Data directory not found at {data_path}")
        print("  Please create it and add some documents")
        sys.exit(1)
    print(f"✓ Data directory found at {data_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python app.py \"Your question here\"")
        print("\nExample:")
        print("  python app.py \"What topics are covered in my documents?\"")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    print("=" * 60)
    print("RAG Agent with MCP Tools")
    print("=" * 60)
    print()
    
    # Check prerequisites
    check_prerequisites()
    print()
    
    print(f"Question: {question}")
    
    # Run the agent
    import asyncio
    asyncio.run(run_agent(question))
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
