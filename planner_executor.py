#!/usr/bin/env python3
"""
planner_executor.py - Planner-Executor Multi-Agent Workflow

Implements a two-agent architecture:
- Planner: Analyzes the question and creates a step-by-step plan
- Executor: Executes each step using available tools

Uses LangGraph for orchestration with checkpointing.
"""

import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict, Literal
from dataclasses import dataclass

from dotenv import load_dotenv

# LangChain and LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# MCP
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SCRIPT_DIR = Path(__file__).parent.resolve()


# Prompts
PLANNER_PROMPT = """You are a planning agent. Your job is to analyze questions and create step-by-step plans.

Given a question, output a clear plan with numbered steps.
Each step should be a single, actionable task.

Format your plan as:
PLAN:
1. [First step]
2. [Second step]
...

Rules:
- Keep plans simple (2-4 steps max)
- Focus on information retrieval and synthesis
- If the question is simple, a 1-step plan is fine

Example:
Question: "What are the main topics in the documentation?"
PLAN:
1. Search documents for overview of topics
2. Synthesize findings into a summary with citations
"""

EXECUTOR_PROMPT = """You are an executor agent. You execute plans step by step.

You are given:
- The original question
- A plan to follow
- Retrieved context from documents
- The current step to execute

Rules:
1. Focus only on the current step
2. Use ONLY the provided context
3. Cite sources using [SOURCE: filename]
4. If context is insufficient, say so
5. Do not use prior knowledge

After completing all steps, provide a final synthesized answer.
"""


@dataclass
class Plan:
    """A plan with steps."""
    steps: list[str]
    current_step: int = 0
    
    def current(self) -> str:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return ""
    
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)
    
    def advance(self):
        self.current_step += 1


class WorkflowState(TypedDict):
    """State for the planner-executor workflow."""
    question: str
    plan: dict  # Serialized Plan
    context: str
    step_results: list[str]
    final_answer: str
    messages: Annotated[list, add_messages]


def parse_plan(response: str) -> list[str]:
    """Parse a plan from the planner's response."""
    lines = response.split('\n')
    steps = []
    
    for line in lines:
        line = line.strip()
        # Match numbered steps like "1.", "2.", etc.
        if line and len(line) > 2 and line[0].isdigit() and line[1] == '.':
            step = line[2:].strip()
            if step:
                steps.append(step)
    
    # Fallback: if no numbered steps found, treat the whole thing as one step
    if not steps:
        steps = ["Answer the question based on retrieved context"]
    
    return steps


async def run_planner_executor(question: str):
    """Run the planner-executor workflow."""
    
    print(f"Connecting to Ollama ({OLLAMA_MODEL})...")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )
    
    # MCP setup
    mcp_server_path = str(SCRIPT_DIR / "mcp_server.py")
    print("Connecting to MCP server...")
    mcp_client = MultiServerMCPClient({
        "rag-server": {
            "command": sys.executable,
            "args": [mcp_server_path],
            "transport": "stdio",
        }
    })
    
    all_tools = await mcp_client.get_tools()
    print(f"Available tools: {[t.name for t in all_tools]}")
    
    # Get search tool for retrieval
    search_tool = next((t for t in all_tools if t.name == "search_docs"), None)
    
    # Step 1: Retrieval (mandatory gate)
    print("\n🔍 Step 1: Retrieving context...")
    context = ""
    if search_tool:
        try:
            results = await search_tool.ainvoke({"query": question})
            if isinstance(results, str):
                context = results
            elif isinstance(results, list):
                context = "\n\n".join(
                    f"[SOURCE: {r.get('source', 'unknown')}]\n{r.get('content', str(r))}"
                    for r in results
                )
            else:
                context = str(results)
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
    
    if not context:
        print("\n📝 No relevant information found in documents.")
        return
    
    print(f"✓ Retrieved context ({len(context)} chars)")
    
    # Step 2: Planning
    print("\n📋 Step 2: Creating plan...")
    plan_messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Question: {question}"),
    ]
    plan_response = llm.invoke(plan_messages)
    plan_text = plan_response.content if hasattr(plan_response, 'content') else str(plan_response)
    
    steps = parse_plan(plan_text)
    print(f"✓ Plan created with {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    
    # Step 3: Execution
    print("\n⚡ Step 3: Executing plan...")
    step_results = []
    
    for i, step in enumerate(steps, 1):
        print(f"\n   Executing step {i}/{len(steps)}: {step[:50]}...")
        
        exec_messages = [
            SystemMessage(content=EXECUTOR_PROMPT),
            HumanMessage(content=f"""
Original Question: {question}

Current Step ({i}/{len(steps)}): {step}

Context:
{context}

Previous Step Results:
{chr(10).join(step_results) if step_results else "None yet"}

Execute this step and provide your result.
"""),
        ]
        
        exec_response = llm.invoke(exec_messages)
        result = exec_response.content if hasattr(exec_response, 'content') else str(exec_response)
        step_results.append(f"Step {i} Result: {result}")
        print(f"   ✓ Step {i} complete")
    
    # Step 4: Final synthesis
    print("\n📝 Step 4: Synthesizing final answer...")
    
    synth_messages = [
        SystemMessage(content="""You are a synthesis agent. 
Combine the step results into a coherent final answer.
Keep citations [SOURCE: filename] intact.
Be concise but complete."""),
        HumanMessage(content=f"""
Question: {question}

Step Results:
{chr(10).join(step_results)}

Provide a final, synthesized answer.
"""),
    ]
    
    final_response = llm.invoke(synth_messages)
    final_answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
    
    print("-" * 60)
    print("\n📝 Final Answer:\n")
    print(final_answer)
    
    return final_answer


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python planner_executor.py \"Your question here\"")
        print("\nExample:")
        print("  python planner_executor.py \"Summarize the main topics in my documents\"")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    print("=" * 60)
    print("Planner-Executor RAG Agent")
    print("=" * 60)
    print()
    print(f"Question: {question}")
    
    import asyncio
    asyncio.run(run_planner_executor(question))
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
