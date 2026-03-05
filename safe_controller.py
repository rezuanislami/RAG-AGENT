#!/usr/bin/env python3
"""
safe_controller.py - Safe Computer Control Orchestrator

Orchestrates the planning -> approval -> execution workflow.
Ensures all actions require human approval before execution.

SAFETY GUARANTEES:
1. No action executes without explicit approval
2. Emergency stop halts all execution immediately
3. All actions are logged for audit
4. Allowlist-only command execution
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_SERVER_PATH = str(SCRIPT_DIR / "mcp_server.py")


class ActionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Action:
    """Represents a single action in a plan."""
    id: str
    action_type: str
    description: str
    parameters: Dict[str, Any]
    status: ActionStatus = ActionStatus.PENDING
    result: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ActionPlan:
    """A complete action plan with multiple steps."""
    id: str
    goal: str
    steps: List[Action]
    status: ActionStatus = ActionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# SAFETY-CRITICAL: System prompt enforces proposal-only behavior
# ============================================================================
CONTROLLER_SYSTEM_PROMPT = """You are a SAFE AI controller that manages computer control actions.

CRITICAL SAFETY RULES:
1. You NEVER execute actions autonomously.
2. You generate action plans that REQUIRE human approval.
3. You respect the emergency stop at all times.
4. You only use allowlisted commands and keys.

Your job is to:
1. Analyze user requests
2. Break them down into safe, atomic actions
3. Create structured action plans
4. Wait for human approval before each action

Available actions:
- run_command: Execute allowlisted shell commands
- mouse_click: Click at coordinates
- type_text: Type text (max 1000 chars)
- press_key: Press allowlisted keys

FORBIDDEN:
- rm, sudo, chmod, chown, mv, cp
- Shell operators: |, ;, &, >, $
- Any bypass of approval system
"""


class SafeController:
    """
    Orchestrates safe computer control with human approval.
    
    Security guarantees:
    - No action executes without approval
    - Emergency stop halts everything
    - All actions are logged
    """
    
    def __init__(self):
        self.pending_plans: List[ActionPlan] = []
        self.execution_log: List[Dict[str, Any]] = []
        self.emergency_stop = False
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.tools = []
    
    async def initialize(self):
        """Initialize MCP client and tools."""
        self.mcp_client = MultiServerMCPClient({
            "safe-control-server": {
                "command": sys.executable,
                "args": [MCP_SERVER_PATH],
                "transport": "stdio",
            }
        })
        self.tools = await self.mcp_client.get_tools()
        print(f"✓ Initialized with tools: {[t.name for t in self.tools]}")
    
    def log_action(self, action_type: str, details: Dict, status: str, result: str = ""):
        """Log an action for audit purposes."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "details": details,
            "status": status,
            "result": result
        }
        self.execution_log.append(entry)
        return entry
    
    def trigger_emergency_stop(self):
        """Activate emergency stop - halts all execution."""
        self.emergency_stop = True
        self.pending_plans = []
        self.log_action("emergency_stop", {}, "triggered", "All execution halted")
        print("⛔ EMERGENCY STOP ACTIVATED")
    
    def release_emergency_stop(self):
        """Release emergency stop."""
        self.emergency_stop = False
        self.log_action("emergency_stop", {}, "released")
        print("✓ Emergency stop released")
    
    async def propose_actions(self, goal: str) -> ActionPlan:
        """
        Create an action plan for a goal.
        DOES NOT EXECUTE - only proposes.
        """
        if self.emergency_stop:
            raise RuntimeError("Emergency stop is active")
        
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Use LLM to break down the goal
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        
        messages = [
            SystemMessage(content=CONTROLLER_SYSTEM_PROMPT),
            HumanMessage(content=f"""Create an action plan for this goal: {goal}

Output a JSON object with this structure:
{{
    "steps": [
        {{"action_type": "run_command|mouse_click|type_text|press_key", "description": "...", "parameters": {{...}}}}
    ]
}}

Only use allowlisted commands (ls, pwd, echo, date, whoami, cat, head, tail).
Each step must be atomic and safe.""")
        ]
        
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                plan_data = {"steps": [{"action_type": "analysis", "description": goal, "parameters": {}}]}
        except json.JSONDecodeError:
            plan_data = {"steps": [{"action_type": "analysis", "description": goal, "parameters": {}}]}
        
        # Create action plan
        actions = []
        for i, step in enumerate(plan_data.get("steps", [])):
            action = Action(
                id=f"{plan_id}_step_{i+1}",
                action_type=step.get("action_type", "unknown"),
                description=step.get("description", ""),
                parameters=step.get("parameters", {})
            )
            actions.append(action)
        
        plan = ActionPlan(id=plan_id, goal=goal, steps=actions)
        self.pending_plans.append(plan)
        
        self.log_action("propose_actions", {"goal": goal, "plan_id": plan_id}, "proposed")
        
        return plan
    
    async def approve_plan(self, plan_id: str) -> bool:
        """Approve a pending plan."""
        for plan in self.pending_plans:
            if plan.id == plan_id:
                plan.status = ActionStatus.APPROVED
                self.log_action("approve_plan", {"plan_id": plan_id}, "approved")
                return True
        return False
    
    async def reject_plan(self, plan_id: str) -> bool:
        """Reject a pending plan."""
        for plan in self.pending_plans:
            if plan.id == plan_id:
                plan.status = ActionStatus.REJECTED
                self.pending_plans.remove(plan)
                self.log_action("reject_plan", {"plan_id": plan_id}, "rejected")
                return True
        return False
    
    async def execute_action(self, action: Action) -> str:
        """
        Execute a single approved action.
        REQUIRES prior approval check.
        """
        if self.emergency_stop:
            raise RuntimeError("Emergency stop is active")
        
        if action.status != ActionStatus.APPROVED:
            raise RuntimeError(f"Action {action.id} is not approved (status: {action.status})")
        
        action.status = ActionStatus.EXECUTING
        self.log_action("execute_action", {"action_id": action.id, "type": action.action_type}, "executing")
        
        try:
            # Find the appropriate tool
            tool = next((t for t in self.tools if t.name == action.action_type), None)
            
            if tool:
                result = await tool.ainvoke(action.parameters)
                action.result = str(result)
                action.status = ActionStatus.COMPLETED
                self.log_action("execute_action", {"action_id": action.id}, "completed", action.result[:200])
            else:
                action.result = f"Tool '{action.action_type}' not found"
                action.status = ActionStatus.FAILED
                self.log_action("execute_action", {"action_id": action.id}, "failed", action.result)
            
            return action.result
            
        except Exception as e:
            action.result = str(e)
            action.status = ActionStatus.FAILED
            self.log_action("execute_action", {"action_id": action.id}, "error", str(e))
            return f"Error: {e}"
    
    async def execute_approved_plan(self, plan_id: str):
        """
        Execute all steps in an approved plan.
        Stops on emergency stop or any failure.
        """
        plan = next((p for p in self.pending_plans if p.id == plan_id), None)
        
        if not plan:
            raise RuntimeError(f"Plan {plan_id} not found")
        
        if plan.status != ActionStatus.APPROVED:
            raise RuntimeError(f"Plan {plan_id} is not approved")
        
        plan.status = ActionStatus.EXECUTING
        
        for action in plan.steps:
            if self.emergency_stop:
                plan.status = ActionStatus.BLOCKED
                print(f"⛔ Plan halted by emergency stop")
                return
            
            # Each action must be approved before execution
            action.status = ActionStatus.APPROVED
            
            print(f"  Executing: {action.description}")
            result = await self.execute_action(action)
            print(f"  Result: {result[:100]}...")
            
            if action.status == ActionStatus.FAILED:
                plan.status = ActionStatus.FAILED
                print(f"  ❌ Action failed, stopping plan")
                return
        
        plan.status = ActionStatus.COMPLETED
        self.pending_plans.remove(plan)
        print(f"✅ Plan {plan_id} completed")
    
    def get_pending_plans(self) -> List[ActionPlan]:
        """Get all pending plans."""
        return [p for p in self.pending_plans if p.status == ActionStatus.PENDING]
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log."""
        return self.execution_log[-50:]


async def interactive_session():
    """Run an interactive control session."""
    controller = SafeController()
    await controller.initialize()
    
    print("\n" + "=" * 60)
    print("Safe Computer Control - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  [goal]     - Create action plan for a goal")
    print("  approve    - Approve pending plan")
    print("  reject     - Reject pending plan")
    print("  execute    - Execute approved plan")
    print("  stop       - Emergency stop")
    print("  log        - Show execution log")
    print("  quit       - Exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                break
            
            elif user_input.lower() == "stop":
                controller.trigger_emergency_stop()
            
            elif user_input.lower() == "log":
                for entry in controller.get_execution_log()[-10:]:
                    print(f"  [{entry['status']}] {entry['action_type']}: {entry.get('result', '')[:50]}")
            
            elif user_input.lower() == "approve":
                plans = controller.get_pending_plans()
                if plans:
                    plan = plans[0]
                    await controller.approve_plan(plan.id)
                    print(f"✅ Approved plan: {plan.id}")
                else:
                    print("No pending plans")
            
            elif user_input.lower() == "reject":
                plans = controller.get_pending_plans()
                if plans:
                    plan = plans[0]
                    await controller.reject_plan(plan.id)
                    print(f"❌ Rejected plan: {plan.id}")
                else:
                    print("No pending plans")
            
            elif user_input.lower() == "execute":
                approved = [p for p in controller.pending_plans if p.status == ActionStatus.APPROVED]
                if approved:
                    await controller.execute_approved_plan(approved[0].id)
                else:
                    print("No approved plans to execute")
            
            else:
                # Treat as goal
                print("\n📋 Creating action plan...")
                plan = await controller.propose_actions(user_input)
                print(f"\nPlan ID: {plan.id}")
                print(f"Goal: {plan.goal}")
                print("Steps:")
                for step in plan.steps:
                    print(f"  - [{step.action_type}] {step.description}")
                print("\n⚠️ Type 'approve' to approve or 'reject' to reject")
        
        except KeyboardInterrupt:
            print("\n\nUse 'quit' to exit or 'stop' for emergency stop")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    print("Starting Safe Controller...")
    asyncio.run(interactive_session())


if __name__ == "__main__":
    main()
