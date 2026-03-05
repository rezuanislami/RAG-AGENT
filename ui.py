#!/usr/bin/env python3
"""
ui.py - Native Desktop UI for Safe Computer Control

A Tkinter-based desktop interface for reviewing and approving AI-proposed actions.
All actions require explicit human approval before execution.

Run with: python ui.py

Framework choice: Tkinter
- Bundled with Python (no extra install)
- Simple, lightweight, works on macOS/Linux/Windows
- Sufficient for this approval-based workflow
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))


class ActionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ActionStep:
    """A single action step in a plan."""
    id: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    status: ActionStatus = ActionStatus.PENDING


@dataclass
class ActionPlan:
    """A complete action plan."""
    id: str
    goal: str
    steps: List[ActionStep]
    status: ActionStatus = ActionStatus.PENDING


# ============================================================================
# UI <-> Agent Communication via thread-safe queues
# ============================================================================
class AgentBridge:
    """
    Thread-safe bridge between UI and agent logic.
    
    Communication flow:
    1. UI puts user request in request_queue
    2. Agent thread reads request, generates plan
    3. Agent puts plan in response_queue
    4. UI reads and displays plan
    5. User approves/rejects in UI
    6. Approval sent via approval_queue
    7. Agent executes and sends results to log_queue
    """
    
    def __init__(self):
        self.request_queue = queue.Queue()      # UI -> Agent: user requests
        self.response_queue = queue.Queue()     # Agent -> UI: action plans
        self.approval_queue = queue.Queue()     # UI -> Agent: approvals
        self.log_queue = queue.Queue()          # Agent -> UI: execution logs
        self.emergency_stop = threading.Event() # Emergency stop flag
        self.agent_thread: Optional[threading.Thread] = None
    
    def submit_request(self, goal: str):
        """Submit a user request to the agent."""
        self.request_queue.put({"type": "request", "goal": goal})
    
    def approve_step(self, step_id: str):
        """Approve a specific step."""
        self.approval_queue.put({"type": "approve", "step_id": step_id})
    
    def reject_step(self, step_id: str):
        """Reject a specific step."""
        self.approval_queue.put({"type": "reject", "step_id": step_id})
    
    def approve_all(self, plan_id: str):
        """Approve all steps in a plan."""
        self.approval_queue.put({"type": "approve_all", "plan_id": plan_id})
    
    def reject_all(self, plan_id: str):
        """Reject entire plan."""
        self.approval_queue.put({"type": "reject_all", "plan_id": plan_id})
    
    def trigger_stop(self):
        """Trigger emergency stop."""
        self.emergency_stop.set()
        self.log_queue.put({"type": "log", "level": "STOP", 
                           "message": "⛔ EMERGENCY STOP ACTIVATED"})
    
    def release_stop(self):
        """Release emergency stop."""
        self.emergency_stop.clear()
        self.log_queue.put({"type": "log", "level": "INFO", 
                           "message": "✓ Emergency stop released"})
    
    def log(self, message: str, level: str = "INFO"):
        """Add a log entry."""
        self.log_queue.put({
            "type": "log",
            "level": level,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })


# ============================================================================
# Agent Worker Thread
# ============================================================================
class AgentWorker:
    """
    Background worker that processes requests and executes approved actions.
    Runs in a separate thread to keep UI responsive.
    """
    
    # SAFETY: Command allowlist (hardcoded)
    ALLOWED_COMMANDS = frozenset({"ls", "pwd", "open", "echo", "date", "whoami", "cat", "head", "tail"})
    BLOCKED_PATTERNS = frozenset({"rm", "sudo", "chmod", "chown", "mv", "cp", "kill", 
                                   ">", ">>", "|", ";", "&", "$", "`", "eval", "exec"})
    
    def __init__(self, bridge: AgentBridge):
        self.bridge = bridge
        self.current_plan: Optional[ActionPlan] = None
        self.running = False
    
    def is_command_allowed(self, command: str) -> tuple[bool, str]:
        """SAFETY-CRITICAL: Check command against allowlist."""
        for blocked in self.BLOCKED_PATTERNS:
            if blocked in command.lower():
                return False, f"Blocked pattern: '{blocked}'"
        
        parts = command.strip().split()
        if not parts:
            return False, "Empty command"
        
        base_cmd = parts[0].lower()
        if base_cmd not in self.ALLOWED_COMMANDS:
            return False, f"Not in allowlist: {base_cmd}"
        
        return True, "Allowed"
    
    def generate_plan(self, goal: str) -> ActionPlan:
        """Generate an action plan for a goal."""
        import subprocess
        
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simple rule-based planning (no LLM needed for basic demo)
        # In production, this would call the LLM
        steps = []
        
        goal_lower = goal.lower()
        
        if "list" in goal_lower or "show" in goal_lower or "files" in goal_lower:
            steps.append(ActionStep(
                id=f"{plan_id}_1",
                description="List current directory contents",
                action_type="run_command",
                parameters={"command": "ls -la"}
            ))
        elif "date" in goal_lower or "time" in goal_lower:
            steps.append(ActionStep(
                id=f"{plan_id}_1",
                description="Show current date and time",
                action_type="run_command",
                parameters={"command": "date"}
            ))
        elif "who" in goal_lower or "user" in goal_lower:
            steps.append(ActionStep(
                id=f"{plan_id}_1",
                description="Show current user",
                action_type="run_command",
                parameters={"command": "whoami"}
            ))
        elif "directory" in goal_lower or "pwd" in goal_lower or "where" in goal_lower:
            steps.append(ActionStep(
                id=f"{plan_id}_1",
                description="Show current working directory",
                action_type="run_command",
                parameters={"command": "pwd"}
            ))
        else:
            # Default: echo the goal
            steps.append(ActionStep(
                id=f"{plan_id}_1",
                description=f"Echo: {goal[:50]}",
                action_type="run_command",
                parameters={"command": f"echo 'Processing: {goal[:30]}'"}
            ))
        
        return ActionPlan(id=plan_id, goal=goal, steps=steps)
    
    def execute_step(self, step: ActionStep) -> str:
        """Execute a single approved action step."""
        import subprocess
        
        if self.bridge.emergency_stop.is_set():
            return "⛔ STOPPED: Emergency stop active"
        
        if step.action_type == "run_command":
            command = step.parameters.get("command", "")
            
            # SAFETY CHECK
            allowed, reason = self.is_command_allowed(command)
            if not allowed:
                return f"⛔ BLOCKED: {reason}"
            
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout or result.stderr or "(no output)"
                return f"✅ {output.strip()}"
            except subprocess.TimeoutExpired:
                return "⚠️ Command timed out"
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        return f"Unknown action type: {step.action_type}"
    
    def run(self):
        """Main worker loop."""
        self.running = True
        self.bridge.log("Agent worker started", "INFO")
        
        while self.running:
            try:
                # Check for user requests
                try:
                    request = self.bridge.request_queue.get(timeout=0.1)
                    if request["type"] == "request":
                        goal = request["goal"]
                        self.bridge.log(f"Received request: {goal}", "INFO")
                        
                        # Generate plan
                        self.current_plan = self.generate_plan(goal)
                        self.bridge.log(f"Generated plan with {len(self.current_plan.steps)} steps", "INFO")
                        
                        # Send plan to UI
                        self.bridge.response_queue.put({
                            "type": "plan",
                            "plan": self.current_plan
                        })
                except queue.Empty:
                    pass
                
                # Check for approvals
                try:
                    approval = self.bridge.approval_queue.get(timeout=0.1)
                    
                    if self.bridge.emergency_stop.is_set():
                        self.bridge.log("Action blocked: emergency stop active", "WARN")
                        continue
                    
                    if approval["type"] == "approve" and self.current_plan:
                        step_id = approval["step_id"]
                        for step in self.current_plan.steps:
                            if step.id == step_id and step.status == ActionStatus.PENDING:
                                step.status = ActionStatus.APPROVED
                                self.bridge.log(f"Step approved: {step.description}", "INFO")
                                
                                # Execute immediately after approval
                                step.status = ActionStatus.EXECUTING
                                self.bridge.log(f"Executing: {step.description}", "INFO")
                                
                                result = self.execute_step(step)
                                
                                if "BLOCKED" in result or "STOPPED" in result:
                                    step.status = ActionStatus.FAILED
                                else:
                                    step.status = ActionStatus.COMPLETED
                                
                                self.bridge.log(f"Result: {result}", "INFO")
                                
                                # Notify UI of status change
                                self.bridge.response_queue.put({
                                    "type": "step_update",
                                    "step_id": step_id,
                                    "status": step.status.value,
                                    "result": result
                                })
                                break
                    
                    elif approval["type"] == "reject" and self.current_plan:
                        step_id = approval["step_id"]
                        for step in self.current_plan.steps:
                            if step.id == step_id:
                                step.status = ActionStatus.REJECTED
                                self.bridge.log(f"Step rejected: {step.description}", "WARN")
                                self.bridge.response_queue.put({
                                    "type": "step_update",
                                    "step_id": step_id,
                                    "status": "rejected"
                                })
                                break
                    
                    elif approval["type"] == "approve_all" and self.current_plan:
                        for step in self.current_plan.steps:
                            if step.status == ActionStatus.PENDING:
                                if self.bridge.emergency_stop.is_set():
                                    break
                                step.status = ActionStatus.APPROVED
                                step.status = ActionStatus.EXECUTING
                                self.bridge.log(f"Executing: {step.description}", "INFO")
                                result = self.execute_step(step)
                                step.status = ActionStatus.COMPLETED if "✅" in result else ActionStatus.FAILED
                                self.bridge.log(f"Result: {result}", "INFO")
                                self.bridge.response_queue.put({
                                    "type": "step_update",
                                    "step_id": step.id,
                                    "status": step.status.value,
                                    "result": result
                                })
                    
                    elif approval["type"] == "reject_all" and self.current_plan:
                        for step in self.current_plan.steps:
                            step.status = ActionStatus.REJECTED
                        self.bridge.log("All steps rejected", "WARN")
                        self.current_plan = None
                
                except queue.Empty:
                    pass
                
            except Exception as e:
                self.bridge.log(f"Worker error: {str(e)}", "ERROR")
    
    def stop(self):
        """Stop the worker."""
        self.running = False


# ============================================================================
# Main Desktop UI
# ============================================================================
class SafeControlUI:
    """
    Native Tkinter desktop UI for safe computer control.
    
    Layout:
    +--------------------------------------------------+
    | [EMERGENCY STOP]                                 |
    +--------------------------------------------------+
    | Request: [________________] [Submit]             |
    +--------------------------------------------------+
    | Action Plan:                                     |
    | +-----------------------------------------+      |
    | | Step 1: Description           [✓] [✗]  |      |
    | | Step 2: Description           [✓] [✗]  |      |
    | +-----------------------------------------+      |
    | [Approve All] [Reject All]                       |
    +--------------------------------------------------+
    | Execution Log:                                   |
    | +-----------------------------------------+      |
    | | [INFO] 12:34:56 - Message...            |      |
    | +-----------------------------------------+      |
    +--------------------------------------------------+
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Safe Computer Control")
        self.root.geometry("800x700")
        self.root.minsize(600, 500)
        
        # Communication bridge
        self.bridge = AgentBridge()
        
        # Agent worker
        self.worker = AgentWorker(self.bridge)
        self.worker_thread: Optional[threading.Thread] = None
        
        # Current plan tracking
        self.current_plan: Optional[ActionPlan] = None
        self.step_widgets: Dict[str, Dict] = {}
        
        # Build UI
        self._build_ui()
        
        # Start agent worker
        self._start_worker()
        
        # Start polling for updates
        self._poll_updates()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_ui(self):
        """Build the UI components."""
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)  # Plan section
        self.root.rowconfigure(3, weight=1)  # Log section
        
        # ===== Emergency Stop Button =====
        stop_frame = ttk.Frame(self.root)
        stop_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        self.stop_btn = tk.Button(
            stop_frame,
            text="🛑 EMERGENCY STOP",
            command=self._toggle_stop,
            bg="#ff4444",
            fg="white",
            font=("Helvetica", 14, "bold"),
            height=2
        )
        self.stop_btn.pack(fill="x")
        
        self.stop_active = False
        
        # ===== Request Input =====
        input_frame = ttk.LabelFrame(self.root, text="Request", padding=10)
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        self.request_var = tk.StringVar()
        self.request_entry = ttk.Entry(input_frame, textvariable=self.request_var, font=("Helvetica", 12))
        self.request_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.request_entry.bind("<Return>", lambda e: self._submit_request())
        
        self.submit_btn = ttk.Button(input_frame, text="Submit", command=self._submit_request)
        self.submit_btn.grid(row=0, column=1)
        
        # ===== Action Plan Panel =====
        plan_frame = ttk.LabelFrame(self.root, text="Action Plan", padding=10)
        plan_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        plan_frame.columnconfigure(0, weight=1)
        plan_frame.rowconfigure(0, weight=1)
        
        # Scrollable container for steps
        self.plan_canvas = tk.Canvas(plan_frame, highlightthickness=0)
        plan_scrollbar = ttk.Scrollbar(plan_frame, orient="vertical", command=self.plan_canvas.yview)
        self.plan_inner = ttk.Frame(self.plan_canvas)
        
        self.plan_canvas.configure(yscrollcommand=plan_scrollbar.set)
        plan_scrollbar.grid(row=0, column=1, sticky="ns")
        self.plan_canvas.grid(row=0, column=0, sticky="nsew")
        self.plan_canvas.create_window((0, 0), window=self.plan_inner, anchor="nw")
        
        self.plan_inner.bind("<Configure>", lambda e: self.plan_canvas.configure(scrollregion=self.plan_canvas.bbox("all")))
        
        # Approve/Reject All buttons
        bulk_frame = ttk.Frame(plan_frame)
        bulk_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.approve_all_btn = ttk.Button(bulk_frame, text="✓ Approve All", command=self._approve_all)
        self.approve_all_btn.pack(side="left", padx=(0, 10))
        
        self.reject_all_btn = ttk.Button(bulk_frame, text="✗ Reject All", command=self._reject_all)
        self.reject_all_btn.pack(side="left")
        
        self.clear_btn = ttk.Button(bulk_frame, text="Clear", command=self._clear_plan)
        self.clear_btn.pack(side="right")
        
        # ===== Execution Log =====
        log_frame = ttk.LabelFrame(self.root, text="Execution Log", padding=10)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            state="disabled",
            font=("Courier", 10),
            wrap="word"
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        # Configure log colors
        self.log_text.tag_configure("INFO", foreground="black")
        self.log_text.tag_configure("WARN", foreground="orange")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("STOP", foreground="red", font=("Courier", 10, "bold"))
        
        # ===== Status Bar =====
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
    
    def _start_worker(self):
        """Start the agent worker thread."""
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        self._log("Agent worker started", "INFO")
    
    def _poll_updates(self):
        """Poll for updates from the agent worker."""
        # Check response queue
        try:
            while True:
                response = self.bridge.response_queue.get_nowait()
                
                if response["type"] == "plan":
                    self._display_plan(response["plan"])
                
                elif response["type"] == "step_update":
                    self._update_step_status(
                        response["step_id"],
                        response["status"],
                        response.get("result", "")
                    )
        except queue.Empty:
            pass
        
        # Check log queue
        try:
            while True:
                log = self.bridge.log_queue.get_nowait()
                self._log(log["message"], log["level"])
        except queue.Empty:
            pass
        
        # Schedule next poll
        self.root.after(100, self._poll_updates)
    
    def _submit_request(self):
        """Submit a request to the agent."""
        goal = self.request_var.get().strip()
        if not goal:
            return
        
        if self.stop_active:
            messagebox.showwarning("Emergency Stop", "Emergency stop is active. Release it first.")
            return
        
        self._log(f"Submitting: {goal}", "INFO")
        self.status_var.set("Processing request...")
        self.bridge.submit_request(goal)
        self.request_var.set("")
    
    def _display_plan(self, plan: ActionPlan):
        """Display an action plan in the UI."""
        self.current_plan = plan
        
        # Clear existing steps
        for widget in self.plan_inner.winfo_children():
            widget.destroy()
        self.step_widgets.clear()
        
        # Goal header
        goal_label = ttk.Label(
            self.plan_inner,
            text=f"Goal: {plan.goal}",
            font=("Helvetica", 11, "bold"),
            wraplength=500
        )
        goal_label.pack(anchor="w", pady=(0, 10))
        
        # Display each step
        for i, step in enumerate(plan.steps):
            self._create_step_widget(step, i + 1)
        
        self.status_var.set(f"Plan ready: {len(plan.steps)} steps")
    
    def _create_step_widget(self, step: ActionStep, num: int):
        """Create a widget for a single step."""
        frame = ttk.Frame(self.plan_inner)
        frame.pack(fill="x", pady=2)
        
        # Status indicator
        status_label = ttk.Label(frame, text="⏳", width=3)
        status_label.pack(side="left")
        
        # Step description
        desc_label = ttk.Label(
            frame,
            text=f"{num}. {step.description}",
            wraplength=400,
            anchor="w"
        )
        desc_label.pack(side="left", fill="x", expand=True)
        
        # Parameters info
        if step.parameters:
            params = json.dumps(step.parameters, indent=2)
            params_label = ttk.Label(frame, text=f"({step.action_type})", foreground="gray")
            params_label.pack(side="left", padx=5)
        
        # Approve button
        approve_btn = ttk.Button(
            frame,
            text="✓",
            width=3,
            command=lambda s=step: self._approve_step(s.id)
        )
        approve_btn.pack(side="right", padx=2)
        
        # Reject button
        reject_btn = ttk.Button(
            frame,
            text="✗",
            width=3,
            command=lambda s=step: self._reject_step(s.id)
        )
        reject_btn.pack(side="right", padx=2)
        
        # Store references
        self.step_widgets[step.id] = {
            "frame": frame,
            "status_label": status_label,
            "desc_label": desc_label,
            "approve_btn": approve_btn,
            "reject_btn": reject_btn
        }
    
    def _update_step_status(self, step_id: str, status: str, result: str = ""):
        """Update the visual status of a step."""
        if step_id not in self.step_widgets:
            return
        
        widgets = self.step_widgets[step_id]
        
        status_icons = {
            "pending": "⏳",
            "approved": "🔄",
            "executing": "🔄",
            "completed": "✅",
            "failed": "❌",
            "rejected": "🚫"
        }
        
        widgets["status_label"].configure(text=status_icons.get(status, "?"))
        
        # Disable buttons after action
        if status in ["completed", "failed", "rejected"]:
            widgets["approve_btn"].configure(state="disabled")
            widgets["reject_btn"].configure(state="disabled")
    
    def _approve_step(self, step_id: str):
        """Approve a specific step."""
        if self.stop_active:
            messagebox.showwarning("Emergency Stop", "Emergency stop is active.")
            return
        self.bridge.approve_step(step_id)
    
    def _reject_step(self, step_id: str):
        """Reject a specific step."""
        self.bridge.reject_step(step_id)
    
    def _approve_all(self):
        """Approve all pending steps."""
        if self.stop_active:
            messagebox.showwarning("Emergency Stop", "Emergency stop is active.")
            return
        if self.current_plan:
            self.bridge.approve_all(self.current_plan.id)
    
    def _reject_all(self):
        """Reject all steps."""
        if self.current_plan:
            self.bridge.reject_all(self.current_plan.id)
            self._clear_plan()
    
    def _clear_plan(self):
        """Clear the current plan display."""
        for widget in self.plan_inner.winfo_children():
            widget.destroy()
        self.step_widgets.clear()
        self.current_plan = None
        self.status_var.set("Ready")
    
    def _toggle_stop(self):
        """Toggle emergency stop."""
        if self.stop_active:
            self.stop_active = False
            self.stop_btn.configure(bg="#ff4444", text="🛑 EMERGENCY STOP")
            self.bridge.release_stop()
            self.status_var.set("Emergency stop released")
        else:
            self.stop_active = True
            self.stop_btn.configure(bg="#880000", text="🔓 RELEASE STOP")
            self.bridge.trigger_stop()
            self.status_var.set("⛔ EMERGENCY STOP ACTIVE")
    
    def _log(self, message: str, level: str = "INFO"):
        """Add a message to the log."""
        self.log_text.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] ", level)
        self.log_text.insert("end", f"{message}\n", level)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
    
    def _on_close(self):
        """Handle window close."""
        self.worker.stop()
        self.root.destroy()
    
    def run(self):
        """Start the UI main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    print("Starting Safe Control Desktop UI...")
    app = SafeControlUI()
    app.run()


if __name__ == "__main__":
    main()
