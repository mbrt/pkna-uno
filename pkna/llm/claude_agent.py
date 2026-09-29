"""Run headless Claude Code sessions in a prepared workspace.

Each session is one request: the agent reads and writes files in the workspace
and runs shell commands without approval prompts, so the workspace should hold
only what the task needs.
"""

import json
import subprocess
from pathlib import Path


def run_claude_agent(workspace: Path, model: str, prompt: str, timeout_s: int) -> dict:
    """Run one session and return its id, request id, duration, and token usage.

    The raw result is kept in agent-output.json in the workspace.
    """
    workspace = workspace.resolve()
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--effort",
        "high",
        "--permission-mode",
        "bypassPermissions",
        prompt,
    ]
    proc = subprocess.run(
        cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout_s
    )
    (workspace / "agent-output.json").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited with {proc.returncode}: {proc.stderr.strip()[-1000:]}"
        )
    result = json.loads(proc.stdout)
    if result.get("is_error"):
        raise RuntimeError(f"claude reported an error: {result.get('result')}")
    return {
        "session_id": result.get("session_id"),
        "request_id": result.get("uuid"),
        "duration_ms": result.get("duration_ms"),
        "usage": result.get("usage"),
        "cost_usd": result.get("total_cost_usd"),
    }
