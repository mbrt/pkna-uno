"""Run headless Cursor agent sessions in a prepared workspace.

Each session is one request: the agent reads and writes files in the workspace
and runs shell commands without approval prompts, so the workspace should hold
only what the task needs.
"""

import json
import subprocess
from pathlib import Path


def run_cursor_agent(workspace: Path, model: str, prompt: str, timeout_s: int) -> dict:
    """Run one session and return its id, request id, duration, and token usage.

    The raw result is kept in agent-output.json in the workspace.
    """
    workspace = workspace.resolve()
    cmd = [
        "cursor-agent",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--trust",
        "--force",
        "--workspace",
        str(workspace),
        prompt,
    ]
    proc = subprocess.run(
        cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout_s
    )
    (workspace / "agent-output.json").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"cursor-agent exited with {proc.returncode}: {proc.stderr.strip()[-1000:]}"
        )
    result = json.loads(proc.stdout)
    if result.get("is_error"):
        raise RuntimeError(f"cursor-agent reported an error: {result.get('result')}")
    return {
        key: result.get(key)
        for key in ("session_id", "request_id", "duration_ms", "usage")
    }
