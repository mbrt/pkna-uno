"""Shared data models for the eval pipeline.

Defines the contract between the three eval stages:
1. generate_eval_prompts -> EvalPrompt (JSONL)
2. run_eval_inference -> EvalTrace (JSONL)
3. score_eval_traces -> ScoredTrace, SuiteResult, EvalReport (JSON)
"""

from typing import Any

from pydantic import BaseModel


SUITES = frozenset(
    [
        "personality",
        "social_reasoning",
        "tool_use",
        "memory_handling",
        "stability",
        "language",
    ]
)


class EvalPrompt(BaseModel):
    """A single eval prompt ready for inference.

    Written by stage 1 (generate_eval_prompts), read by stage 2
    (run_eval_inference).
    """

    id: str
    suite: str
    messages: list[dict[str, str]]
    user_summary: str
    memory_context: str
    tools: list[str]
    metadata: dict[str, Any]


class EvalTrace(BaseModel):
    """Raw inference output for one eval prompt.

    Written by stage 2 (run_eval_inference), read by stage 3
    (score_eval_traces).
    """

    prompt_id: str
    suite: str
    model: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    thinking: str | None = None


class JudgeScore(BaseModel):
    """Score assigned by the judge model."""

    score: float
    justification: str
    sub_scores: dict[str, float] | None = None


class ScoredTrace(BaseModel):
    """A trace with its scores attached."""

    prompt_id: str
    suite: str
    judge_score: JudgeScore | None = None
    programmatic_pass: bool | None = None
    trace: EvalTrace


class SuiteResult(BaseModel):
    """Aggregate metrics for one eval suite."""

    suite: str
    mean_score: float
    n: int
    details: dict[str, Any]


class EvalReport(BaseModel):
    """Final report aggregating all suite results."""

    model: str
    timestamp: str
    suites: dict[str, SuiteResult]
    flagged_traces: list[str]
