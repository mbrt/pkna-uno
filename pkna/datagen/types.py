"""Shared data models for the dataset generation pipeline.

Defines the contract between the datagen stages:
1. generate_prompts -> DatagenPrompt (JSONL)
2. run_datagen -> DatagenTrace (JSONL)
3. filter_traces -> ScoredTrace with QualityScore (JSONL)
"""

from typing import Any, Literal

from pydantic import BaseModel


class MemoryCorpusEntry(BaseModel):
    """A single tagged entry in the shared memory corpus."""

    key: str
    value: str
    timestamp: str
    tags: list[str]
    archetype: str  # "roleplay" or "casual"
    character: str  # e.g. "paperino", "xadhoom", "anonymous"


class MemoryProfile(BaseModel):
    """Describes how to compose per-trace memory from the corpus."""

    archetype: str  # "roleplay" or "casual"
    character: str  # e.g. "paperino", "anonymous"
    relevant_tags: list[str]  # tags that should match
    n_relevant: int  # how many relevant entries to include
    n_irrelevant: int  # how many irrelevant/noise entries to include


class DatagenPrompt(BaseModel):
    """A single prompt ready for trace generation.

    Written by generate_prompts, read by run_datagen.
    """

    id: str
    messages: list[dict[str, str]]
    user_summary: str
    memory_context: str  # kept for backward compat
    memory_bank_id: str = ""  # kept for backward compat
    memory_profile: MemoryProfile | None = None
    tools: list[str]
    metadata: dict[str, Any]


class DatagenTrace(BaseModel):
    """One recorded training example with full context.

    The system prompt is stored externally (template + profile sidecar files)
    so it can be swapped at SFT time. Per-prompt context (user_summary,
    memory_context) is prepended to the first user message.
    """

    id: str
    metadata: dict[str, Any]
    memory_context: str
    user_summary: str
    messages: list[dict[str, Any]]


ToolCorrectnessResult = Literal["pass", "fail", "na"]


class QualityScore(BaseModel):
    """Judge output for a single trace."""

    trace_id: str
    character_consistency: float
    thinking_quality: float
    tool_correctness: ToolCorrectnessResult
    language_consistent: bool
    response_length_ok: bool
    overall_pass: bool
    justification: str


class ScoredTrace(BaseModel):
    """A trace with its quality score attached."""

    trace: DatagenTrace
    score: QualityScore
