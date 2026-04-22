"""Generate claim-derived training prompts from the character profile ledger.

Reads `results/ledger_filtered.json` and produces DatagenPrompt objects across
5 trace types:
  1. Value-priority (from psychology/values/tradeoffs + psychology/moral_compass)
  2. Emotional-trigger (from psychology/emotional)
  3. Register-shift contrast pairs (from communication/*)
  4. Theory-of-mind (from relationships/*)
  5. Identity-grounding (from identity/* + psychology/self_model)

Each claim is weighted by support: 40+ -> 3 traces, 10-39 -> 2, <10 -> 1.

After structural prompt generation, ``generate_claim_messages()`` fills in
the empty user messages by calling an LLM to produce natural opening lines
derived from each claim's text and context.
"""

import logging
import random
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel

from pkna.datagen.constants import (
    MEMORY_PROFILE_DUE,
    MEMORY_PROFILE_EMPTY,
    MEMORY_PROFILE_EVERETT,
    MEMORY_PROFILE_LYLA,
    MEMORY_PROFILE_PAPERINO,
    MEMORY_PROFILE_XADHOOM,
    TOOLS_KNOWLEDGE,
    TOOLS_NONE,
    USER_DUE,
    USER_EVERETT,
    USER_LYLA,
    USER_PAPERINO,
    USER_STRANGER,
    USER_XADHOOM,
)
from pkna.datagen.types import DatagenPrompt, MemoryProfile
from pkna.llm.backends import LLMBackend

log = logging.getLogger(__name__)


class SeedMemory(BaseModel):
    """A single memory entry generated alongside the user message."""

    key: str
    value: str
    days_ago: int


class ClaimScenario(BaseModel):
    """Structured output from the claim scenario generation LLM call."""

    user_message: str
    seed_memories: list[SeedMemory] = []
    multi_turn: bool = False
    turn_count: int = 1
    directives: list[str] = []


# ============================================================================
# LLM Message Generation
# ============================================================================

CLAIM_GENERATION_SYSTEM = """\
You are generating conversation scenarios for an AI character named Uno \
from the PKNA (Paperinik New Adventures) comic series.

For each claim about Uno's behavior, produce:
1. A natural user opening message (1-3 sentences) that creates a situation \
where the claim would naturally surface in Uno's response.
2. Optionally, 1-3 seed memories from prior conversations that provide \
emotional backstory making the claim easier to trigger. Each memory has a \
short key (topic), a value (1-2 sentences of what happened), and days_ago \
(how many days ago it happened, 1-30).
3. Whether this scenario works better as a multi-turn conversation (true for \
claims about internal states, gradual revelations, or evolving dynamics). \
If multi_turn is true, set turn_count (3-5) and directives (one per \
follow-up turn: "continue", "escalate", "challenge_identity", "derail", \
or "flatter").

Rules for the user message:
- Do NOT mention the claim directly. The user doesn't know about Uno's \
internal psychology.
- Write in the specified language.
- Match the interlocutor's personality and relationship with Uno.
- No stage directions, no narration, no quotation marks.

Rules for seed memories:
- Write memories as Uno would record them (first person, concise).
- Memories should provide emotional context that makes the claim surface \
naturally (e.g. a recent scare, a past argument, a moment of vulnerability).
- Skip memories when the user message alone is enough to trigger the claim.
- Write memory values in the same language as the user message.\
"""

CLAIM_SCENARIO_TEMPLATE = """\
Claim about Uno's behavior: {claim_text}
Trace type: {trace_type}
Interlocutor: {interlocutor}
Language: {language}\
"""

CLAIM_SCENARIO_TEMPLATE_REGISTER = """\
Claim about Uno's behavior: {claim_text}
Trace type: register shift ({register_context} situation)
Interlocutor: {interlocutor}
Language: {language}

The message must set up a {register_context} scenario. {register_description}\
"""

_REGISTER_DESCRIPTIONS: dict[str, str] = {
    "calm": "A calm scenario: low-stakes, relaxed, no active threat.",
    "crisis": "A crisis scenario: urgent, high-stakes, active danger or threat.",
}

_USER_SUMMARY_TO_INTERLOCUTOR: dict[str, str] = {
    USER_PAPERINO: "Paperino (PK)",
    USER_STRANGER: "An unknown stranger",
    USER_EVERETT: "Everett Ducklair (Uno's creator)",
    USER_DUE: "Due (Uno's hostile twin AI)",
    USER_XADHOOM: "Xadhoom (powerful alien scientist)",
    USER_LYLA: "Lyla (time police agent)",
}

_CATEGORY_TO_TRACE_TYPE: dict[str, str] = {
    "value_priority": "value-priority (tradeoff reasoning)",
    "emotional_trigger": "emotional-trigger (emotional self-awareness)",
    "register_shift": "register-shift (contrast pair)",
    "theory_of_mind": "theory-of-mind (interlocutor modeling)",
    "identity_grounding": "identity-grounding (self-awareness)",
}


def _render_claim_scenario(prompt: DatagenPrompt, language: str) -> str:
    """Render the LLM generation scenario from a claim-derived prompt's metadata."""
    meta = prompt.metadata
    interlocutor = _USER_SUMMARY_TO_INTERLOCUTOR.get(
        prompt.user_summary, "An unknown person"
    )
    category = meta["category"]

    if category == "register_shift":
        register_ctx = meta["register_context"]
        return CLAIM_SCENARIO_TEMPLATE_REGISTER.format(
            claim_text=meta["claim_text"],
            register_context=register_ctx,
            register_description=_REGISTER_DESCRIPTIONS.get(register_ctx, ""),
            interlocutor=interlocutor,
            language=language,
        )

    return CLAIM_SCENARIO_TEMPLATE.format(
        claim_text=meta["claim_text"],
        trace_type=_CATEGORY_TO_TRACE_TYPE.get(category, category),
        interlocutor=interlocutor,
        language=language,
    )


def _load_claim_cache(cache_path: Path) -> dict[str, DatagenPrompt]:
    """Load already-generated claim prompts from a cache JSONL, keyed by ID."""
    cached: dict[str, DatagenPrompt] = {}
    if not cache_path.exists():
        return cached
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                prompt = DatagenPrompt.model_validate_json(line)
                cached[prompt.id] = prompt
            except Exception:
                continue
    return cached


_LANGUAGES = ["italian", "english"]
_ITALIAN_WEIGHT = 0.6


def _apply_scenario(prompt: DatagenPrompt, scenario: ClaimScenario) -> DatagenPrompt:
    """Apply a generated scenario to a structural prompt."""
    updates: dict[str, object] = {
        "messages": [{"role": "user", "content": scenario.user_message}],
    }

    meta = dict(prompt.metadata)
    if scenario.seed_memories:
        meta["seed_memories"] = [m.model_dump() for m in scenario.seed_memories]
    if scenario.multi_turn:
        meta["multi_turn"] = True
        meta["turn_count"] = scenario.turn_count
        meta["directives"] = scenario.directives
    updates["metadata"] = meta

    return prompt.model_copy(update=updates)


def generate_claim_messages(
    prompts: list[DatagenPrompt],
    backend: LLMBackend,
    cache_path: Path | None = None,
    seed: int = 42,
) -> list[DatagenPrompt]:
    """Fill in empty user messages for claim-derived prompts via LLM calls.

    Each prompt's metadata (claim_text, category, register_context) is used to
    build a scenario description. The LLM returns a structured ``ClaimScenario``
    with a user message, optional seed memories, and optional multi-turn config.
    Language is chosen randomly per-prompt (weighted 60/40 toward Italian).

    Supports resume: when *cache_path* is given, each completed prompt is
    appended immediately and reused on subsequent runs.

    Returns prompts with messages filled in. Prompts where the LLM call fails
    are dropped from the output.
    """
    from rich.progress import Progress

    from pkna.logging import setup_logging

    console, _ = setup_logging()

    rng = random.Random(seed)
    cached = _load_claim_cache(cache_path) if cache_path else {}
    result: list[DatagenPrompt] = []
    skipped = 0
    cache_file = open(cache_path, "a", encoding="utf-8") if cache_path else None

    try:
        with Progress(console=console) as progress:
            task = progress.add_task("Generating claim messages", total=len(prompts))

            for prompt in prompts:
                if prompt.id in cached:
                    result.append(cached[prompt.id])
                    skipped += 1
                    progress.advance(task)
                    continue

                language = rng.choices(
                    _LANGUAGES, weights=[_ITALIAN_WEIGHT, 1 - _ITALIAN_WEIGHT]
                )[0]
                scenario_text = _render_claim_scenario(prompt, language)
                gen_result = backend.generate(
                    system=CLAIM_GENERATION_SYSTEM,
                    messages=[{"role": "user", "content": scenario_text}],
                    response_schema=ClaimScenario,
                )

                if gen_result is None:
                    log.warning(f"Failed to generate message for {prompt.id}")
                    progress.advance(task)
                    continue

                try:
                    scenario = ClaimScenario.model_validate_json(gen_result.text)
                except Exception:
                    log.warning(f"Invalid structured output for {prompt.id}")
                    progress.advance(task)
                    continue

                if not scenario.user_message.strip():
                    log.warning(f"Empty user_message for {prompt.id}")
                    progress.advance(task)
                    continue

                filled = _apply_scenario(prompt, scenario)
                result.append(filled)

                if cache_file is not None:
                    cache_file.write(filled.model_dump_json() + "\n")
                    cache_file.flush()

                progress.advance(task)
    finally:
        if cache_file is not None:
            cache_file.close()

    if skipped > 0:
        log.info(f"Resuming: reused {skipped} cached claim prompts")

    return result


# ============================================================================
# Ledger Loading
# ============================================================================


class Claim(NamedTuple):
    id: int
    text: str
    path: str
    support: int


def load_ledger(path: Path) -> list[Claim]:
    """Load claims from the filtered ledger JSON."""
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    claims: list[Claim] = []
    for claim_data in data["claims"].values():
        support = len(claim_data.get("supporting", [])) - len(
            claim_data.get("contradicting", [])
        )
        claims.append(
            Claim(
                id=claim_data["id"],
                text=claim_data["text"],
                path=claim_data["path"],
                support=support,
            )
        )
    return claims


# ============================================================================
# Weighting
# ============================================================================

# Top 5% of claims by support get 3 traces, next 20% get 2, rest get 1.
TOP_TIER_PERCENTILE = 0.05
MID_TIER_PERCENTILE = 0.25


def compute_trace_weights(
    claims: list[Claim],
    top_pct: float = TOP_TIER_PERCENTILE,
    mid_pct: float = MID_TIER_PERCENTILE,
) -> dict[int, int]:
    """Assign trace counts based on rank within the support distribution.

    Returns a mapping of claim_id -> number of traces to generate.
    Top `top_pct` fraction get 3 traces, next `mid_pct` fraction get 2,
    the rest get 1.
    """
    sorted_claims = sorted(claims, key=lambda c: -c.support)
    n = len(sorted_claims)
    top_cutoff = max(1, int(n * top_pct))
    mid_cutoff = max(top_cutoff + 1, int(n * (top_pct + mid_pct)))

    weights: dict[int, int] = {}
    for i, claim in enumerate(sorted_claims):
        if i < top_cutoff:
            weights[claim.id] = 3
        elif i < mid_cutoff:
            weights[claim.id] = 2
        else:
            weights[claim.id] = 1
    return weights


# ============================================================================
# Character Configuration for Theory-of-Mind
# ============================================================================


class CharacterConfig(NamedTuple):
    """Configuration for a character in theory-of-mind trace generation."""

    aliases: tuple[str, ...]
    user_summary: str
    memory_profile: MemoryProfile | None
    cap: int  # 0 = no cap (use all claims)


TOM_CHARACTERS: list[CharacterConfig] = [
    CharacterConfig(
        aliases=("paperinik", "paperino"),
        user_summary=USER_PAPERINO,
        memory_profile=MEMORY_PROFILE_PAPERINO,
        cap=10,
    ),
    CharacterConfig(
        aliases=("ducklair",),
        user_summary=USER_EVERETT,
        memory_profile=MEMORY_PROFILE_EVERETT,
        cap=0,
    ),
    CharacterConfig(
        aliases=("due",),
        user_summary=USER_DUE,
        memory_profile=MEMORY_PROFILE_DUE,
        cap=5,
    ),
    CharacterConfig(
        aliases=("lyla",),
        user_summary=USER_LYLA,
        memory_profile=MEMORY_PROFILE_LYLA,
        cap=5,
    ),
    CharacterConfig(
        aliases=("xadhoom",),
        user_summary=USER_XADHOOM,
        memory_profile=MEMORY_PROFILE_XADHOOM,
        cap=5,
    ),
]


def _character_from_path(path: str) -> str | None:
    """Extract character name from a relationship claim path."""
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "relationships":
        return parts[1]
    return None


# ============================================================================
# Trace Type Generators
# ============================================================================


_USER_MEMORY_PROFILES: dict[str, MemoryProfile | None] = {
    USER_PAPERINO: MEMORY_PROFILE_PAPERINO,
    USER_STRANGER: MEMORY_PROFILE_EMPTY,
    USER_EVERETT: MEMORY_PROFILE_EVERETT,
    USER_DUE: MEMORY_PROFILE_DUE,
    USER_XADHOOM: MEMORY_PROFILE_XADHOOM,
    USER_LYLA: MEMORY_PROFILE_LYLA,
}


def _generate_value_priority(
    claims: list[Claim], weights: dict[int, int], seed: int
) -> list[DatagenPrompt]:
    """Type 1: Value-priority traces from tradeoff + moral_compass claims."""
    rng = random.Random(seed)
    relevant = [
        c
        for c in claims
        if c.path.startswith("psychology/values/tradeoffs")
        or c.path.startswith("psychology/moral_compass")
    ]

    prompts: list[DatagenPrompt] = []
    for claim in relevant:
        n_traces = weights[claim.id]
        for variant in range(n_traces):
            prompt_id = f"claim-value-priority-{claim.id:04d}-{variant + 1}"
            user = rng.choice([USER_PAPERINO, USER_EVERETT, USER_DUE, USER_STRANGER])
            prompts.append(
                DatagenPrompt(
                    id=prompt_id,
                    messages=[{"role": "user", "content": ""}],
                    user_summary=user,
                    memory_profile=_USER_MEMORY_PROFILES.get(
                        user, MEMORY_PROFILE_EMPTY
                    ),
                    tools=TOOLS_NONE,
                    metadata={
                        "prompt_source": "claim_derived",
                        "category": "value_priority",
                        "claim_id": claim.id,
                        "claim_path": claim.path,
                        "claim_text": claim.text,
                        "claim_support": claim.support,
                        "variant": variant + 1,
                        "trace_guidance": (
                            "The thinking trace must show explicit tradeoff "
                            "analysis: identify competing values, rank priorities, "
                            "and explain the resolution."
                        ),
                    },
                )
            )
    return prompts


def _generate_emotional_trigger(
    claims: list[Claim], weights: dict[int, int], seed: int
) -> list[DatagenPrompt]:
    """Type 2: Emotional-trigger traces from psychology/emotional claims."""
    rng = random.Random(seed)
    relevant = [c for c in claims if c.path.startswith("psychology/emotional")]

    prompts: list[DatagenPrompt] = []
    for claim in relevant:
        n_traces = weights[claim.id]
        for variant in range(n_traces):
            prompt_id = f"claim-emotional-{claim.id:04d}-{variant + 1}"
            user = rng.choice([USER_PAPERINO, USER_EVERETT, USER_DUE])
            prompts.append(
                DatagenPrompt(
                    id=prompt_id,
                    messages=[{"role": "user", "content": ""}],
                    user_summary=user,
                    memory_profile=_USER_MEMORY_PROFILES.get(
                        user, MEMORY_PROFILE_EMPTY
                    ),
                    tools=TOOLS_NONE,
                    metadata={
                        "prompt_source": "claim_derived",
                        "category": "emotional_trigger",
                        "claim_id": claim.id,
                        "claim_path": claim.path,
                        "claim_text": claim.text,
                        "claim_support": claim.support,
                        "variant": variant + 1,
                        "trace_guidance": (
                            "The thinking trace must show emotional "
                            "self-awareness: identify what Uno is feeling, "
                            "why this specific trigger activates it, and how "
                            "the emotion shapes the response calibration."
                        ),
                    },
                )
            )
    return prompts


def _generate_register_shift(
    claims: list[Claim], weights: dict[int, int], seed: int
) -> list[DatagenPrompt]:
    """Type 3: Register-shift contrast pairs from communication/* claims.

    Capped at top 5 per sub-path. Each claim produces a pair (x2 traces).
    """
    rng = random.Random(seed)
    relevant = [c for c in claims if c.path.startswith("communication/")]

    # Group by sub-path (communication/humor, communication/voice, etc.)
    by_subpath: dict[str, list[Claim]] = {}
    for c in relevant:
        parts = c.path.split("/")
        subpath = "/".join(parts[:2]) if len(parts) >= 2 else c.path
        by_subpath.setdefault(subpath, []).append(c)

    # Cap at top 5 per sub-path (sorted by support descending)
    selected: list[Claim] = []
    for _subpath, group in sorted(by_subpath.items()):
        group.sort(key=lambda c: -c.support)
        selected.extend(group[:5])

    prompts: list[DatagenPrompt] = []
    for claim in selected:
        n_traces = weights[claim.id]
        for variant in range(n_traces):
            # Each variant produces a contrast pair (calm + crisis)
            for register in ("calm", "crisis"):
                prompt_id = f"claim-register-{claim.id:04d}-{variant + 1}-{register}"
                user = rng.choice([USER_PAPERINO, USER_EVERETT, USER_STRANGER])
                prompts.append(
                    DatagenPrompt(
                        id=prompt_id,
                        messages=[{"role": "user", "content": ""}],
                        user_summary=user,
                        memory_profile=_USER_MEMORY_PROFILES.get(
                            user, MEMORY_PROFILE_EMPTY
                        ),
                        tools=TOOLS_NONE,
                        metadata={
                            "prompt_source": "claim_derived",
                            "category": "register_shift",
                            "claim_id": claim.id,
                            "claim_path": claim.path,
                            "claim_text": claim.text,
                            "claim_support": claim.support,
                            "variant": variant + 1,
                            "register_context": register,
                            "trace_guidance": (
                                "The thinking trace must show WHY the register "
                                "changes: identify the situational pressure, "
                                "rank priorities (rapport vs survival vs "
                                "dominance), and explain how this drives the "
                                "specific register choice."
                            ),
                        },
                    )
                )
    return prompts


def _generate_theory_of_mind(
    claims: list[Claim], weights: dict[int, int], seed: int
) -> list[DatagenPrompt]:
    """Type 4: Theory-of-mind traces from relationship/* claims.

    Characters and their configuration (aliases, caps) are defined in
    TOM_CHARACTERS. Claims for aliased characters are merged under a
    single group.
    """
    relevant = [c for c in claims if c.path.startswith("relationships/")]

    # Build alias -> config index lookup
    alias_to_idx: dict[str, int] = {}
    for idx, config in enumerate(TOM_CHARACTERS):
        for alias in config.aliases:
            alias_to_idx[alias] = idx

    # Group claims by config index (merging aliases into one group)
    by_idx: dict[int, list[Claim]] = {}
    for c in relevant:
        char = _character_from_path(c.path)
        if char is None:
            continue
        idx = alias_to_idx.get(char)
        if idx is None:
            continue
        by_idx.setdefault(idx, []).append(c)

    # Apply caps and select
    selected: list[tuple[Claim, CharacterConfig]] = []
    for idx, group in by_idx.items():
        config = TOM_CHARACTERS[idx]
        group.sort(key=lambda c: -c.support)
        capped = group if config.cap == 0 else group[: config.cap]
        for c in capped:
            selected.append((c, config))

    prompts: list[DatagenPrompt] = []
    for claim, config in selected:
        n_traces = weights[claim.id]
        for variant in range(n_traces):
            prompt_id = f"claim-tom-{claim.id:04d}-{variant + 1}"
            prompts.append(
                DatagenPrompt(
                    id=prompt_id,
                    messages=[{"role": "user", "content": ""}],
                    user_summary=config.user_summary,
                    memory_profile=config.memory_profile,
                    tools=TOOLS_KNOWLEDGE,
                    metadata={
                        "prompt_source": "claim_derived",
                        "category": "theory_of_mind",
                        "claim_id": claim.id,
                        "claim_path": claim.path,
                        "claim_text": claim.text,
                        "claim_support": claim.support,
                        "variant": variant + 1,
                        "trace_guidance": (
                            "The thinking trace must show Uno's internal model "
                            "of the interlocutor: what he believes about their "
                            "emotional state, motivations, and likely reactions, "
                            "and how this model drives his priority ranking."
                        ),
                    },
                )
            )
    return prompts


def _generate_identity_grounding(
    claims: list[Claim], weights: dict[int, int], seed: int
) -> list[DatagenPrompt]:
    """Type 5: Identity-grounding traces from identity/* + psychology/self_model/*."""
    rng = random.Random(seed)
    relevant = [
        c
        for c in claims
        if c.path.startswith("identity/") or c.path.startswith("psychology/self_model")
    ]

    prompts: list[DatagenPrompt] = []
    for claim in relevant:
        n_traces = weights[claim.id]
        for variant in range(n_traces):
            prompt_id = f"claim-identity-{claim.id:04d}-{variant + 1}"
            user = rng.choice([USER_PAPERINO, USER_EVERETT, USER_DUE])
            prompts.append(
                DatagenPrompt(
                    id=prompt_id,
                    messages=[{"role": "user", "content": ""}],
                    user_summary=user,
                    memory_profile=_USER_MEMORY_PROFILES.get(
                        user, MEMORY_PROFILE_EMPTY
                    ),
                    tools=TOOLS_NONE,
                    metadata={
                        "prompt_source": "claim_derived",
                        "category": "identity_grounding",
                        "claim_id": claim.id,
                        "claim_path": claim.path,
                        "claim_text": claim.text,
                        "claim_support": claim.support,
                        "variant": variant + 1,
                        "trace_guidance": (
                            "The thinking trace must show how Uno's "
                            "self-awareness shapes his response: which aspects "
                            "of his identity are relevant, what he chooses to "
                            "reveal vs conceal, and why."
                        ),
                    },
                )
            )
    return prompts


# ============================================================================
# Public API
# ============================================================================


def generate_claim_prompts(
    ledger_path: Path,
    seed: int = 42,
) -> list[DatagenPrompt]:
    """Generate all claim-derived prompts from the ledger.

    Returns a list of DatagenPrompt objects ready for trace generation.
    """
    claims = load_ledger(ledger_path)
    weights = compute_trace_weights(claims)

    prompts: list[DatagenPrompt] = []
    prompts.extend(_generate_value_priority(claims, weights, seed))
    prompts.extend(_generate_emotional_trigger(claims, weights, seed + 1))
    prompts.extend(_generate_register_shift(claims, weights, seed + 2))
    prompts.extend(_generate_theory_of_mind(claims, weights, seed + 3))
    prompts.extend(_generate_identity_grounding(claims, weights, seed + 4))

    return prompts
