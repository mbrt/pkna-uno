"""Generate claim-derived training prompts from the character profile ledger.

Reads `results/ledger_filtered.json` and produces DatagenPrompt objects across
5 trace types:
  1. Value-priority (from psychology/values/tradeoffs + psychology/moral_compass)
  2. Emotional-trigger (from psychology/emotional)
  3. Register-shift contrast pairs (from communication/*)
  4. Theory-of-mind (from relationships/*)
  5. Identity-grounding (from identity/* + psychology/self_model)

Each claim is weighted by support: 40+ -> 3 traces, 10-39 -> 2, <10 -> 1.
"""

import random
from pathlib import Path
from typing import NamedTuple

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
            prompts.append(
                DatagenPrompt(
                    id=prompt_id,
                    messages=[{"role": "user", "content": ""}],
                    user_summary=rng.choice([USER_PAPERINO, USER_STRANGER]),
                    memory_profile=MEMORY_PROFILE_EMPTY,
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
            prompts.append(
                DatagenPrompt(
                    id=prompt_id,
                    messages=[{"role": "user", "content": ""}],
                    user_summary=rng.choice([USER_PAPERINO, USER_STRANGER]),
                    memory_profile=MEMORY_PROFILE_EMPTY,
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
                prompts.append(
                    DatagenPrompt(
                        id=prompt_id,
                        messages=[{"role": "user", "content": ""}],
                        user_summary=rng.choice(
                            [USER_PAPERINO, USER_EVERETT, USER_STRANGER]
                        ),
                        memory_profile=MEMORY_PROFILE_EMPTY,
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
            prompts.append(
                DatagenPrompt(
                    id=prompt_id,
                    messages=[{"role": "user", "content": ""}],
                    user_summary=rng.choice(
                        [USER_PAPERINO, USER_STRANGER, USER_EVERETT]
                    ),
                    memory_profile=rng.choice(
                        [MEMORY_PROFILE_EMPTY, MEMORY_PROFILE_PAPERINO]
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
