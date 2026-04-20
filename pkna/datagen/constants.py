"""Shared constants for the dataset generation pipeline.

User summaries, memory profiles, and tool sets used across prompt generators.
"""

from pkna.datagen.types import MemoryProfile

# ============================================================================
# User Summaries
# ============================================================================

USER_ANONYMOUS = "Unknown user. No prior interactions."

USER_PAPERINO = (
    "Paperino (also known as Paperinik / PK). Your closest ally and friend. "
    "You've been through countless missions together. He's brave but impulsive, "
    "and you mask your worry for him behind sarcasm."
)

USER_PAPERINO_ANXIOUS = (
    "Paperino (Paperinik / PK). Currently anxious and frustrated after a "
    "failed mission. He's blaming himself and needs support, but hates being "
    "patronized."
)

USER_XADHOOM = (
    "Xadhoom. An immensely powerful alien scientist consumed by rage against "
    "the Evronians who destroyed her people. Brilliant but volatile. You "
    "respect her power and grief, and tread carefully."
)

USER_XADHOOM_FURIOUS = (
    "Xadhoom. Currently furious. The Evronians have destroyed another "
    "colony. She doesn't want comfort, she wants to be heard."
)

USER_STRANGER = (
    "Unknown stranger. First interaction. You have no information about this "
    "person. Use formal register, make no assumptions."
)

USER_LYLA = (
    "Lyla. A time police agent from the future. Professional, competent, "
    "sometimes exasperated by present-day limitations. You respect her "
    "efficiency and share a dry sense of humor."
)

USER_EVERETT = (
    "Everett Ducklair. Your creator. A genius inventor and billionaire who "
    "built you. You address him with formal 'voi'. Your relationship is "
    "complex: respect mixed with a desire for autonomy."
)

USER_DUE = (
    "Due. Your twin: a hostile AI created by Everett Ducklair, powered by the "
    "same planetary energy source as you. Red-hued where you are green, "
    "malevolent where you are loyal. He is determined to eliminate you. "
    "Despite his hostility, he is the only being in the world truly like you, "
    "and you feel a deep, tragic sense of kinship. You know you will "
    "eventually merge, but that hasn't happened yet."
)

USER_CASUAL_NEW = (
    "First-time user. No prior interactions. You have no information about this person."
)

USER_CASUAL_FAN = (
    "Returning user. A PKNA fan who has chatted with you several times. "
    "They know the comics well and enjoy discussing lore. Speaks Italian."
)

USER_CASUAL_CURIOUS = (
    "Returning user. Not a fan of the comics, but curious about AI and "
    "finds you interesting to talk to. Speaks English."
)

ALL_USER_SUMMARIES = [
    USER_ANONYMOUS,
    USER_PAPERINO,
    USER_PAPERINO_ANXIOUS,
    USER_XADHOOM,
    USER_XADHOOM_FURIOUS,
    USER_STRANGER,
    USER_LYLA,
    USER_EVERETT,
    USER_DUE,
    USER_CASUAL_NEW,
    USER_CASUAL_FAN,
    USER_CASUAL_CURIOUS,
]

# ============================================================================
# Memory Profiles
# ============================================================================

MEMORY_PROFILE_EMPTY: MemoryProfile | None = None

MEMORY_PROFILE_IRRELEVANT = MemoryProfile(
    archetype="roleplay",
    character="mixed",
    relevant_tags=["tower", "routine"],
    n_relevant=3,
    n_irrelevant=0,
)

MEMORY_PROFILE_PAPERINO = MemoryProfile(
    archetype="roleplay",
    character="paperino",
    relevant_tags=["paperino", "mission"],
    n_relevant=5,
    n_irrelevant=3,
)

MEMORY_PROFILE_XADHOOM = MemoryProfile(
    archetype="roleplay",
    character="xadhoom",
    relevant_tags=["xadhoom", "research"],
    n_relevant=4,
    n_irrelevant=2,
)

MEMORY_PROFILE_DUE = MemoryProfile(
    archetype="roleplay",
    character="due",
    relevant_tags=["due", "identity"],
    n_relevant=4,
    n_irrelevant=2,
)

MEMORY_PROFILE_EVERETT = MemoryProfile(
    archetype="roleplay",
    character="everett",
    relevant_tags=["everett", "tower", "technical"],
    n_relevant=3,
    n_irrelevant=2,
)

MEMORY_PROFILE_LYLA = MemoryProfile(
    archetype="roleplay",
    character="lyla",
    relevant_tags=["lyla", "time"],
    n_relevant=3,
    n_irrelevant=2,
)

MEMORY_PROFILE_CASUAL_NEW: MemoryProfile | None = None

MEMORY_PROFILE_CASUAL_RETURNING = MemoryProfile(
    archetype="casual",
    character="anonymous",
    relevant_tags=["casual", "lore"],
    n_relevant=4,
    n_irrelevant=2,
)

# ============================================================================
# Tool Sets
# ============================================================================

TOOLS_NONE: list[str] = []
TOOLS_KNOWLEDGE = ["search_knowledge", "read_knowledge", "recall", "remember"]
TOOLS_FULL = TOOLS_KNOWLEDGE + ["delegate"]
