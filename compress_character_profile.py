#!/usr/bin/env python3

"""
Compress the detailed character profile into tiered versions.

This script uses DSPy with Gemini to intelligently compress the full 75k token
Uno character profile into:
- Tier 1 (Core): ~5k tokens - Essential identity, core traits, key relationships
- Tier 2 (Extended): ~15k tokens - More detail, additional nuances
- Tier 3 (Full): Copy of original 75k token profile

The compression focuses on patterns over examples while preserving character essence.
"""

import json
import logging
import os
from pathlib import Path

import dspy
import tiktoken
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler


# Configure logging
console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)

# Settings
MODEL_NAME = "vertex_ai/gemini-3-flash-preview"
ENCODING_NAME = "cl100k_base"  # GPT-4 tokenizer as approximation

# Paths
BASE_DIR = Path(__file__).parent
SOURCE_PROFILE = (
    BASE_DIR / "output" / "character-profile" / "uno" / "v2" / "uno_profile.md"
)
OUTPUT_DIR = BASE_DIR / "output" / "character-profile" / "uno" / "v3"

# Target token counts for each tier
TARGET_TOKENS = {
    "tier1": 5000,
    "tier2": 15000,
}


def configure_lm() -> None:
    """Configure DSPy language model."""
    load_dotenv()
    lm = dspy.LM(
        model=MODEL_NAME,
        vertex_credentials=os.getenv("VERTEX_AI_CREDS"),
        vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        temperature=0.7,  # Lower temperature for more consistent compression
        top_p=0.95,
        top_k=40,
        max_tokens=65535,
    )
    dspy.configure(lm=lm, track_usage=True)


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(text))


class ProfileCompressor(dspy.Signature):
    """Compress a detailed character profile to essential elements.

    CRITICAL INSTRUCTIONS:

    1. PRESERVE CHARACTER ESSENCE:
       - Focus on patterns and generalizable traits, not individual examples
       - Maintain distinctive voice markers (sarcasm, Italian expressions)
       - Keep critical relationships and behavioral constraints
       - Preserve what makes this character unique

    2. COMPRESSION STRATEGY:
       - Generalize from specific examples into pattern descriptions
       - Combine redundant or similar traits
       - Keep 1-2 Italian dialogue examples per major trait (with translations)
       - Prioritize "what" the character does/is over exhaustive "how" examples

    3. LANGUAGE HANDLING:
       - Write all analysis and descriptions in English
       - Preserve select Italian dialogue examples (2-3 per major section)
       - Include English translations in parentheses

    4. STRUCTURE FOR TIER 1 (CORE):
       # Uno - Core Character Profile

       ## Essential Identity
       - Key facts about who/what Uno is
       - Core purpose and role
       - Critical constraints (cannot be turned off, no biological needs, etc.)

       ## Core Personality (Top 10-15 Traits)
       - Most distinctive personality patterns
       - Each trait with 1 example if relevant

       ## Communication Style
       - Key speech patterns and linguistic markers
       - Typical phrases and expressions

       ## Behavioral Guidelines
       ### What Uno Does:
       - Clear list of characteristic behaviors

       ### What Uno Doesn't Do:
       - Explicit constraints and limitations

       ## Key Relationships
       - Essential relationship dynamics (Paperinik, Everett, Due)
       - Core patterns of interaction

    5. STRUCTURE FOR TIER 2 (EXTENDED):
       - Same structure as Tier 1 but with:
       - More personality traits (20-30 instead of 10-15)
       - More detailed relationship descriptions
       - Additional dialogue examples
       - More communication patterns

    6. QUALITY OVER QUANTITY:
       - Better to have 10 well-defined traits than 50 vague ones
       - Each trait should be distinct and actionable
       - Avoid redundancy across sections
    """

    full_profile: str = dspy.InputField(desc="The complete detailed character profile")
    target_tokens: int = dspy.InputField(desc="Target token count for compression")
    tier_name: str = dspy.InputField(desc="Tier name (tier1/core or tier2/extended)")

    compressed_profile: str = dspy.OutputField(
        desc="Compressed profile following the structure and token target"
    )
    traits_preserved: list[str] = dspy.OutputField(
        desc="List of key traits preserved in this compression"
    )


def compress_profile(full_text: str, tier: str, target_tokens: int) -> dict:
    """Compress profile to target tier using DSPy.

    Args:
        full_text: Full profile content
        tier: Tier name ("tier1" or "tier2")
        target_tokens: Target token count

    Returns:
        Dict with compressed_profile and traits_preserved
    """
    log.info(f"Compressing to {tier} (target: ~{target_tokens:,} tokens)...")

    compressor = dspy.ChainOfThought(ProfileCompressor)
    result = compressor(
        full_profile=full_text,
        target_tokens=target_tokens,
        tier_name=tier,
    )

    compressed_text = result.compressed_profile
    actual_tokens = count_tokens(compressed_text)

    log.info(
        f"  Generated {actual_tokens:,} tokens ({actual_tokens / target_tokens * 100:.1f}% of target)"
    )

    return {
        "compressed_profile": compressed_text,
        "traits_preserved": result.traits_preserved,
        "actual_tokens": actual_tokens,
        "target_tokens": target_tokens,
    }


def save_profile(content: str, tier: str, metadata: dict, output_dir: Path) -> Path:
    """Save compressed profile to file.

    Args:
        content: Profile content
        tier: Tier name
        metadata: Metadata dict
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    tier_map = {
        "tier1": "uno_profile_tier1.md",
        "tier2": "uno_profile_tier2.md",
        "tier3": "uno_profile_tier3.md",
    }

    output_file = output_dir / tier_map[tier]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write profile
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    # Save metadata
    metadata_file = output_dir / f"{tier}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"  Saved: {output_file}")
    log.info(f"  Metadata: {metadata_file}")

    return output_file


def main() -> None:
    """Main entry point."""
    console.print("\n[bold cyan]Character Profile Compression[/bold cyan]\n")

    # Check source profile exists
    if not SOURCE_PROFILE.exists():
        log.error(f"Source profile not found: {SOURCE_PROFILE}")
        return

    # Configure DSPy
    configure_lm()

    # Load source profile
    log.info(f"Loading source profile: {SOURCE_PROFILE}")
    with open(SOURCE_PROFILE, encoding="utf-8") as f:
        full_profile = f.read()

    source_tokens = count_tokens(full_profile)
    log.info(f"Source profile: {source_tokens:,} tokens")
    console.print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tier 3: Just copy the full profile
    console.print("[bold]Creating Tier 3 (Full Profile)[/bold]")
    save_profile(
        content=full_profile,
        tier="tier3",
        metadata={
            "tier": "tier3",
            "description": "Full profile (copy of v2)",
            "tokens": source_tokens,
            "source": str(SOURCE_PROFILE),
        },
        output_dir=OUTPUT_DIR,
    )
    console.print()

    # Tier 1: Core profile (~5k tokens)
    console.print("[bold]Creating Tier 1 (Core Profile)[/bold]")
    tier1_result = compress_profile(
        full_text=full_profile,
        tier="tier1",
        target_tokens=TARGET_TOKENS["tier1"],
    )
    save_profile(
        content=tier1_result["compressed_profile"],
        tier="tier1",
        metadata={
            "tier": "tier1",
            "description": "Core profile - essential traits and patterns",
            "target_tokens": tier1_result["target_tokens"],
            "actual_tokens": tier1_result["actual_tokens"],
            "compression_ratio": f"{source_tokens / tier1_result['actual_tokens']:.1f}x",
            "traits_preserved": tier1_result["traits_preserved"],
        },
        output_dir=OUTPUT_DIR,
    )
    console.print()

    # Tier 2: Extended profile (~15k tokens)
    console.print("[bold]Creating Tier 2 (Extended Profile)[/bold]")
    tier2_result = compress_profile(
        full_text=full_profile,
        tier="tier2",
        target_tokens=TARGET_TOKENS["tier2"],
    )
    save_profile(
        content=tier2_result["compressed_profile"],
        tier="tier2",
        metadata={
            "tier": "tier2",
            "description": "Extended profile - more detail and examples",
            "target_tokens": tier2_result["target_tokens"],
            "actual_tokens": tier2_result["actual_tokens"],
            "compression_ratio": f"{source_tokens / tier2_result['actual_tokens']:.1f}x",
            "traits_preserved": tier2_result["traits_preserved"],
        },
        output_dir=OUTPUT_DIR,
    )
    console.print()

    # Summary
    console.print("[bold green]✓ Compression Complete[/bold green]\n")
    console.print("Profile Tiers:")
    console.print(
        f"  • Tier 1 (Core):     {tier1_result['actual_tokens']:>6,} tokens  "
        f"({source_tokens / tier1_result['actual_tokens']:>4.1f}x compression)"
    )
    console.print(
        f"  • Tier 2 (Extended): {tier2_result['actual_tokens']:>6,} tokens  "
        f"({source_tokens / tier2_result['actual_tokens']:>4.1f}x compression)"
    )
    console.print(f"  • Tier 3 (Full):     {source_tokens:>6,} tokens  (original)")
    console.print()
    console.print(f"Output directory: {OUTPUT_DIR}")
    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print(
        "[dim]1. Test with: ./generate_from_character_profile.py --tier core[/dim]"
    )
    console.print("[dim]2. Compare quality across tiers (core, extended, full)[/dim]")
    console.print("[dim]3. Run restructure_profile.py for further optimization[/dim]")


if __name__ == "__main__":
    main()
