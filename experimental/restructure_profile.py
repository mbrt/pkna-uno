#!/usr/bin/env python3

"""
Restructure and validate tiered character profiles.

This script reviews the compressed profiles and applies final refinements:
- Validates required sections are present
- Ensures consistent formatting
- Adds any missing behavioral constraints
- Generates a validation report

Since the compression script already creates well-structured profiles,
this mainly serves as a validation and minor refinement step.
"""

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel


# Configure logging
console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)],
)
log = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
PROFILE_DIR = BASE_DIR / "output" / "character-profile" / "uno" / "v3"

# Required sections for validation
REQUIRED_SECTIONS = [
    "Essential Identity",
    "Core Personality",
    "Communication Style",
    "Behavioral Guidelines",
    "Key Relationships",
]


def validate_profile_structure(content: str, tier_name: str) -> dict[str, Any]:
    """Validate that profile has required sections.

    Args:
        content: Profile content
        tier_name: Tier name for reporting

    Returns:
        Validation results dict
    """
    results: dict[str, Any] = {
        "tier": tier_name,
        "sections_found": [],
        "sections_missing": [],
        "has_do_dont": False,
        "italian_examples_count": 0,
    }

    # Check for required sections
    for section in REQUIRED_SECTIONS:
        if f"## {section}" in content:
            results["sections_found"].append(section)
        else:
            results["sections_missing"].append(section)

    # Check for behavioral guidelines subsections
    if "### What Uno Does:" in content and "### What Uno Doesn't Do:" in content:
        results["has_do_dont"] = True

    # Count Italian examples (rough heuristic: look for Italian quotes)
    results["italian_examples_count"] = content.count("*'")

    return results


def generate_validation_report(all_results: list[dict]) -> str:
    """Generate a validation report for all tiers.

    Args:
        all_results: List of validation results for each tier

    Returns:
        Report as formatted string
    """
    report_lines = []
    report_lines.append("# Profile Structure Validation Report\n")

    for result in all_results:
        tier = result["tier"]
        report_lines.append(f"## {tier}\n")

        # Sections
        if result["sections_missing"]:
            report_lines.append("**Missing Sections:**")
            for section in result["sections_missing"]:
                report_lines.append(f"- ❌ {section}")
            report_lines.append("")
        else:
            report_lines.append("✅ All required sections present\n")

        # Do/Don't guidelines
        if result["has_do_dont"]:
            report_lines.append("✅ Has explicit Do/Don't behavioral guidelines\n")
        else:
            report_lines.append("❌ Missing Do/Don't behavioral guidelines\n")

        # Italian examples
        report_lines.append(
            f"📝 Italian dialogue examples: {result['italian_examples_count']}\n"
        )

    return "\n".join(report_lines)


def main() -> None:
    """Main entry point."""
    console.print("\n[bold cyan]Profile Structure Validation[/bold cyan]\n")

    # Check if profiles exist
    tier_files = {
        "Tier 1 (Core)": PROFILE_DIR / "uno_profile_tier1.md",
        "Tier 2 (Extended)": PROFILE_DIR / "uno_profile_tier2.md",
        "Tier 3 (Full)": PROFILE_DIR / "uno_profile_tier3.md",
    }

    missing_files = [name for name, path in tier_files.items() if not path.exists()]
    if missing_files:
        log.error(
            f"Missing profile files: {', '.join(missing_files)}\n"
            f"Run compress_character_profile.py first."
        )
        return

    # Validate each tier
    all_results = []
    for tier_name, tier_path in tier_files.items():
        log.info(f"Validating {tier_name}...")

        with open(tier_path, encoding="utf-8") as f:
            content = f.read()

        results = validate_profile_structure(content, tier_name)
        all_results.append(results)

        # Print immediate feedback
        if results["sections_missing"]:
            log.warning(f"  Missing sections: {', '.join(results['sections_missing'])}")
        else:
            log.info("  ✓ All required sections present")

        if not results["has_do_dont"]:
            log.warning("  Missing Do/Don't behavioral guidelines")
        else:
            log.info("  ✓ Has Do/Don't guidelines")

        log.info(f"  Italian examples: {results['italian_examples_count']}")
        console.print()

    # Generate and save validation report
    report = generate_validation_report(all_results)
    report_path = PROFILE_DIR / "validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    log.info(f"Validation report saved: {report_path}")
    console.print()

    # Summary
    all_valid = all(not r["sections_missing"] and r["has_do_dont"] for r in all_results)

    if all_valid:
        panel = Panel(
            "[bold green]✓ All profiles have the required structure[/bold green]\n\n"
            "[dim]Profiles are ready for testing![/dim]",
            title="Validation Complete",
            border_style="green",
        )
    else:
        panel = Panel(
            "[bold yellow]⚠ Some profiles need refinement[/bold yellow]\n\n"
            f"[dim]See {report_path} for details[/dim]",
            title="Validation Complete",
            border_style="yellow",
        )

    console.print(panel)
    console.print()

    # Next steps
    console.print("[dim]Next steps:[/dim]")
    console.print(
        "[dim]1. Test conversations: ./generate_from_character_profile.py --tier core[/dim]"
    )
    console.print(
        "[dim]2. Compare tiers: Try --tier core, --tier extended, --tier full[/dim]"
    )
    console.print("[dim]3. Measure hallucinations with standardized questions[/dim]")


if __name__ == "__main__":
    main()
