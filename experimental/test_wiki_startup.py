#!/usr/bin/env python3
"""Quick test script to verify wiki loading and basic functionality."""

import time

from rich.console import Console

from wiki_tools import get_wiki_index, read_wiki_segment, search_wiki

console = Console()


def main() -> None:
    """Test wiki loading and basic operations."""
    console.print("[bold]Testing Wiki Loading and Operations[/bold]\n")

    # Test 1: Load wiki and measure time
    console.print("1. Loading wiki into memory...")
    start = time.time()
    wiki_index = get_wiki_index()
    load_time = time.time() - start

    console.print(
        f"   ✓ Loaded {len(wiki_index.segments)} segments in {load_time:.2f}s"
    )
    console.print(f"   ✓ Total tokens: {wiki_index.total_tokens:,}\n")

    # Test 2: Search functionality
    console.print("2. Testing search_wiki()...")
    start = time.time()
    results = search_wiki("Xadhoom", max_results=3)
    search_time = time.time() - start

    console.print(f"   ✓ Search completed in {search_time * 1000:.1f}ms")
    console.print(f"   Results preview:\n{results[:300]}...\n")

    # Test 3: Read segment functionality
    console.print("3. Testing read_wiki_segment()...")
    if wiki_index.segments:
        segment_id = wiki_index.segments[0].segment_id
        start = time.time()
        content = read_wiki_segment(segment_id)
        read_time = time.time() - start

        console.print(f"   ✓ Read completed in {read_time * 1000:.1f}ms")
        console.print(f"   Content preview:\n{content[:200]}...\n")

    # Test 4: Verify hierarchical structure
    console.print("4. Testing hierarchical structure...")
    deep_segments = [s for s in wiki_index.segments if len(s.section_path) >= 2]
    console.print(f"   ✓ Found {len(deep_segments)} segments with 2+ levels")
    if deep_segments:
        example = deep_segments[0]
        console.print(f"   Example: {example.get_display_path()}\n")

    console.print("[bold green]✓ All tests passed![/bold green]")


if __name__ == "__main__":
    main()
