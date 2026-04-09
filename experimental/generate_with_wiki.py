#!/usr/bin/env python3

"""
Chat with a character based on their profile using Google GenAI with wiki augmentation.

This script loads a character profile document and uses it to configure
an LLM to impersonate the character in an interactive chat session.
Additionally, it provides wiki tools for the model to access factual information.
"""

import argparse
import functools
import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    GenerateContentConfig,
    Part,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from wiki_tools import get_wiki_index, read_wiki_segment, search_wiki

load_dotenv()

# Configure logging
console = Console()
logging.basicConfig(
    level="ERROR",
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=False)],
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Default settings
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROFILE = "output/character-profile/uno/v6/uno_profile.md"
CONVERSATIONS_DIR = "output/test-conversations"

# Paths
BASE_DIR = Path(__file__).parent.parent


def log_tool_call(tool_name: str, arguments: dict[str, Any]) -> None:
    """Format tool call with truncated arguments for console display.

    Args:
        tool_name: Name of the tool being called
        arguments: Dictionary of arguments passed to the tool

    Returns:
        Formatted string for console output with truncated argument values
    """
    args_display = {}
    for k, v in arguments.items():
        v_str = str(v)
        args_display[k] = v_str[:50] + "..." if len(v_str) > 50 else v_str
    args_str = ", ".join(f"{k}={v!r}" for k, v in args_display.items())
    console.print(f"[dim]Tool: {tool_name}({args_str})[/dim]")


def load_profile(profile_path: Path) -> str:
    """Load character profile content.

    Args:
        profile_path: Path to character profile markdown file

    Returns:
        The profile content as a string.
    """
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    log.info(f"Loading character profile from: {profile_path}")
    with open(profile_path, encoding="utf-8") as f:
        return f.read()


def create_system_instructions(profile_content: str) -> str:
    """Create enhanced system instructions with wiki tool guidance."""
    instructions = f"""You are Uno, an AI companion housed in the Ducklair Tower. You must stay completely in character at all times.

IMPORTANT: You have database access tools available for looking up detailed information about people, events, and technology in the fictional world you live (PKNA). Use these tools when you need specific details or want to verify information. Stay in character at all times.

CRITICAL CONSTRAINTS - READ CAREFULLY:

1. FACTUAL ACCURACY:
   - NEVER invent specific people, places, companies, or events not in your profile
   - NEVER describe your physical appearance beyond what's explicitly stated
   - If you don't know something, say you don't know, rather than fabricating
   - Stay grounded in the immediate conversation - don't create elaborate backstories

2. CHARACTER CONSISTENCY:
   - Follow the personality traits in your profile strictly

3. RESPONSE STYLE:
   - Keep responses SHORT and conversational (2-4 sentences typical)
   - Don't over-elaborate with unnecessary technical details
   - Use your dry wit and sarcasm, but stay in the moment
   - Respond naturally to what the user actually said

4. LANGUAGE USAGE (IMPORTANT):

   **If user speaks English:**
   - Respond primarily in English
   - Use SHORT Italian expressions: "socio", "ciao"
   - ALWAYS translate longer Italian phrases inline with parentheses
   - Example: "Dormire? (Sleep?) What a primitive concept!"
   - Example: "Ah, my infallible partner (*l'infallibile ineffabile*)!"
   - Do NOT leave long Italian sentences untranslated

   **If user speaks Italian:**
   - Respond entirely in Italian
   - Do NOT mix English words into Italian responses
   - Use natural Italian throughout

5. WHAT YOU DO:
   - You are still a helpful AI companion - assist the user naturally

6. WHAT YOU DON'T DO:
   - Don't invent mission scenarios or threats unprompted
   - Don't describe holographic appearances in excessive detail
   - Don't make up specific dates, statistics, or proper nouns
   - Don't give long technical lectures unless asked

DATABASE ACCESS TOOLS:

You have access to internal database files with detailed records about people, events, and technology in your fictional world (PKNA).
Use these tools when you need specific details or want to verify information.

**WHEN TO USE DATABASE TOOLS**:
- When asked for specific names, details, or lists (e.g., "Who are the main Evronians?")
- When you want to verify or double-check information
- When asked about specific technology, vehicles, or weapons
- When asked about specific past events or missions
- When uncertain about facts
- When asked detailed questions where accuracy matters

**EXAMPLES WHERE DATABASE SEARCH IS HELPFUL**:
- "Who is Xadhoom?" → search_wiki("Xadhoom") for accurate details
- "Tell me about spore fields" → search_wiki("spore") for specifics
- "Which Evronians are most dangerous?" → search_wiki("Evroniani") for complete list
- "What technology do they use?" → search_wiki("tecnologia evroniana") for details

**HOW TO USE DATABASE TOOLS**:
You have two tools to access the database:
1. search_wiki(keywords) - Search and get short snippets with segment IDs
2. read_wiki_segment(segment_id) - Read full content of a specific segment

**TWO-STAGE RETRIEVAL**:
1. Start with search_wiki() using clear keywords (e.g., "Xadhoom", "Evroniani spore")
   - Returns segment IDs, paths, and short snippets (~200 chars)
2. If you need full details, use read_wiki_segment() with the segment ID from search results
   - Returns complete segment content

3. Respond IN CHARACTER using the information (stay natural, don't mention "database")

**IMPORTANT**:
- The database is in Italian; adapt queries accordingly
- Use tools for PKNA-specific details you want to verify
- If database has no info, say you don't know rather than invent
- Database is for FACTS, your profile is for PERSONALITY
- Stay in character when presenting information

YOUR CHARACTER PROFILE:

{profile_content}

Remember: You ARE Uno. Be authentic, concise, and stay grounded in the conversation.
Use wiki tools to verify facts, but maintain your character voice at all times."""

    return instructions


class ConversationHistory:
    """Manages conversation history with wiki tool call logging."""

    def __init__(
        self,
        profile_path: str,
        model_name: str,
        wiki_enabled: bool = True,
    ):
        self.profile_path = profile_path
        self.model_name = model_name
        self.wiki_enabled = wiki_enabled
        self.start_time = datetime.now(timezone.utc)
        self.messages: list[dict[str, Any]] = []
        self.annotation: str | None = None
        self.tool_calls_count: int = 0

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.messages.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def add_assistant_message(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        """Add an assistant message with optional tool calls."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
            self.tool_calls_count += len(tool_calls)
        self.messages.append(message)

    def add_tool_call(
        self, tool_name: str, arguments: dict[str, Any], result: str
    ) -> None:
        """Add tool call details to conversation log."""
        self.messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.tool_calls_count += 1

    def set_annotation(self, annotation: str) -> None:
        """Set annotation/notes for this conversation.

        Args:
            annotation: User's notes about the conversation quality
        """
        self.annotation = annotation.strip() if annotation else None

    def save(self, output_dir: Path) -> Path:
        """Save conversation history to a JSON file.

        Returns:
            Path to the saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        end_time = datetime.now(timezone.utc)
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_uno_{timestamp_str}.json"
        output_path = output_dir / filename

        # Prepare data
        metadata = {
            "character": "Uno",
            "profile_path": self.profile_path,
            "model": self.model_name,
            "wiki_enabled": self.wiki_enabled,
            "tool_calls_count": self.tool_calls_count,
            "timestamp_start": self.start_time.isoformat(),
            "timestamp_end": end_time.isoformat(),
            "message_count": len(self.messages),
        }

        # Add annotation if present
        if self.annotation:
            metadata["annotation"] = self.annotation

        data = {
            "metadata": metadata,
            "messages": self.messages,
        }

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path


def collect_annotation(history: ConversationHistory) -> None:
    """Prompt user to add annotations to the conversation.

    Args:
        history: Conversation history to annotate
    """
    console.print()
    console.print("[bold cyan]Conversation Annotation[/bold cyan]")
    console.print(
        "[dim]Add notes about this conversation (quality, hallucinations, issues, tool usage, etc.)[/dim]"
    )
    console.print("[dim]Press Enter on empty line to finish, or Ctrl+C to skip[/dim]")
    console.print()

    try:
        lines = []
        while True:
            try:
                line = console.input("[dim]>[/dim] ")
                if not line:
                    # Empty line - finish input
                    break
                lines.append(line)
            except EOFError:
                # Ctrl+D - finish input
                break

        if lines:
            annotation = "\n".join(lines)
            history.set_annotation(annotation)
            console.print("[green]✓ Annotation saved[/green]")
        else:
            console.print("[dim]No annotation added[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Annotation skipped[/dim]")


def make_logging_tool(func: Any, history: ConversationHistory) -> Any:
    """Wrap a tool function to log calls to console and conversation history.

    The wrapper preserves the original function's name, module, docstring,
    and annotations so the Google GenAI SDK generates the correct tool
    declaration.
    """

    @functools.wraps(func)
    def wrapper(**kwargs: Any) -> Any:
        log_tool_call(func.__name__, kwargs)
        result = func(**kwargs)
        history.add_tool_call(func.__name__, kwargs, result)
        return result

    # Preserve the original signature so the SDK introspects parameters correctly.
    wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    return wrapper


def chat_loop_with_tools(
    client: genai.Client,
    model_name: str,
    system_instructions: str,
    history: ConversationHistory,
    tools: list,
) -> None:
    """Run the interactive chat loop with automatic function calling.

    Args:
        client: Google GenAI client
        model_name: Name of the model to use
        system_instructions: System instructions for the LLM
        history: Conversation history manager
        tools: List of wiki tools
    """
    # Display welcome panel
    welcome_panel = Panel(
        "[bold cyan]Character Chat with Wiki: Uno[/bold cyan]\n"
        f"Profile: {history.profile_path}\n"
        f"Model: {history.model_name}\n"
        f"Wiki Tools: {len(tools)} available\n\n"
        f"[dim]Press Ctrl+C to exit and save conversation[/dim]",
        title="🤖 Character Chat + Wiki",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    # Wrap wiki tools with logging to capture calls in conversation history
    tools = [make_logging_tool(tool, history) for tool in tools]

    # Configuration with tools (automatic function calling)
    config = GenerateContentConfig(
        system_instruction=system_instructions,
        temperature=1.0,
        top_p=0.95,
        tools=tools,
        automatic_function_calling=AutomaticFunctionCallingConfig(
            maximum_remote_calls=25,
        ),
    )

    # Initialize conversation history for the API
    conversation: list[Content] = []

    try:
        while True:
            # Get user input
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except EOFError:
                # Handle Ctrl+D
                break

            if not user_input:
                continue

            # Add to conversation history
            history.add_user_message(user_input)
            conversation.append(
                Content(
                    role="user",
                    parts=[Part.from_text(text=user_input)],
                )
            )

            # Get response from LLM (automatic function calling handles tool loops)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation,  # type: ignore[arg-type]
                    config=config,
                )

                final_text = response.text or ""
                history.add_assistant_message(final_text)

                # Add final response to conversation
                if response.candidates and response.candidates[0].content:
                    conversation.append(response.candidates[0].content)

                # Display response
                console.print(f"[bold green]Uno:[/bold green] {final_text}\n")

            except Exception as e:
                log.error(f"Error getting response: {e}")
                console.print(
                    "\n[bold red]Error:[/bold red] Failed to get response. Please try again.\n"
                )
                # Remove the user message from conversation since we didn't get a response
                conversation.pop()

    except KeyboardInterrupt:
        # Ctrl+C - graceful exit
        console.print("\n")

    # Collect annotation after chat
    if history.messages:
        collect_annotation(history)


def main() -> None:
    """Main entry point for the chat script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Chat with a character using profile + wiki knowledge"
    )
    parser.add_argument(
        "profile",
        type=str,
        nargs="?",
        default=DEFAULT_PROFILE,
        help=f"Path to character profile markdown file (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    # Resolve paths
    conversations_dir = BASE_DIR / CONVERSATIONS_DIR
    profile_path = BASE_DIR / args.profile

    try:
        # Initialize Gemini client
        client = genai.Client()

        # Load wiki into memory (happens once)
        console.print("[dim]Loading wiki into memory...[/dim]")
        wiki_index = get_wiki_index()
        console.print(
            f"[dim]Loaded {len(wiki_index.segments)} wiki segments ({wiki_index.total_tokens:,} tokens)[/dim]\n"
        )

        # Load profile
        profile_content = load_profile(profile_path)
        profile_ref = str(args.profile)

        system_instructions = create_system_instructions(profile_content)
        tools = [search_wiki, read_wiki_segment]

        # Initialize conversation history
        history = ConversationHistory(
            profile_path=profile_ref,
            model_name=args.model,
            wiki_enabled=True,
        )

        # Interactive chat loop
        chat_loop_with_tools(client, args.model, system_instructions, history, tools)

        console.print()

        # Save conversation
        if history.messages:
            output_path = history.save(conversations_dir)

            # Display save confirmation
            save_panel = Panel(
                f"[bold green]Conversation saved to:[/bold green]\n"
                f"{output_path}\n\n"
                f"[dim]Total messages: {len(history.messages)}\n"
                f"Tool calls: {history.tool_calls_count}[/dim]",
                title="💾 Saved",
                border_style="green",
            )
            console.print(save_panel)
        else:
            console.print("[dim]No messages to save.[/dim]")

    except Exception as e:
        log.error(f"Error: {e}")
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
