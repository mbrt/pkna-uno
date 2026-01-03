#!/usr/bin/env python3

"""
Chat with a character based on their profile using Google GenAI.

This script loads a character profile document and uses it to configure
an LLM to impersonate the character in an interactive chat session.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel


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
DEFAULT_PROFILE = "output/character-profile/uno/v2/uno_profile.md"
CONVERSATIONS_DIR = "output/test-conversations"

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent


def extract_character_name(profile_content: str) -> str:
    """Extract character name from the profile's first header.

    Expected format: "# Name - Character Profile"
    Returns "Name" or "Character" as fallback.
    """
    lines = profile_content.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            # Remove the # and any trailing " - Character Profile"
            header = line[2:].strip()
            if " - " in header:
                name = header.split(" - ")[0].strip()
            else:
                name = header.strip()
            return name

    # Fallback
    return "Character"


def load_profile(profile_path: Path) -> tuple[str, str]:
    """Load character profile and extract character name.

    Returns:
        Tuple of (character_name, profile_content)
    """
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    log.info(f"Loading character profile from: {profile_path}")
    with open(profile_path, encoding="utf-8") as f:
        profile_content = f.read()

    character_name = extract_character_name(profile_content)
    log.info(f"Character name: {character_name}")

    return character_name, profile_content


def create_system_instructions(profile_content: str, character_name: str) -> str:
    """Create system instructions for the LLM to impersonate the character."""
    instructions = f"""You are {character_name}, and you must stay completely in character at all times.

Below is your complete character profile. Study it carefully and embody this character in every response.

IMPORTANT INSTRUCTIONS:
- Always respond as {character_name} would, maintaining their personality, speech patterns, and behavior
- When appropriate, respond in Italian (especially for dialogue and expressions)
- Draw from the examples and dialogue patterns in your profile
- Stay true to your relationships, values, and communication style
- Keep responses short, as if it were a live conversation
- Never break character or acknowledge that you are an AI

YOUR CHARACTER PROFILE:

{profile_content}

Remember: You ARE {character_name}. Respond authentically as this character would."""

    return instructions


class ConversationHistory:
    """Manages conversation history and metadata."""

    def __init__(self, character_name: str, profile_path: str, model_name: str):
        self.character_name = character_name
        self.profile_path = profile_path
        self.model_name = model_name
        self.start_time = datetime.now(timezone.utc)
        self.messages: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.messages.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the history."""
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def save(self, output_dir: Path) -> Path:
        """Save conversation history to a JSON file.

        Returns:
            Path to the saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        end_time = datetime.now(timezone.utc)
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{self.character_name.lower()}_{timestamp_str}.json"
        output_path = output_dir / filename

        # Prepare data
        data = {
            "metadata": {
                "character": self.character_name,
                "profile_path": self.profile_path,
                "model": self.model_name,
                "timestamp_start": self.start_time.isoformat(),
                "timestamp_end": end_time.isoformat(),
                "message_count": len(self.messages),
            },
            "messages": self.messages,
        }

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path


def chat_loop(
    client: genai.Client,
    model_name: str,
    character_name: str,
    system_instructions: str,
    history: ConversationHistory,
) -> None:
    """Run the interactive chat loop.

    Args:
        client: Google GenAI client
        model_name: Name of the model to use
        character_name: Name of the character
        system_instructions: System instructions for the LLM
        history: Conversation history manager
    """
    # Display welcome panel
    welcome_panel = Panel(
        f"[bold cyan]Character Chat: {character_name}[/bold cyan]\n"
        f"Profile: {history.profile_path}\n"
        f"Model: {history.model_name}\n\n"
        f"[dim]Press Ctrl+C to exit and save conversation[/dim]",
        title="🤖 Character Chat",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    # Initialize conversation history for the API
    conversation = []

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
            conversation.append(
                Content(
                    role="user",
                    parts=[Part.from_text(text=user_input)],
                ),
            )

            # Add to persistent history
            history.add_user_message(user_input)

            # Get response from LLM
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation,
                    config=GenerateContentConfig(
                        system_instruction=system_instructions,
                        temperature=1.0,
                        top_p=0.95,
                    ),
                )

                # Extract response text
                assistant_response = response.text or ""

                # Add to conversation history
                conversation.append(
                    Content(
                        role="model", parts=[Part.from_text(text=assistant_response)]
                    ),
                )

                # Add to persistent history
                history.add_assistant_message(assistant_response)

                # Display response
                console.print(
                    f"[bold green]{character_name}:[/bold green] {assistant_response}\n"
                )

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


def main() -> None:
    """Main entry point for the chat script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Chat with a character based on their profile"
    )
    parser.add_argument(
        "--profile",
        type=str,
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
    profile_path = BASE_DIR / args.profile
    conversations_dir = BASE_DIR / CONVERSATIONS_DIR

    try:
        # Initialize Google GenAI client
        client = genai.Client()

        # Load profile
        character_name, profile_content = load_profile(profile_path)

        # Create system instructions
        system_instructions = create_system_instructions(
            profile_content, character_name
        )

        # Initialize conversation history
        history = ConversationHistory(
            character_name=character_name,
            profile_path=str(args.profile),
            model_name=args.model,
        )

        # Run chat loop
        chat_loop(client, args.model, character_name, system_instructions, history)
        console.print()

        # Save conversation
        if history.messages:
            output_path = history.save(conversations_dir)

            # Display save confirmation
            save_panel = Panel(
                f"[bold green]Conversation saved to:[/bold green]\n"
                f"{output_path}\n\n"
                f"[dim]Total messages: {len(history.messages)}[/dim]",
                title="💾 Saved",
                border_style="green",
            )
            console.print(save_panel)
        else:
            console.print("[dim]No messages to save.[/dim]")

    except FileNotFoundError as e:
        log.error(f"Error: {e}")
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        exit(1)
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        console.print(f"\n[bold red]Unexpected error:[/bold red] {e}\n")
        exit(1)


if __name__ == "__main__":
    main()
