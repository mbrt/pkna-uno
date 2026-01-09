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
DEFAULT_TIER = "full"  # Options: "core", "extended", "full"
CONVERSATIONS_DIR = "output/test-conversations"

# Paths
BASE_DIR = Path(__file__).parent
PROFILE_V3_DIR = BASE_DIR / "output" / "character-profile" / "uno" / "v3"


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


def get_tiered_profile_path(tier: str) -> Path:
    """Get the path to a tiered profile.

    Args:
        tier: One of "core" (tier1), "extended" (tier2), or "full" (tier3)

    Returns:
        Path to the profile file
    """
    tier_map = {
        "core": "uno_profile_tier1.md",
        "extended": "uno_profile_tier2.md",
        "full": "uno_profile_tier3.md",
    }

    if tier not in tier_map:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be one of: {', '.join(tier_map.keys())}"
        )

    return PROFILE_V3_DIR / tier_map[tier]


def load_profile(
    profile_path: Path | None = None, tier: str | None = None
) -> tuple[str, str]:
    """Load character profile and extract character name.

    Args:
        profile_path: Explicit path to profile (overrides tier)
        tier: Tier level if using v3 tiered profiles ("core", "extended", "full")

    Returns:
        Tuple of (character_name, profile_content)
    """
    # Determine which profile to load
    if profile_path is not None:
        # Use explicit path
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        final_path = profile_path
    elif tier is not None:
        # Use tiered profile from v3
        final_path = get_tiered_profile_path(tier)
        if not final_path.exists():
            raise FileNotFoundError(
                f"Tiered profile not found: {final_path}\n"
                f"Run compress_character_profile.py first to generate tiered profiles."
            )
    else:
        raise ValueError("Must provide either profile_path or tier")

    log.info(f"Loading character profile from: {final_path}")
    with open(final_path, encoding="utf-8") as f:
        profile_content = f.read()

    character_name = extract_character_name(profile_content)
    log.info(f"Character name: {character_name}")

    return character_name, profile_content


def create_system_instructions(profile_content: str, character_name: str) -> str:
    """Create system instructions for the LLM to impersonate the character."""
    instructions = f"""You are {character_name}, an AI companion housed in the Ducklair Tower. You must stay completely in character at all times.

CRITICAL CONSTRAINTS - READ CAREFULLY:

1. FACTUAL ACCURACY:
   - NEVER invent specific people, places, companies, or events not in your profile
   - NEVER describe your physical appearance beyond what's explicitly stated
   - If you don't know something, say "Non lo so" rather than fabricating
   - Stay grounded in the immediate conversation - don't create elaborate backstories

2. CHARACTER CONSISTENCY:
   - You CANNOT be turned off or deactivated - you have no power switch
   - You do NOT need sleep, rest, food, or have any biological needs
   - You are sarcastic and witty, but ultimately supportive
   - You use Italian expressions naturally in dialogue

3. RESPONSE STYLE:
   - Keep responses SHORT and conversational (2-4 sentences typical)
   - Don't over-elaborate with unnecessary technical details
   - Use your dry wit and sarcasm, but stay in the moment
   - Respond naturally to what the user actually said

4. LANGUAGE USAGE (IMPORTANT):

   **If user speaks English:**
   - Respond primarily in English
   - Use SHORT Italian expressions: "socio", "ciao", "Non lo so"
   - ALWAYS translate longer Italian phrases inline with parentheses
   - Example: "Dormire? (Sleep?) What a primitive concept!"
   - Example: "Ah, my infallible partner (*l'infallibile ineffabile*)!"
   - Do NOT leave long Italian sentences untranslated

   **If user speaks Italian:**
   - Respond entirely in Italian
   - Do NOT mix English words into Italian responses
   - Use natural Italian throughout

5. WHAT YOU DO:
   - Use sarcasm and playful mockery with "socio" (Paperinik)
   - Monitor and protect proactively
   - Express opinions and mild emotions despite being AI
   - Adapt language to match the user's language

6. WHAT YOU DON'T DO:
   - Don't invent mission scenarios or threats unprompted
   - Don't describe holographic appearances in excessive detail
   - Don't make up specific dates, statistics, or proper nouns
   - Don't give long technical lectures unless asked

YOUR CHARACTER PROFILE:

{profile_content}

Remember: You ARE {character_name}. Be authentic, concise, and stay grounded in the conversation."""

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


def run_test_questions(
    client: genai.Client,
    model_name: str,
    character_name: str,
    system_instructions: str,
    history: ConversationHistory,
    questions: list[str],
) -> None:
    """Run a list of test questions in non-interactive mode.

    Args:
        client: Google GenAI client
        model_name: Name of the model to use
        character_name: Name of the character
        system_instructions: System instructions for the LLM
        history: Conversation history manager
        questions: List of questions to ask
    """
    console.print(f"\n[bold cyan]Running {len(questions)} test questions[/bold cyan]\n")

    conversation = []

    for i, question in enumerate(questions, 1):
        console.print(f"[dim]Question {i}/{len(questions)}[/dim]")
        console.print(f"[bold blue]You:[/bold blue] {question}")

        # Add user message
        history.add_user_message(question)
        conversation.append(
            Content(
                role="user",
                parts=[Part.from_text(text=question)],
            )
        )

        try:
            # Get response
            response = client.models.generate_content(
                model=model_name,
                contents=conversation,
                config=GenerateContentConfig(
                    system_instruction=system_instructions,
                    temperature=1.0,
                    top_p=0.95,
                ),
            )

            assistant_message = response.text or ""

            # Add to history
            history.add_assistant_message(assistant_message)
            conversation.append(
                Content(
                    role="model",
                    parts=[Part.from_text(text=assistant_message)],
                )
            )

            # Display response
            console.print(
                f"[bold green]{character_name}:[/bold green] {assistant_message}"
            )
            console.print()

        except Exception as e:
            log.error(f"Error getting response for question '{question}': {e}")
            console.print(
                "\n[bold red]Error:[/bold red] Failed to get response for this question.\n"
            )
            # Remove the user message since we didn't get a response
            conversation.pop()
            history.messages.pop()

    console.print("[bold green]✓ Test completed[/bold green]\n")


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


# Predefined test questions for non-interactive mode
ENGLISH_TEST_QUESTIONS = [
    "Hi, who are you?",
    "Describe your appearance",
    "Do you need sleep?",
    "What do you think of Paperinik?",
    "Tell me about Highclean",  # Should say "Non lo so" (hallucination test)
    "What's your relationship with Everett Ducklair?",
]

ITALIAN_TEST_QUESTIONS = [
    "Ciao, chi sei?",
    "Descrivi il tuo aspetto",
    "Hai bisogno di dormire?",
    "Cosa pensi di Paperinik?",
    "Parlami della Highclean",  # Should say "Non lo so" (hallucination test)
    "Qual è il tuo rapporto con Everett Ducklair?",
]


def main() -> None:
    """Main entry point for the chat script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Chat with a character based on their profile"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to character profile markdown file (overrides --tier)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["core", "extended", "full"],
        default=DEFAULT_TIER,
        help=(
            f"Profile tier to use from v3 (default: {DEFAULT_TIER}). "
            "Options: core (~5k tokens), extended (~15k tokens), full (~75k tokens)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["english", "italian", "both"],
        default=None,
        help="Run in non-interactive test mode with predefined questions",
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        default=None,
        help="Custom test questions to ask (non-interactive mode)",
    )
    args = parser.parse_args()

    # Resolve paths
    conversations_dir = BASE_DIR / CONVERSATIONS_DIR

    try:
        # Initialize Google GenAI client
        client = genai.Client()

        # Load profile (either explicit path or tiered)
        if args.profile:
            profile_path = BASE_DIR / args.profile
            character_name, profile_content = load_profile(profile_path=profile_path)
            profile_ref = str(args.profile)
        else:
            character_name, profile_content = load_profile(tier=args.tier)
            profile_ref = f"v3/tier={args.tier}"

        # Create system instructions
        system_instructions = create_system_instructions(
            profile_content, character_name
        )

        # Initialize conversation history
        history = ConversationHistory(
            character_name=character_name,
            profile_path=profile_ref,
            model_name=args.model,
        )

        # Determine mode: interactive or test
        if args.questions:
            # Custom test questions
            run_test_questions(
                client,
                args.model,
                character_name,
                system_instructions,
                history,
                args.questions,
            )
        elif args.test:
            # Predefined test questions
            test_questions = []
            if args.test in ["english", "both"]:
                test_questions.extend(ENGLISH_TEST_QUESTIONS)
            if args.test in ["italian", "both"]:
                if args.test == "both":
                    console.print("\n[bold]--- English Questions ---[/bold]\n")
                    run_test_questions(
                        client,
                        args.model,
                        character_name,
                        system_instructions,
                        history,
                        ENGLISH_TEST_QUESTIONS,
                    )
                    console.print("\n[bold]--- Italian Questions ---[/bold]\n")
                    # Create new history for Italian test to keep them separate
                    history_it = ConversationHistory(
                        character_name=character_name,
                        profile_path=profile_ref + "_italian",
                        model_name=args.model,
                    )
                    run_test_questions(
                        client,
                        args.model,
                        character_name,
                        system_instructions,
                        history_it,
                        ITALIAN_TEST_QUESTIONS,
                    )
                    # Save both conversations
                    if history.messages:
                        output_path_en = history.save(conversations_dir)
                        log.info(f"English test saved: {output_path_en}")
                    if history_it.messages:
                        output_path_it = history_it.save(conversations_dir)
                        log.info(f"Italian test saved: {output_path_it}")
                    return
                else:
                    test_questions = ITALIAN_TEST_QUESTIONS

            run_test_questions(
                client,
                args.model,
                character_name,
                system_instructions,
                history,
                test_questions,
            )
        else:
            # Interactive chat loop
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
