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
from typing import Any

import torch
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from transformers import pipeline


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
BASE_DIR = Path(__file__).parent


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

    Args:
        profile_path: Path to character profile markdown file

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
    instructions = f"""You are {character_name}, an AI companion housed in the Ducklair Tower. You must stay completely in character at all times.

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
        self.annotation: str | None = None

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
        filename = f"conversation_{self.character_name.lower()}_{timestamp_str}.json"
        output_path = output_dir / filename

        # Prepare data
        metadata = {
            "character": self.character_name,
            "profile_path": self.profile_path,
            "model": self.model_name,
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
        "[dim]Add notes about this conversation (quality, hallucinations, issues, etc.)[/dim]"
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


def chat_loop(
    backend: dict[str, Any],
    model_name: str,
    character_name: str,
    system_instructions: str,
    history: ConversationHistory,
) -> None:
    """Run the interactive chat loop.

    Args:
        backend: Backend configuration dict (Gemini or HuggingFace)
        model_name: Name of the model to use
        character_name: Name of the character
        system_instructions: System instructions for the LLM
        history: Conversation history manager
    """
    # Display welcome panel
    backend_type = backend["type"]
    welcome_panel = Panel(
        f"[bold cyan]Character Chat: {character_name}[/bold cyan]\n"
        f"Profile: {history.profile_path}\n"
        f"Backend: {backend_type}\n"
        f"Model: {history.model_name}\n\n"
        f"[dim]Press Ctrl+C to exit and save conversation[/dim]",
        title="🤖 Character Chat",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    # Initialize conversation history for the API (format depends on backend)
    if backend_type == "gemini":
        conversation_gemini: list[Content] = []
        conversation_hf: list[dict[str, str]] = []
    else:  # huggingface
        conversation_gemini = []
        conversation_hf = []

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

            # Add to conversation history (backend-specific format)
            if backend_type == "gemini":
                conversation_gemini.append(
                    Content(
                        role="user",
                        parts=[Part.from_text(text=user_input)],
                    ),
                )
            else:  # huggingface
                conversation_hf.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

            # Add to persistent history
            history.add_user_message(user_input)

            # Get response from LLM (backend-specific)
            try:
                if backend_type == "gemini":
                    assistant_response = generate_response_gemini(
                        backend,
                        conversation_gemini,
                        system_instructions,
                        model_name,
                    )
                    # Add to conversation history
                    conversation_gemini.append(
                        Content(
                            role="model",
                            parts=[Part.from_text(text=assistant_response)],
                        ),
                    )
                else:  # huggingface
                    assistant_response = generate_response_huggingface(
                        backend,
                        conversation_hf,
                        system_instructions,
                    )
                    # Add to conversation history
                    conversation_hf.append(
                        {
                            "role": "assistant",
                            "content": assistant_response,
                        }
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
                if backend_type == "gemini":
                    conversation_gemini.pop()
                else:
                    conversation_hf.pop()

    except KeyboardInterrupt:
        # Ctrl+C - graceful exit
        console.print("\n")

    # Collect annotation after chat
    if history.messages:
        collect_annotation(history)


# Backend abstraction
def initialize_gemini_backend() -> dict[str, Any]:
    """Initialize Gemini backend with Google GenAI client.

    Returns:
        Backend configuration dict with client and metadata
    """
    client = genai.Client()
    return {
        "type": "gemini",
        "client": client,
    }


def initialize_huggingface_backend(model_name: str) -> dict[str, Any]:
    """Initialize Hugging Face backend with transformers pipeline.

    Args:
        model_name: HF model name or local path

    Returns:
        Backend configuration dict with pipeline and metadata
    """
    log.info(f"Loading Hugging Face model: {model_name}")

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available, using CPU (will be slower)")

    # Load pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    log.info(f"Model loaded successfully on {device}")

    return {
        "type": "huggingface",
        "pipeline": pipe,
    }


def generate_response_gemini(
    backend: dict[str, Any],
    conversation: list[Content],
    system_instructions: str,
    model_name: str,
) -> str:
    """Generate response using Gemini backend.

    Args:
        backend: Backend configuration dict
        conversation: List of Content objects (Gemini format)
        system_instructions: System instructions for the model
        model_name: Model name to use

    Returns:
        Generated response text
    """
    client = backend["client"]
    response = client.models.generate_content(
        model=model_name,
        contents=conversation,
        config=GenerateContentConfig(
            system_instruction=system_instructions,
            temperature=1.0,
            top_p=0.95,
        ),
    )
    return response.text or ""


def generate_response_huggingface(
    backend: dict[str, Any],
    conversation: list[dict[str, str]],
    system_instructions: str,
) -> str:
    """Generate response using Hugging Face backend.

    Args:
        backend: Backend configuration dict
        conversation: List of message dicts (HF format)
        system_instructions: System instructions for the model

    Returns:
        Generated response text
    """
    pipe = backend["pipeline"]

    # Prepend system message if not already present
    messages = conversation.copy()
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system_instructions})

    # Generate response
    output = pipe(messages, max_new_tokens=1024)

    # Extract assistant's response (last message in generated text)
    generated = output[0]["generated_text"]
    assistant_message = generated[-1]["content"]

    return assistant_message


def main() -> None:
    """Main entry point for the chat script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Chat with a character based on their profile"
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
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gemini", "huggingface"],
        default="gemini",
        help="Model backend to use (default: gemini)",
    )
    args = parser.parse_args()

    # Resolve paths
    conversations_dir = BASE_DIR / CONVERSATIONS_DIR
    profile_path = BASE_DIR / args.profile

    try:
        # Initialize backend based on CLI argument
        if args.backend == "gemini":
            backend = initialize_gemini_backend()
        else:  # huggingface
            backend = initialize_huggingface_backend(args.model)

        # Load profile
        character_name, profile_content = load_profile(profile_path)
        profile_ref = str(args.profile)

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

        # Run interactive chat loop
        chat_loop(backend, args.model, character_name, system_instructions, history)

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
