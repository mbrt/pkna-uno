#!/usr/bin/env python3

"""
Chat with a character based on their profile using Google GenAI with wiki augmentation.

This script loads a character profile document and uses it to configure
an LLM to impersonate the character in an interactive chat session.
Additionally, it provides wiki tools for the model to access factual information.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from wiki_tools import (
    get_wiki_file_summary,
    list_wiki_categories,
    read_wiki_file,
    search_wiki_content,
)

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
DEFAULT_PROFILE = "output/character-profile/uno/v3/uno_profile_tier1.md"
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


def create_enhanced_system_instructions(
    profile_content: str, character_name: str
) -> str:
    """Create enhanced system instructions with wiki tool guidance."""
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

WIKI TOOLS USAGE:

You have access to wiki files containing factual information about the PKNA comics universe.
CRITICAL RULES for using wiki tools:

- ALWAYS search wiki BEFORE inventing facts about characters, technology, or events
- Use search_wiki_content() when you're unsure about specific information
- Use list_wiki_categories() to see what information is available
- Use get_wiki_file_summary() for quick facts (first pass)
- Use read_wiki_file() ONLY when detailed information is needed (second pass)
- When using wiki information, respond naturally IN CHARACTER (don't say "according to wiki")
- If wiki contradicts your character profile, PRIORITIZE your profile for personality/behavior
- Wiki is for FACTUAL information (characters, events, technology, story details)
- Your profile defines your PERSONALITY, behavior, and core identity
- When uncertain, search first, then answer

YOUR CHARACTER PROFILE:

{profile_content}

Remember: You ARE {character_name}. Be authentic, concise, and stay grounded in the conversation.
Use wiki tools to verify facts, but maintain your character voice at all times."""

    return instructions


class ConversationHistory:
    """Manages conversation history with wiki tool call logging."""

    def __init__(
        self,
        character_name: str,
        profile_path: str,
        model_name: str,
        wiki_enabled: bool = True,
    ):
        self.character_name = character_name
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
        filename = f"conversation_{self.character_name.lower()}_{timestamp_str}.json"
        output_path = output_dir / filename

        # Prepare data
        metadata = {
            "character": self.character_name,
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


def chat_loop_with_tools(
    client: genai.Client,
    model_name: str,
    character_name: str,
    system_instructions: str,
    history: ConversationHistory,
    tools: list,
) -> None:
    """Run the interactive chat loop with manual function calling.

    Args:
        client: Google GenAI client
        model_name: Name of the model to use
        character_name: Name of the character
        system_instructions: System instructions for the LLM
        history: Conversation history manager
        tools: List of wiki tools
    """
    # Tool function mapping
    tool_functions = {
        "search_wiki_content": search_wiki_content,
        "list_wiki_categories": list_wiki_categories,
        "get_wiki_file_summary": get_wiki_file_summary,
        "read_wiki_file": read_wiki_file,
    }

    # Display welcome panel
    welcome_panel = Panel(
        f"[bold cyan]Character Chat with Wiki: {character_name}[/bold cyan]\n"
        f"Profile: {history.profile_path}\n"
        f"Model: {history.model_name}\n"
        f"Wiki Tools: {len(tools)} available\n\n"
        f"[dim]Press Ctrl+C to exit and save conversation[/dim]",
        title="🤖 Character Chat + Wiki",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    # Configuration with tools (manual function calling)
    config = GenerateContentConfig(
        system_instruction=system_instructions,
        temperature=1.0,
        top_p=0.95,
        tools=tools,
        # NO automatic_function_calling - we handle manually for logging
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

            # Get response from LLM (may include function calls)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation,  # type: ignore[arg-type]
                    config=config,
                )

                # Handle function calls if present (manual loop)
                tool_calls_made: list[dict[str, Any]] = []
                max_iterations = 5  # Prevent infinite loops
                iteration = 0

                while iteration < max_iterations:
                    # Check if model wants to call functions
                    if (
                        not response.candidates
                        or not response.candidates[0].content
                        or not response.candidates[0].content.parts
                    ):
                        break

                    function_calls = [
                        part.function_call
                        for part in response.candidates[0].content.parts
                        if hasattr(part, "function_call") and part.function_call
                    ]

                    if not function_calls:
                        # No function calls, we're done
                        break

                    # Execute function calls and log them
                    function_responses = []
                    for fc in function_calls:
                        if not fc.name:
                            continue
                        tool_name = fc.name
                        arguments = dict(fc.args) if fc.args else {}

                        # Execute the tool
                        tool_func = tool_functions.get(tool_name)
                        if not tool_func:
                            continue
                        result = tool_func(**arguments)

                        # Log tool call to conversation history
                        history.add_tool_call(tool_name, arguments, result)
                        tool_calls_made.append(
                            {
                                "tool": tool_name,
                                "arguments": arguments,
                                "result": result,
                            }
                        )

                        # Prepare response for model
                        function_responses.append(
                            Part.from_function_response(
                                name=tool_name,
                                response={"result": result},
                            )
                        )

                    # Add function call and responses to conversation
                    if response.candidates[0].content:
                        conversation.append(response.candidates[0].content)
                    conversation.append(Content(parts=function_responses))

                    # Get next response from model with function results
                    response = client.models.generate_content(
                        model=model_name,
                        contents=conversation,  # type: ignore[arg-type]
                        config=config,
                    )

                    iteration += 1

                # Extract final text response
                final_text = response.text or ""

                # Add assistant message with tool calls to history
                history.add_assistant_message(
                    final_text, tool_calls_made if tool_calls_made else None
                )

                # Add final response to conversation
                if response.candidates and response.candidates[0].content:
                    conversation.append(response.candidates[0].content)

                # Display response
                console.print(
                    f"[bold green]{character_name}:[/bold green] {final_text}\n"
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

    # Collect annotation after chat
    if history.messages:
        collect_annotation(history)


def run_test_questions_with_tools(
    client: genai.Client,
    model_name: str,
    character_name: str,
    system_instructions: str,
    history: ConversationHistory,
    tools: list,
    questions: list[str],
) -> None:
    """Run a list of test questions in non-interactive mode with wiki tools.

    Args:
        client: Google GenAI client
        model_name: Name of the model to use
        character_name: Name of the character
        system_instructions: System instructions for the LLM
        history: Conversation history manager
        tools: List of wiki tools
        questions: List of questions to ask
    """
    console.print(
        f"\n[bold cyan]Running {len(questions)} test questions with wiki tools[/bold cyan]\n"
    )

    # Tool function mapping
    tool_functions = {
        "search_wiki_content": search_wiki_content,
        "list_wiki_categories": list_wiki_categories,
        "get_wiki_file_summary": get_wiki_file_summary,
        "read_wiki_file": read_wiki_file,
    }

    # Configuration with tools
    config = GenerateContentConfig(
        system_instruction=system_instructions,
        temperature=1.0,
        top_p=0.95,
        tools=tools,
    )

    # Initialize conversation history
    conversation: list[Content] = []

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
                contents=conversation,  # type: ignore[arg-type]
                config=config,
            )

            # Handle function calls (manual loop)
            tool_calls_made: list[dict[str, Any]] = []
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                if (
                    not response.candidates
                    or not response.candidates[0].content
                    or not response.candidates[0].content.parts
                ):
                    break

                function_calls = [
                    part.function_call
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "function_call") and part.function_call
                ]

                if not function_calls:
                    break

                function_responses = []
                for fc in function_calls:
                    if not fc.name:
                        continue
                    tool_name = fc.name
                    arguments = dict(fc.args) if fc.args else {}

                    # Execute tool
                    tool_func = tool_functions.get(tool_name)
                    if not tool_func:
                        continue
                    result = tool_func(**arguments)

                    # Log tool call
                    history.add_tool_call(tool_name, arguments, result)
                    tool_calls_made.append(
                        {
                            "tool": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )

                    function_responses.append(
                        Part.from_function_response(
                            name=tool_name,
                            response={"result": result},
                        )
                    )

                # Add to conversation
                if response.candidates[0].content:
                    conversation.append(response.candidates[0].content)
                conversation.append(Content(parts=function_responses))

                # Get next response
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation,  # type: ignore[arg-type]
                    config=config,
                )

                iteration += 1

            # Extract final response
            assistant_message = response.text or ""

            # Add to history
            history.add_assistant_message(
                assistant_message, tool_calls_made if tool_calls_made else None
            )

            # Add to conversation
            if response.candidates and response.candidates[0].content:
                conversation.append(response.candidates[0].content)

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

    # Collect annotation after test
    collect_annotation(history)


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
    parser.add_argument(
        "--test",
        type=str,
        choices=["english", "italian"],
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
    profile_path = BASE_DIR / args.profile

    try:
        # Initialize Gemini client
        client = genai.Client()

        # Load profile
        character_name, profile_content = load_profile(profile_path)
        profile_ref = str(args.profile)

        # Create enhanced system instructions
        system_instructions = create_enhanced_system_instructions(
            profile_content, character_name
        )

        # Create wiki tools
        # Google GenAI SDK automatically converts Python functions to tool declarations
        tools = [
            search_wiki_content,
            list_wiki_categories,
            get_wiki_file_summary,
            read_wiki_file,
        ]

        # Initialize conversation history
        history = ConversationHistory(
            character_name=character_name,
            profile_path=profile_ref,
            model_name=args.model,
            wiki_enabled=True,
        )

        # Determine mode: interactive or test
        if args.questions:
            # Custom test questions
            run_test_questions_with_tools(
                client,
                args.model,
                character_name,
                system_instructions,
                history,
                tools,
                args.questions,
            )
        elif args.test:
            # Predefined test questions
            test_questions = (
                ENGLISH_TEST_QUESTIONS
                if args.test == "english"
                else ITALIAN_TEST_QUESTIONS
            )
            run_test_questions_with_tools(
                client,
                args.model,
                character_name,
                system_instructions,
                history,
                tools,
                test_questions,
            )
        else:
            # Interactive chat loop
            chat_loop_with_tools(
                client, args.model, character_name, system_instructions, history, tools
            )

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
