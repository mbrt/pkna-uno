#!/usr/bin/env python3

from typing import Literal, Any
import json
import os
import logging

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import dspy
from rich.console import Console
from rich.logging import RichHandler


TRAINING_MODE = "light"
EXAMPLES_PATH = "../output/dataset/reviewed.jsonl"
OPTIMIZED_PATH = "../output/models/dspy-extract-filtered.json"

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


Character = Literal['uno', 'paperinik', 'other']


class ExtractedLine(BaseModel):
    """A line of dialogue extracted from a comic book page."""
    character: Character = Field(
        description="The character who said the line."
    )
    text: str = Field(
        description="The text of the dialogue line."
    )


class CharacterDescription(BaseModel):
    """Description of a character in the comic."""
    name: Character = Field(
        description="The name of the character."
    )
    description: str = Field(
        description="A description and biography of the character."
    )
    model_config = {
        "frozen": True,
    }


class DialogueExtraction(dspy.Signature):
    """Extract dialogues from a comic book page.

    The dialogues are expected to be in the form of text bubbles, typically found in comic books."""

    characters: list[CharacterDescription] = dspy.InputField(
        desc="List of character descriptions we care about."
    )
    page: dspy.Image = dspy.InputField(desc="Comic book page image")
    dialogue: list[ExtractedLine] = dspy.OutputField(
        desc="List of dialogues extracted from the page, each with a character name and text."
    )


class ExtractorModule(dspy.Module):
    """Extract dialogues from a comic book page."""

    def __init__(self, characters: list[CharacterDescription]):
        self.characters = characters
        self.extractor = dspy.ChainOfThought(DialogueExtraction)
        self.normalize = dspy.Predict(
            dspy.make_signature(
                signature='text -> normalized',
                instructions="Normalize text by using normal caps instead of all caps, remove line-break hyphens, and accented letters instead of apostrophes when appropriate.",
            )
        )

    def forward(self, page: dspy.Image) -> dspy.Prediction:
        extracted = self.extractor(
            characters=self.characters, page=page).dialogue
        normalized = [
            self.normalize(text=line.text).normalized
            for line in extracted
        ]
        return dspy.Prediction(
            dialogue=[
                ExtractedLine(character=line.character, text=normalized_text)
                for line, normalized_text in zip(extracted, normalized)
            ],
        )


def make_gemini_llm(name: str) -> dspy.LM:
    return dspy.LM(
        model=f'vertex_ai/{name}',
        vertex_credentials=os.getenv('VERTEX_AI_CREDS'),
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=65535,
    )


def init_llms() -> tuple[dspy.LM, dspy.LM]:
    load_dotenv()
    task_lm = make_gemini_llm('gemini-2.5-flash')
    prompt_lm = make_gemini_llm('gemini-2.5-pro')
    dspy.configure(lm=task_lm, track_usage=True)
    logger.info("LLMs initialized - Task: gemini-2.5-flash, Prompt: gemini-2.5-pro")
    return task_lm, prompt_lm


def load_character(name: Character) -> CharacterDescription:
    """Load a character description from the environment variable."""
    p = os.path.join('../input/bios', f'{name}.md')
    with open(p, 'r', encoding='utf-8') as f:
        description = f.read()
    return CharacterDescription(name=name, description=description)


def validate(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """Validate the prediction."""
    return example.dialogue == pred.dialogue


def optimize(
        examples: list[dspy.Example],
        training_mode: Literal["light", "medium"],
) -> Any:
    task_lm, prompt_lm = init_llms()

    characters = [
        load_character('uno'),
        load_character('paperinik'),
        CharacterDescription(
            name='other',
            description='Any other character in the comic book.'
        ),
    ]
    extractor = ExtractorModule(characters=characters)
    teleprompter = dspy.MIPROv2(
        metric=validate,
        auto=training_mode,
        prompt_model=prompt_lm,
        task_model=task_lm,
        num_threads=8,
        verbose=True,
    )
    compiled_model = teleprompter.compile(
        extractor,
        trainset=examples,
        requires_permission_to_run=False,
    )
    return compiled_model


def classify_character(character: str) -> Character:
    """Classify the character based on the text."""
    match character.lower():
        case 'uno':
            return 'uno'
        case 'pk' | 'paperinik':
            return 'paperinik'
        case _:
            return 'other'


def parse_reviewed(line: str) -> dspy.Example | None:
    """Parse a line from the reviewed dataset."""
    data = json.loads(line)
    dialogue = data['ocr']['dialogue']

    if not dialogue or len(dialogue) == 0:
        return None

    return dspy.Example(
        page=dspy.Image.from_file(data['image']),
        dialogue=[
            ExtractedLine(
                character=classify_character(line['character']),
                text=line['text'],
            )
            for line in dialogue
        ]
    ).with_inputs('page')


def build_dataset(examples_path: str) -> list[dspy.Example]:
    res = []

    with open(examples_path, 'r') as f:
        for line in f:
            example = parse_reviewed(line)
            if example:
                res.append(example)

    logger.info(f"Dataset loaded: {len(res)}")
    return res


def save_model(model: Any, output_path: str) -> None:
    """Save the optimized model to the specified output path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    examples = build_dataset(EXAMPLES_PATH)
    m = optimize(examples, TRAINING_MODE)
    save_model(m, OPTIMIZED_PATH)


if __name__ == "__main__":
    main()
