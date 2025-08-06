#!/usr/bin/env python3

from typing import Literal
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import dspy


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
    return task_lm, prompt_lm


def validate(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """Validate the prediction."""
    return example.dialogue == pred.dialogue


def optimize():
    task_lm, prompt_lm = init_llms()
    characters = [
        CharacterDescription(name='uno', description='A character named Uno.'),
        CharacterDescription(
            name='paperinik', description='A character named Paperinik.'),
        CharacterDescription(
            name='other', description='Other characters in the comic book.'),
    ]
    extractor = ExtractorModule(characters=characters)

    examples = [
        x.with_inputs("page") for x in [
            dspy.Example(
                page=dspy.Image.from_file(
                    '../input/pkna/pkna-0/pkna-0-029.jpg'),
                dialogue=[
                    ExtractedLine(character='uno', text='Hello!'),
                    ExtractedLine(character='paperinik', text='Hi there!'),
                ]
            ),
            dspy.Example(
                page=dspy.Image.from_file(
                    '../input/pkna/pkna-0/pkna-0-030.jpg'),
                dialogue=[
                    ExtractedLine(character='paperinik', text='Well well'),
                ]
            )
        ]
    ]

    teleprompter = dspy.MIPROv2(
        metric=validate,
        auto="light",
        prompt_model=prompt_lm,
        task_model=task_lm,
        num_threads=8,
        verbose=True,
    )
    teleprompter.compile(
        extractor,
        trainset=examples,
        requires_permission_to_run=False,
    )


def main():
    optimize()


if __name__ == "__main__":
    main()
