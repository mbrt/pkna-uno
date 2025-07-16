# This is just a test to check that the exporter can work with a MIPROv2 teleprompter.

from typing import Literal
import os
from dataclasses import dataclass
from typing import overload
from itertools import zip_longest


import dspy
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class UnoDetection(dspy.Signature):
    """Detect the presence of the character 'Uno' in a comic book page.

    Description of the character: has a duck-like appearance, inside a sphere that appears to be made of a bright green gelatinous substance, with small bubbles. It has a short, rounded beak, large, black eyes without defined pupils."""

    page: dspy.Image = dspy.InputField(desc="Comic book page image")
    uno_present: bool = dspy.OutputField(
        desc="Is the character 'Uno' present in the page?"
    )


Character = Literal['uno', 'pk', 'other']


class ExtractedLine(BaseModel):
    """A line of dialogue extracted from a comic book page."""
    character: Character = Field(
        description="The character who said the line."
    )
    text: str = Field(
        description="The text of the dialogue line."
    )


class DialogueExtraction(dspy.Signature):
    """Extract dialogues from a comic book page.

    The dialogues are expected to be in the form of text bubbles, typically found in comic books."""

    page: dspy.Image = dspy.InputField(desc="Comic book page image")
    dialogue: list[ExtractedLine] = dspy.OutputField(
        desc="List of dialogues extracted from the page, each with a character name and text"
    )


class ComicBookPage(BaseModel):
    """A class representing a comic book page."""
    uno_present: bool = False
    dialogue: list[ExtractedLine] = Field(default_factory=list)


class ComicBookExtractor(dspy.Module):

    def __init__(self):
        self.detect_uno = dspy.ChainOfThought(UnoDetection)
        self.extractor = dspy.ChainOfThought(DialogueExtraction)
        self.normalize = dspy.Predict(
            dspy.make_signature(
                signature='text -> normalized',
                instructions="Normalize text by using normal caps instead of all caps, remove line-break hyphens, and accented letters instead of apostrophes when appropriate.",
            )
        )

    def forward(self, img: dspy.Image) -> ComicBookPage:
        if not self.detect_uno(page=img).uno_present:
            return ComicBookPage(uno_present=False)

        extracted = self.extractor(page=img).dialogue
        normalized = [
            self.normalize(text=line.text).normalized
            for line in extracted
        ]
        return ComicBookPage(
            uno_present=True,
            dialogue=[
                ExtractedLine(character=line.character, text=normalized_text)
                for line, normalized_text in zip(extracted, normalized)
            ],
        )
        


load_dotenv()

lm = dspy.LM(
    model='vertex_ai/gemini-2.5-flash',
    vertex_credentials=os.getenv('VERTEX_AI_CREDS'),
    temperature=1.0,
    max_tokens=65535,
)
dspy.configure(lm=lm)

extractor = ComicBookExtractor()

examples = [
    dspy.Example(
        img=dspy.Image.from_file('../input/pkna/pkna-0/pkna-0-029.jpg'),
        output=ComicBookPage(
            uno_present=False,
            dialogue=[],
        )
    ).with_inputs('img'),
    dspy.Example(
        img=dspy.Image.from_file('../input/pkna/pkna-0/pkna-0-030.jpg'),
        output=ComicBookPage(
            uno_present=True,
            dialogue=[],
        )
    ).with_inputs('img'),
]


@dataclass
class Score:
    value: float = 0.0
    max: float = 0.0

    @property
    def normalized(self) -> float:
        if self.max == 0:
            return 0
        return self.value / self.max

    @overload
    def add(self, score: bool, weight: float = 1.0) -> 'Score': ...
    @overload
    def add(self, score: float|int, weight: float = 1.0) -> 'Score': ...
    @overload
    def add(self, score: list[bool], weight: float = 1.0) -> 'Score': ...

    def add(self, score, weight: float = 1.0) -> 'Score':
        """Add a score to the current score."""
        if isinstance(score, bool):
            return self._add_bool(score, weight)
        elif isinstance(score, list):
            return self._add_list(score, weight)
        elif isinstance(score, (int, float)):
            return self._add_float(float(score), 1.0, weight)
        else:
            raise TypeError(f"Unsupported score type: {type(score)}")

    def _add_bool(self, score: bool, weight: float = 1.0) -> 'Score':
        if score:
            self.value += weight
        self.max += weight
        return self

    def _add_list(self, score: list[bool], weight: float = 1.0) -> 'Score':
        """Add a list of boolean scores to the current score."""
        for s in score:
            self._add_bool(s, weight / len(score))
        return self

    def _add_float(self, score: float|int, max_score: float, weight: float = 1.0) -> 'Score':
        """Add a normalized score to the current score."""
        if max_score == 0:
            max_score = 1.0
            score = 1.0
        else:
            score /= max_score
            max_score = 1.0

        score *= weight
        max_score *= weight
        self.value += score
        self.max += max_score
        return self


def check_character(a: ExtractedLine | None, b: ExtractedLine | None) -> bool:
    """Check if two characters are the same."""
    if a is None or b is None:
        return False
    return a.character == b.character


def check_text(a: ExtractedLine | None, b: ExtractedLine | None) -> bool:
    """Check if two lines of text are the same."""
    if a is None or b is None:
        return False
    return a.text.lower() == b.text.lower()


def validate_answer(example: dspy.Example, pred: ComicBookPage, trace=None) -> float | bool:
    must_score = Score()
    may_score = Score()

    gold_characters = {
        line.character for line in example.output.dialogue
    }
    have_characters = {
        line.character for line in pred.dialogue
    }

    must_score.add([
        # Same prediction about Uno
        pred.uno_present == example.output.uno_present,
        # Same number of dialogue lines
        len(pred.dialogue) == len(example.output.dialogue),
        # Same characters in the dialogue
        gold_characters == have_characters,
    ])

    if example.output.uno_present:
        # Same text in the dialogue lines, but not the order
        must_score.add([
            any(line.text.lower() == example_line.text.lower()
                for line in pred.dialogue)
            for example_line in example.output.dialogue
        ])

        # Same order of dialogue lines
        may_score.add([
            check_text(pred_line, example_line)
            for pred_line, example_line in zip_longest(pred.dialogue, example.output.dialogue)
        ])
        # Same order of characters in the dialogue
        may_score.add([
            check_character(pred_line, example_line)
            for pred_line, example_line in zip_longest(pred.dialogue, example.output.dialogue)
        ], weight=3)
    else:
        # If Uno is not present, we don't care about the dialogue lines.
        # Just check that the number of lines is zero.
        may_score.add([
            len(pred.dialogue) == 0
        ])

    if trace is None:
        # We're doing optimization or evaluation.
        # Return precise score.
        return (must_score.normalized * 2 + may_score.normalized) / 3.0

    # We're doing bootstrap demonstrations. Only accept if must score is
    # reasonably met.
    return must_score.normalized >= 0.99


train_lm = dspy.LM(
    model='vertex_ai/gemini-2.5-flash',
    vertex_credentials=os.getenv('VERTEX_AI_CREDS'),
    max_tokens=65535,
    temperature=1.0,
)
teleprompter = dspy.MIPROv2(
    metric=validate_answer,
    auto="light",
    prompt_model=train_lm,
    task_model=lm,
    num_threads=8,
    verbose=True,
)
optimized_extractor = teleprompter.compile(
    extractor,
    trainset=examples,
    requires_permission_to_run=False,
)
