from glob import glob
from typing import Any
import hashlib
import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel, Field
from rich.logging import RichHandler
import PIL.Image
import PIL.ImageFile


load_dotenv()

# Flags
images_pattern = '../export/pkna-39/*.jp*g'
model_name = 'gemini-2.0-flash'
#model_name = 'gemini-2.5-pro-exp-03-25'
max_batch_size = 10
min_default_batch_size = 8
min_batch_size = 6
max_retries = 3


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_time=True, show_path=False)]
)
log = logging.getLogger('rich')



class Caption(BaseModel):
    text: str

class Bubble(BaseModel):
    text: str
    character: str

class Frame(BaseModel):
    page: int | None
    frame: int
    captions: list[Caption]
    bubbles: list[Bubble]
    description: str = Field(description="One sentence description of the frame")


class Scene(BaseModel):
    frames: list[Frame]
    summary: str = Field(description="Brief summary of the scene")


class Response(BaseModel):
    scenes: list[Scene]


class ImageLoader:

    def __init__(self, pattern: str):
        self.paths = glob(pattern)
        self.paths.sort()
        self.images = [PIL.Image.open(p) for p in self.paths]
        self.curr_index = 0
        self.num_batch = 0
        self.curr_batch_size = self._compute_batch_size(len(self.images))

    def __len__(self):
        return len(self.images)

    def get_batch(self) -> list[PIL.ImageFile.ImageFile]:
        if self.curr_index >= len(self.images):
            return []
        batch = self.images[self.curr_index:self.curr_index + self.curr_batch_size]
        # Do not increment the index, wait for advance_batch
        return batch

    def get_num_batch(self) -> int:
        return self.num_batch

    def advance_batch(self, num_pages: int | None = None) -> None:
        if num_pages is None:
            num_pages = self.curr_batch_size

        self.curr_index += num_pages
        self.num_batch += 1
        self.curr_batch_size = self._compute_batch_size(len(self.images) - self.curr_index)

    def decrease_batch_size(self) -> None:
        if self.curr_batch_size <= min_batch_size:
            raise ValueError(f"Batch size is already at minimum: {self.curr_batch_size}")
        self.curr_batch_size -= 1

    def _compute_batch_size(self, num_images: int) -> int:
        candidate = max_batch_size

        for i in range(max_batch_size, min_default_batch_size - 1, -1):
            if num_images <= i:
                return i
            if num_images % i == 0:
                return i
            # Otherwise, find the largest batch size that makes the last chunk
            # as big as possible.
            if num_images % i > num_images % candidate:
                candidate = i

        return candidate


# Load the images
loader = ImageLoader(images_pattern)

if len(loader) == 0:
    log.error(f"No images found in {images_pattern}")
    raise ValueError(f"No images found in {images_pattern}")

log.info(f"Loaded {len(loader)} images from {images_pattern}")

# Load prompts
with open('prompt.md', 'r') as f:
    prompt = f.read().strip()
with open('../export/characters.json', 'r') as f:
    characters = f.read().strip()

prompt_version = hashlib.sha1(prompt.encode()).hexdigest()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class OverloadException(Exception):
    pass

class BatchTooLargeException(Exception):
    pass


def process_batch(batch: list[PIL.ImageFile.ImageFile]) -> Any:

    try:
        response = client.models.generate_content(
            model=model_name,
            config={
                'response_mime_type': 'application/json',
                'response_schema': Response,
            },
            contents=[prompt, characters] + batch, # type: ignore
        )
    except ServerError as e:
        log.error(f"Server error: {e}")
        raise OverloadException()

    if response.text is None:
        log.error(f"Empty response received: {response}")
        raise BatchTooLargeException()

    # Parse JSON and add metadata
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON: {e}")
        raise BatchTooLargeException()

    parsed["metadata"] = {
        "model_name": model_name,
        "num_pages": len(batch),
        "prompt_version": prompt_version,
    }

    return parsed


root_dir = os.path.join(
    os.path.dirname(images_pattern),
    model_name
)
os.makedirs(root_dir, exist_ok=True)

found_pages = set()
retries = 0


while True:
    batch = loader.get_batch()
    if not batch:
        log.info("All images processed, exiting...")
        break

    log.info(f"Processing batch of {len(batch)} images...")
    out_file = os.path.join(root_dir, f'out-{loader.get_num_batch()}.part.json')

    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            resp = json.loads(f.read())
        num_pages = resp["metadata"]["num_pages"]
        log.info(f"Batch {loader.get_num_batch()} ({num_pages} pages) already processed, skipping...")
        loader.advance_batch(num_pages)
        continue

    try:
        resp = process_batch(batch)

    except OverloadException:
        log.warning("Overload exception, retrying...")
        retries += 1
        if retries >= max_retries:
            log.error("Max retries reached, exiting...")
            raise ValueError("Max retries reached")
        time.sleep(60)
        continue

    except BatchTooLargeException:
        log.warning("Batch too large, decreasing size...")
        loader.decrease_batch_size()
        continue

    # Write the response to a file
    with open(out_file, 'w') as out:
        out.write(json.dumps(resp, indent=2, ensure_ascii=False))

    # Update the found pages
    found_pages.update((
        f["page"]
        for s in resp["scenes"]
        for f in s["frames"]
        if f["page"] is not None
    ))

    log.info(f"Response written to file: {out_file}")
    loader.advance_batch()
    retries = 0


# Validate if all pages are present in the output.
if len(found_pages) > 0:
    max_page = max(found_pages)
    min_page = min(found_pages)
    expected_pages = set(range(min_page, max_page + 1))
    if expected_pages != found_pages:
        log.error(f"Missing pages: {expected_pages - found_pages}")
        raise ValueError(f"Missing pages: {expected_pages - found_pages}")

    log.info(f"All pages found: {len(found_pages)}")
