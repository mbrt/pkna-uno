from glob import glob
from typing import Any
import hashlib
import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from pydantic import BaseModel, Field
from rich.logging import RichHandler
import PIL.Image
import PIL.ImageFile


load_dotenv()

# Flags
images_pattern = 'input/pkna/pkna-0/*.jp*g'
model_name = 'gemini-2.0-flash'
#model_name = 'gemini-2.5-pro-exp-03-25'
max_retries = 3


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_time=True, show_path=False)]
)
log = logging.getLogger('rich')


class Character(BaseModel):
    name: str
    description: str = Field(description="Extensive description of the character appearance and personality")

class Response(BaseModel):
    characters: list[Character]

images = [
    PIL.Image.open(p)
    for p in glob(images_pattern)
]

if len(images) == 0:
    log.error(f"No images found in {images_pattern}")
    raise ValueError(f"No images found in {images_pattern}")

log.info(f"Loaded {len(images)} images from {images_pattern}")

# Load prompts
with open('prompt-characters.md', 'r') as f:
    prompt = f.read().strip()
prompt_version = hashlib.sha1(prompt.encode()).hexdigest()
with open('../export/characters.json', 'r') as f:
    characters = f.read().strip()

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

retries = 0

while True:
    out_file = os.path.join(root_dir, 'characters-id.json')

    try:
        resp = process_batch(images)

    except OverloadException:
        log.warning("Overload exception, retrying...")
        retries += 1
        if retries >= max_retries:
            log.error("Max retries reached, exiting...")
            raise ValueError("Max retries reached")
        time.sleep(60)
        continue

    # Write the response to a file
    with open(out_file, 'w') as out:
        out.write(json.dumps(resp, indent=2, ensure_ascii=False))

    log.info(f"Response written to file: {out_file}")
    break
