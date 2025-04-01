from glob import glob
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


load_dotenv()

# Flags
root_dir = '../export/pkna-18'
#model_name = 'gemini-2.5-pro-exp-03-25'
model_name = 'gemini-2.0-flash'
# If set to None, it will be computed automatically.
chunk_size: int | None = None

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
    page: int
    frame: int
    captions: list[Caption]
    bubbles: list[Bubble]
    description: str = Field(description="One sentence description of the frame")


class Scene(BaseModel):
    frames: list[Frame]
    summary: str = Field(description="Brief summary of the scene")


class Response(BaseModel):
    scenes: list[Scene]


# Load the images
image_paths = glob(os.path.join(root_dir, '*.jp*g'))
image_paths.sort()
images = [
    PIL.Image.open(p)
    for p in image_paths
]

log.info(f"Loaded {len(images)} images from {root_dir}")

# Load prompts
with open('prompt.md', 'r') as f:
    prompt = f.read().strip()
with open('../export/characters.json', 'r') as f:
    characters = f.read().strip()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Find the best chunk size, based on the number of images
# Two cases:
# 1. There's a size that makes all chunks the same size
# 2. The last chunk is as big as possible
if chunk_size is None:
    chunk_size = 10
    best_size = -1
    for candidate in range(10, 7, -1):
        last_chunk_size = len(images) % candidate
        if last_chunk_size == 0:
            chunk_size = candidate
            break
        if last_chunk_size > best_size:
            best_size = last_chunk_size
            chunk_size = candidate
else:
    best_size = len(images) % chunk_size

# Split the images in chunks of chunk_size
image_chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]

log.info(f"Computed chunk size: {chunk_size}, last chunk: {best_size}, num chunks: {len(image_chunks)}")


# Generate content for each chunk
for i, image_chunk in enumerate(image_chunks):
    out_file = os.path.join(root_dir, f'out-{i}.part.json')

    # Skip if the output file is already there
    if os.path.exists(out_file):
        log.info(f"Output file {out_file} already exists, skipping...")
        continue

    log.info(f"Processing chunk {i + 1}/{len(image_chunks)}...")

    # Generate content for each chunk
    max_retries = 3
    retry_delay = 60  # seconds
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': Response,
                },
                contents=[prompt, characters] + image_chunk, # type: ignore
            )
            break
        except ServerError as e:
            if attempt == max_retries - 1:
                raise  # Re-raise if all retries failed
            log.warning(f"Server error, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

    if response.text is None:
        log.error(f"Empty response received: {response}")
        raise ValueError("Response text is None")

    # Parse JSON and add metadata
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as e:
        with open(out_file+".err", 'w') as out:
            out.write(response.text)
        log.error(f"Failed to parse JSON: {e}")
        raise

    parsed["metadata"] = {
        "model_name": model_name,
        "num_pages": len(image_chunk),
    }
    json_out = json.dumps(parsed, indent=2, ensure_ascii=False)

    with open(out_file, 'a') as out:
        out.write(json_out)

    log.info(f"Response written to file: {out_file}")
