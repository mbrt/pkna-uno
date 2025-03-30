from glob import glob
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import PIL.Image


load_dotenv()

# Flags
root_dir = '../export/pkna-1'
model_name = 'gemini-2.5-pro-exp-03-25'
# model_name = 'gemini-2.0-flash'
from_chunk = 0
to_chunk = 256


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
image_paths = glob(os.path.join(root_dir, '*.jpg'))
image_paths.sort()
images = [
    PIL.Image.open(p)
    for p in image_paths
]

print(f"Loaded {len(images)} images from {root_dir}")

# Load prompts
with open('prompt.md', 'r') as f:
    prompt = f.read().strip()
with open('../export/characters.json', 'r') as f:
    characters = f.read().strip()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Split the images in chunks of 10
image_chunks = [images[i:i + 10] for i in range(0, len(images), 10)]

# Generate content for each chunk
for i, image_chunk in enumerate(image_chunks):
    if i < from_chunk or i > to_chunk:
        print(f"Skipping chunk {i + 1}")
        continue

    print(f"Processing chunk {i + 1}/{len(image_chunks)}...")

    # Generate content for each chunk
    response = client.models.generate_content(
        model=model_name,
        config={
            'response_mime_type': 'application/json',
            'response_schema': Response,
        },
        contents=[prompt, characters] + image_chunk, # type: ignore
    )  

    if response.text is None:
        raise ValueError("Response text is None")
    json_out = response.text

    out_file = os.path.join(root_dir, f'out-{i}.part.json')
    with open(out_file, 'a') as out:
        out.write(json_out)

    print("Response written to file:", out_file)
