from glob import glob
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv
import PIL.Image

load_dotenv()

root_dir = '../export/pkna-0'
image_paths = glob(os.path.join(root_dir, '*.jpg'))
image_paths.sort()

images = [
    PIL.Image.open(p)
    for p in image_paths
]
with open('prompt.md', 'r') as f:
    prompt = f.read().strip()
with open('../export/characters.json', 'r') as f:
    characters = f.read().strip()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Split the images in chunks of 10
image_chunks = [images[i:i + 10] for i in range(0, len(images), 10)]

# Generate content for each chunk
for i, image_chunk in enumerate(image_chunks):
    print(f"Processing chunk {i + 1}/{len(image_chunks)}...")

    # Generate content for each chunk
    response = client.models.generate_content(
        model="gemini-2.0-flash",
       contents=[prompt, characters] + image_chunk)  # type: ignore

    out_file = os.path.join(root_dir, f'out-{i}.txt')
    with open(out_file, 'a') as out:
        out.write(response.text)  # type: ignore
        out.write("\n")

    print("Response written to file:", out_file)
