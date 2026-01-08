import time
import os
from vllm import LLM, SamplingParams
from PIL import Image
from docling_core.types.doc.document import DoclingDocument, DocTagsDocument
from pathlib import Path

# Configuration
MODEL_PATH = "ds4sd/SmolDocling-256M-preview"
IMAGE_DIR = "input/pkna/pkna-0"  # Place your page images here
PROMPT_TEXT = "Convert page to Docling."
OUTPUT_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "ocr")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize LLM
llm = LLM(model=MODEL_PATH, limit_mm_per_prompt={"image": 1})

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192)

chat_template = f"""<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance>
Assistant:"""

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

start_time = time.time()
total_tokens = 0

for idx, img_file in enumerate(image_files, 1):
    img_path = os.path.join(IMAGE_DIR, img_file)
    print("Processing image:", img_path)
    image = Image.open(img_path).convert("RGB")

    llm_input = {"prompt": chat_template, "multi_modal_data": {"image": image}}
    output = llm.generate([llm_input], sampling_params=sampling_params)[0]  # type: ignore

    doctags = output.outputs[0].text
    img_fn = os.path.splitext(img_file)[0]
    output_filename = img_fn + ".dt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doctags)

    # To convert to Docling Document, MD, HTML, etc.:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    # export as any format
    # HTML
    # output_path_html = Path(OUTPUT_DIR) / f"{img_fn}.html"
    # doc.save_as_html(output_path_html)
    # MD
    output_path_md = Path(OUTPUT_DIR) / f"{img_fn}.md"
    doc.save_as_markdown(output_path_md)

print(f"Total time: {time.time() - start_time:.2f} sec")
