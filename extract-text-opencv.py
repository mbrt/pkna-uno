#!/usr/bin/env python

import os
import tempfile

import click
import cv2
import pytesseract
from layoutparser.models import Detectron2LayoutModel


def extract_with_opencv(image_path, debug=False, debug_dir=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_text = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        if 500 < area < 20000 and 0.5 < aspect_ratio < 2.5:
            roi = image[y : y + h, x : x + w]
            text = pytesseract.image_to_string(roi, lang="ita").strip()
            if text:
                extracted_text.append(text)
                if debug:
                    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    debug_path = None
    if debug and debug_dir:
        debug_path = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, orig)

    return extracted_text, debug_path


def extract_with_layoutparser(image_path, debug=False, debug_dir=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_rgb = image[..., ::-1]

    model = Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
        enforce_cpu=True,
    )

    layout = model.detect(image_rgb)
    text_blocks = [b for b in layout if b.type == "text"]

    extracted_text = []

    for block in text_blocks:
        x1, y1, x2, y2 = map(int, block.coordinates)
        roi = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi, lang="eng").strip()
        if text:
            extracted_text.append(text)
            if debug:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    debug_path = None
    if debug and debug_dir:
        debug_path = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, image)

    return extracted_text, debug_path


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option(
    "--use-dl", is_flag=True, help="Use deep learning detection with LayoutParser"
)
@click.option("--debug", is_flag=True, help="Save debug images with bounding boxes")
def main(input_dir, output_dir, use_dl, debug):
    """Extracts speech bubble text from comic book scans (JPEG) in INPUT_DIR and saves to OUTPUT_DIR."""
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = tempfile.mkdtemp(prefix="comic_debug_") if debug else None

    extract_fn = extract_with_layoutparser if use_dl else extract_with_opencv

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith((".jpg", ".jpeg")):
            full_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            text_path = os.path.join(output_dir, f"{base_name}.txt")

            text, debug_path = extract_fn(full_path, debug=debug, debug_dir=debug_dir)

            with open(text_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(text))

            click.echo(f"Extracted: {text_path}")
            if debug_path:
                click.echo(f"Debug image: {debug_path}")


if __name__ == "__main__":
    main()
