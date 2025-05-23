import json
import os
import streamlit as st
import sys
from PIL import Image


if len(sys.argv) < 2:
    raise ValueError("Usage: python review-app.py <input_json_path>")

DATA_PATH = sys.argv[1]
REVIEWED_PATH = "../output/dataset/reviewed.jsonl"

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load reviewed items
reviewed_set = set()
if os.path.exists(REVIEWED_PATH):
    with open(REVIEWED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            reviewed_set.add(item["image"])

# Find first unreviewed index
unreviewed_items = [d for d in data if d["image"] not in reviewed_set]
total = len(data)
remaining = len(unreviewed_items)

if remaining == 0:
    st.success("✅ All images have been reviewed!")
    st.stop()

# Use session state to track index
if "index" not in st.session_state:
    st.session_state.index = 0

current = unreviewed_items[st.session_state.index]
image_path = current["image"]
ocr_data = current["ocr"]

st.set_page_config(layout="wide")
st.title("📝 OCR Review Tool")
st.caption(f"{remaining} unreviewed of {total} total")

# Make columns use the full container width
col1, col2 = st.columns(2, gap="large")

with col1:
    st.image(Image.open(image_path), caption=image_path, use_container_width=True)

with col2:
    st.subheader("Edit OCR JSON")
    edited_text = st.text_area("OCR JSON", json.dumps(ocr_data, indent=2), height=400)

    if st.button("✅ Save & Next"):
        try:
            edited_json = json.loads(edited_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        current["ocr"] = edited_json
        with open(REVIEWED_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(current, ensure_ascii=False) + "\n")

        st.session_state.index += 1
        st.rerun()
