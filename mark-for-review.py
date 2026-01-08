#!/usr/bin/env python

import os
import json
import sys

if len(sys.argv) < 2:
    raise ValueError("Usage: python review-app.py <input_json_path>")

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = "input/review/manual.json"

# The input JSON file is expected to be in this format:
#
# {
#   "input_path": "input/pkna/XXX/YYY.jpg",
#   "extracted": {
#     "uno_present": true,
#     "dialogue": [
#       {
#         "character": "pk",
#         "text": "...Hanno ripreso in mano la situazione giusto in tempo per evitare un disastro! Così imparano a fidarsi di certi sistemi sotto-sviluppati!"
#       },
#       {
#         "character": "uno",
#         "text": "Zitto, criticone artificiale!"
#       },
#       ...
#     ]
#   }
# }

# The output JSON file will be in this format:
#
# [
#   {
#     "image": "input/pkna/pkna-0-3/PK.Vol.0-3 002.jpg",
#     "ocr": {
#       "dialogue": [
#         {
#           "character": "pk",
#           "text": "...e di pericoloso!"
#         },
#         {
#           "character": "uno",
#           "text": "Che cosa fate qui, voi Coolflames?"
#         },
#         ...
#       ]
#     }
#   },
#   ...
# ]

# Load the input JSON file
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    input_data = json.load(f)

# Load the output JSON file if it exists, otherwise create an empty list
output_data = []
try:
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        output_data = json.load(f)
except:
    pass


# Transform the data into the desired output format.
output_item = {
    "image": input_data["input_path"],
    "ocr": {
        "dialogue": input_data["extracted"]["dialogue"],
    },
    "full": input_data["extracted"]
}
output_data.append(output_item)


# Write the output JSON file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
