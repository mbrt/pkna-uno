#!/usr/bin/env python3

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(
    model="vertex_ai/claude-opus-4-5@20251101",
    vertex_credentials=os.getenv("VERTEX_AI_CREDS"),
    vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

print(lm("Hello, how are you?"))
