#!/usr/bin/env python3

"""
Example of generating logprobs with Gemini 2.5 Flash Lite.
"""

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()

MODEL_ID = "gemini-2.5-flash-lite"

prompt = "I am not sure if I really like this restaurant a lot."
response_schema = {"type": "STRING", "enum": ["Positive", "Negative", "Neutral"]}

client = genai.Client()
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        response_logprobs=True,
        logprobs=4,
    ),
)
print(response.text)

assert response.candidates is not None
for candidate in response.candidates:
    lpr = candidate.logprobs_result
    assert lpr is not None and lpr.top_candidates is not None
    for c in lpr.top_candidates:
        assert c.candidates is not None
        print([f"{f.log_probability}: {f.token}" for f in c.candidates])
