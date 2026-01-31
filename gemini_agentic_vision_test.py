#!/usr/bin/env python3

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

image = types.Part.from_uri(
    file_uri="https://goo.gle/instrument-img",
    mime_type="image/jpeg",
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        image,
        "Zoom into the expression pedals and tell me how many pedals are there?",
    ],
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    ),
)

print(response.model_dump())
print(response.text)
