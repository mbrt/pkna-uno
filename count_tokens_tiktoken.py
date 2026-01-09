#!/usr/bin/env python3

import sys
import tiktoken

def count_tokens(file_path: str) -> int:
    """Count tokens in a file using tiktoken (OpenAI's tokenizer)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
    # This should give a reasonable approximation for Gemini
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(content)

    return len(tokens)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: count_tokens_tiktoken.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    token_count = count_tokens(file_path)

    print(f"Token count (tiktoken/cl100k_base): {token_count}")
    print("Note: This is an approximation. Gemini's tokenizer may count differently.")
