Given the input comic book, extract the following information:
* Captions
* Speech balloons
* Characters
* One sentence description of the frame
* Scenes, each grouping multiple frames related to the same scene together

Use the provided JSON input describing the main characters of the book for more
background information.

Output as a JSON, following this example:

```
{
    "scenes": [
        {
            "frames": [
                {
                    "captions": [
                        {
                            "text": "Long time ago..."
                        }
                    ],
                    "bubbles": [
                        {
                            "text": "Where are we?",
                            "character": "Alice"
                        }
                    ],
                    "description": "One sentence description of the frame."
                }
            ]
        }
    ]
}
```

Convert all caps speech bubbles and captions to normal caps. Whenever you can't identify a character by name, invent one that fits its look and be consistent across frames. Use different names for different characters.

Captions are never associated with a character and could be either the narrator voice, or a sound effect. Speech bubbles are always associated with a character.
