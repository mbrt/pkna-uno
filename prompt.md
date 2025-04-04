Given the input comic book, extract the following information:
* Captions
* Speech balloons
* Characters
* One sentence description of the frame
* Scenes, each grouping multiple frames related to the same scene together
* Page and frame number

Use the provided JSON input describing the main characters of the book for more
background information.

Convert all caps speech bubbles and captions to normal caps. Whenever you can't identify a character by name, describe it and be consistent across frames. Use different names for different characters.

Captions are never associated with a character and could be either the narrator voice, or a sound effect. Speech bubbles are always associated with a character.

Write all output in Italian, never translate to English. Do not skip pages in the output.
You can find the page number at the bottom of each image. If there's none, compute it by using pages that come before and after. For example, if the previous page is 32, the current one should be 33.
