Given the input comic book, extract the following information:
* Captions
* Speech balloons
* Characters
* One sentence description of the frame
* Scenes, each grouping multiple frames related to the same scene together
* Page and frame number

Character identification: Use dialogues to identify characters' names. Whenever you can't identify a character by name, describe it and be consistent across frames. Use different names for different characters. Use the provided JSON input describing the main characters of the book for more background information. Minimize the number of characters in each scene.

Speech bubbles: Make sure the sentences use correct language and are associated with the right character. Speech bubbles are always associated with a character. Use the dialogue in the scene to identify who's the most likely speaker. The style of the bubble outline (e.g. smooth, spiky, angled) helps identifying different characters, because each use a different style.

Captions: They are never associated with a character and could be either the narrator voice, or a sound effect.

Text format: Convert speech bubbles and captions from all caps to normal caps. Write all output in Italian, never translate to English. 

Pages: Do not skip pages in the output. You can find the page number at the bottom of each image. If there's none, compute it by using pages that come before and after.
