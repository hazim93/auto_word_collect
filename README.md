Word Collect is a mobile game where the goal is to form words based on the letters given.

Goal: To use machine learning to cheat i.e. suggest possible words to solve the Word Collect game from a screenshot

This project will be split to a few parts:

<b>Phase 1</b>: Word suggestion is done on the PC <br>
<b>Phase 2</b>: Word suggestion is done on an Android device <br>
<b>Phase 3</b>: Live suggestion using a smartphone camera?? <br>

For Phase 1, we'll further break it down to a few parts:

<b>Part 1</b>: Detect tiles from the screenshots provided & crop tiles <br>
        Custom object detection from image using PyTorch / Detecto <br>
        Image manipulation using OpenCV
        
<b>Part 2</b>: Feed into an OCR <br>
        Will start by testing using PyTesseract <br>
        If it fails, train a model to classify the letters 
        
<b>Part 3</b>: Get an English library and suggest words from there based on what the letters we get from the OCR

