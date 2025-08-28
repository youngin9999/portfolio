2025 08 29

We replaced Tesseract with PaddleOCR (PP-OCRv5, Korean) and changed the flow to one full-page OCR → merge with YOLO layout. 
This improved mixed Korean+English accuracy, reduced per-ROI OCR calls, and tried to fix text leaking. 
The entry point is yolopad.py.



2025 08- 27

Today I dug into the evaluation logic—IoU, mAP, NED, and reading order—and learned how each impacts the score.
I cleanly mapped DocLayout-YOLO labels to the contest’s six classes, refined reading-order rules (auto 1/2-column), and tightened the pipeline.
I audited environment dependencies (Poppler, LibreOffice, doclayout-yolo) and path issues, confirming root causes of the errors one by one.
Finally, I swapped EasyOCR for Tesseract, built a runnable script.py, and finished a setup that outputs a submission-ready submission.csv.

2025 08-25

github's open model is better than mine.

I should start ocr by using tesseract.

 
2025 08-15
Today, I tried training YOLOv11n.
However, I think my first model was overfitted and didn’t perform well.
The sample model actually produced better results than mine.
I believe the main issue was the small size of the training dataset and overfitting due to a high number of epochs.

2025.0813-14 (sam)
Today, I learned how to use Docker.
Now I can run my script in an environment identical to the competition site.

✅ My To-Do List

1. Search for the official grading criteria on the competition website.

2. Generate YOLO and OCR datasets.

3. Design the post-processing logic.

4. Design the order-setting logic.

2025.0807 (samsung competition)

This competition is in the AI domain. , We need to build a model that imitates how humans read text and interpret images. So I plan to research OCR, how humans actually read, and how the areas of the brain responsible for reading function.
Today, I just planned my schedule and logic First, I will perform OCR to detect texts and images, and classify them into categories.
Second, I will group these blocks and optimize the bounding boxes.
Third, I will determine the reading order.
Finally, I will export the result as a CSV file.

+ I hope I will grow stronger through this process.

2025.0714(Cj competition)

We need to find a better solution here . We have to identify the areas where improvements can be made . Right now , the approach is to theordtically find the minimum number of vehecles first , and then optimize the total distance cost based on that . But im not sure where exactly we can improve. We need to identify potential areas for improvement, since the competition deadline is approaching

We fine the issue . There are three types of boxes . and we curretly classify vehicles into multiple types to determine the minimum nmber of vehicles needed . however , the number of types-0 boxes that can be loaded into vehicles capacle of carrying type-0 boxes is significantly higher than the number of boxes of other types . Every multiple of 84 makes it easier for these vehicels to load type-0 boxes , but since 84 is large number , it seems that if we introduce more vehicle type that can a smaller number of type-0 boxes , we could achieve a more optimal solution .

+ We got second place.

