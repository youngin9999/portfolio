2025 09 01 

계속 사이트에서 코드를 채점할 때 오류가 났었는데 그 원인이 온라인에서 모델을 불러오기 때문이었다. 
paddle_det 모델을 지정해주지 않으면 온라인으로 모델을 불러오는데 그게 원인이었다. 
원인을 찾았으니 해결했고 , 꽤나 개선된 코드 + 모델을 제출했는데 런타임 에러가 나와 아직도 옛날 점수 그대로이다.
런타임 에러가 난 이유는 paddle , yolo를 cpu로 돌리고 있었기 떄문이고 , 그래서  yolo는 gpu 로 추론하도록 바꾸었지만 paddle은 gpu로 바꾸기 어려울 것 같다 . 
그 이유는 채점환경에서는 cuda를 12.8로 고정시켜놓고 있는데 paddlepaddle 에 따르면 cuda12.8에서 돌아가는 버전이 없다 . 그래서 지금 좀 골치아픈 상황이다 . 
직접 실행시켜본 결과 혼합언어 (영어 + 한국어)에서는 tesseract보다 paddle ocr 이 성능이 더 좋아서 paddle ocr 을 사용해야하는데 채점환경에서 cuda버전과 호환이 안되는 문제이다 .
yolo만으로 시간 개선이 좀 됐으면 좋겠다 . 
이제 해야 되는 것들은 yolo bbox 후처리하는 과정에서 ocr bbox는 병합되어 텍스트는 들어가는데 yolo bbox 크기는 병합이 안되는 문제가 있어서 이걸 해결해야한다. 
그리고 order을 subtitle이 연속으로 두번 나오지 않도록 추가 로직 작성해야하고 , 코드 리팩토링으로 최적화 , 
최적화 후 시간이 좀 남는다면 다른 버전의 ocr (easyocr or tesseract)를 돌린 후 좀 더 신뢰도가 높은 ocr로 폴백하는 로직을 만들어서 성능을 개선시키고싶다 . 

Today, I spent time debugging why the code kept failing during evaluation on the competition site.
After some digging, I found that the issue was caused by loading the model online.
When the paddle_det model path isn’t explicitly specified, it tries to download the model during execution.

Since the evaluation environment blocks external internet access, this caused the error.
After identifying the cause, I fixed it and submitted an improved version of both the code and model.
However, the leaderboard score didn’t change because a runtime error occurred .
The new error was due to both PaddleOCR and YOLO running on CPU.
I switched YOLO to use GPU inference, which should help with speed.
However, using GPU for PaddleOCR seems much harder.
The reason is that the evaluation environment is locked to CUDA 12.8,
and according to the official PaddlePaddle documentation,
there is currently no Paddle version that supports CUDA 12.8.
This puts me in a tricky spot.
From my own testing, PaddleOCR performs much better than Tesseract on mixed-language text (Korean + English),
so I really want to keep using it.
But the CUDA incompatibility is a major blocker in the current setup.
I’m hoping that running YOLO on GPU alone will help reduce the overall runtime at least a bit.

 todolist--

Fix the issue where, during YOLO post-processing,
OCR text is correctly merged but the bounding box size is not updated.

Add logic to prevent two subtitles from appearing consecutively in the order.

Refactor and optimize the entire codebase.

If there’s extra time after optimization:

Try running other OCR engines like EasyOCR or Tesseract

Implement a fallback system that selects the most reliable OCR result to improve accuracy.

2025 08 30 
Today, I improved performance by refining the YOLO bounding boxes: removing duplicates, 
merging overlapping boxes, and ensuring that each image contains only one title—any additional titles are downgraded to subtitles. 
Since YOLO detects tables well, I applied a confidence threshold of 0.5 and discarded any detections below that.
 I also utilized OCR data that did not overlap with YOLO detections to further enhance the results.


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

