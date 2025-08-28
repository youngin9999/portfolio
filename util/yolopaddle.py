
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False, # Use use_doc_unwarping to enable/disable document unwarping module
    use_textline_orientation=True, # Use use_textline_orientation to enable/disable textline orientation classification model
    text_recognition_model_dir="/workspace/util/model/ocr/korean_PP-OCRv5_mobile_rec_infer"
)
result = ocr.predict("/workspace/util/data/test/TEST_03.jpg")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")
