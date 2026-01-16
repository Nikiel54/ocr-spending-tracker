from paddleocr import PaddleOCR

class OcrProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang='en',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    
    def process_receipt_img(self, img_path: str):
        print("Processing image now")
        result = self.ocr.predict(
            input=img_path
        )
        
        # Extract just the text and confidence
        extracted_data = []
        
        for line in result[0]['rec_texts']:
            text = line
            extracted_data.append(text)
        
        # Also get confidence scores
        confidences = result[0]['rec_scores']
        
        # Combine text with confidence
        text_with_confidence = []
        for text, conf in zip(extracted_data, confidences):
            text_with_confidence.append({
                'text': text,
                #'confidence': float(conf)
            })
        print("Done with ocr")
        return text_with_confidence
        