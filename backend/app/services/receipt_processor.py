from services.genai_service import GenAIService
from services.ocr_service import OcrProcessor
from fastapi import UploadFile
from typing import List

class ReceiptProcessor:
    def __init__(self):
        self.genai = GenAIService()
        self.ocr = OcrProcessor()

    def process_recipts(self, images: List):
        for img in images:
            extracted_text = self.ocr.process_receipt_img(img)
            formatted_text = self.genai.process_receipt_text(extracted_text)
            print(formatted_text)
            print("\n")
        return True