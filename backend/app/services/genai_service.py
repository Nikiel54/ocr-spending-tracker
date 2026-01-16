import os
from dotenv import load_dotenv
from google import genai
import json
from typing import List, Dict


class GenAIService:
    def __init__(self):
        load_dotenv()
        #genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        self._model_version = 'gemini-3-flash-preview'
        self.client = genai.Client()
    
    def process_receipt_text(self, ocr_text: List) -> Dict:
        print("Genai running now")
        prompt = f"""Extract the following information from this array of receipt OCR text.
            The text may contain OCR errors - please correct obvious typos.

            {json.dumps(ocr_text)}

            Return ONLY valid JSON (no markdown, no explanation) with this structure:
            {{
                "merchant": "store name",
                "date": "YYYY-MM-DD",
                "total": 0.00,
                "currency": "TTD",
                "items": [
                    {{"name": "item name", "price": 0.00}}
                ],
                "tax": 0.00,
                "payment_method": "cash or card"
            }}

            If you cannot determine a field, use null.
            """
        try:
            response = self.client.models.generate_content(
                model=self._model_version,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error in Genai process: {e}")
            return {
                'Error': 'true'
            }
    