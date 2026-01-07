from paddleocr import PaddleOCR # type: ignore
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json


load_dotenv()


# Get free API key from: https://aistudio.google.com/app/apikey
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


def parse_receipt_with_gemini(ocr_text):
    """Parse receipt using Gemini"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
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
    
    response = model.generate_content(prompt)
    return response.text

# Test it
ocr_text = """
STARBUCKS Trinidad & Tobago
Jan05'26 09:32AM
1 Latte Ta         27.00
1 Iced Lenon Pd Ck 26.00
Creditard          92.00
Subiutal           81.78
VAT@12.5%          10.22
"""

ocr = PaddleOCR(
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

def extract_receipt_text(image_path):
    """Extract text from receipt cleanly"""

    # Run OCR
    result = ocr.predict(
        input=image_path
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
    
    return text_with_confidence

# Test it
image_path = "IMG_5790.jpeg"
data = extract_receipt_text(image_path)

result = parse_receipt_with_gemini(data)
print(result)


