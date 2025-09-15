import pytesseract
from PIL import Image
from pdf2image import convert_from_path

def extract_text(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            images = convert_from_path(file_path)
            text = ''
            for img in images:
                text += pytesseract.image_to_string(img)
            return text
        else:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
