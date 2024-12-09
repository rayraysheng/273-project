import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error processing image: {str(e)}"

Delete ocr_module.py from main branch
