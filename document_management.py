from ocr_module import extract_text_from_image

def process_document(image_path):
    """Process a document by extracting text from it using OCR."""
    print(f"Processing image: {image_path}")
    text = extract_text_from_image(image_path)
    print(f"Extracted text: \n{text}")

if __name__ == "__main__":
    # Example image file path
    image_path = 'sample_image.png'  # Replace this with the path to your image file
    process_document(image_path)
