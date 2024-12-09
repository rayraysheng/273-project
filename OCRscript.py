from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure Poppler path
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"

# Initialize Flask app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Process PDF or Image
            if filename.endswith(".pdf"):
                extracted_text = process_pdf(filepath)
            else:
                extracted_text = process_image(filepath)

            return jsonify({"text": extracted_text})

        return jsonify({"error": "Invalid file type. Only PDF and images are allowed."})

    return render_template("upload.html")

def process_image(image_path):
    """Perform OCR on an image file."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="eng")
    return text

def process_pdf(pdf_path):
    """Convert PDF to images and extract text via OCR."""
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    extracted_text = ""
    for image in images:
        extracted_text += pytesseract.image_to_string(image, lang="eng") + "\n"
    return extracted_text

if __name__ == "__main__":
    app.run(debug=True)
