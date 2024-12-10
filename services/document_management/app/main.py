from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import logging
import fitz as pymupdf
from fastapi.middleware.cors import CORSMiddleware
from chromadb import HttpClient
import uuid
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Initialize and configure
app = FastAPI()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
EMBEDDING = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
POPPLER_PATH = "/usr/bin"

# Connect to Chroma DB
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "http://chroma_db_service")
CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT", 8002))

vector_db = HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# File processing
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_doc = pymupdf.open(file_path)
        text = "".join(page.get_text() for page in pdf_doc)
        pdf_doc.close()
        return text
    except Exception as e:
        logging.error("Error extracting text from PDF: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process PDF file.")

def perform_ocr_on_image(image_path: str) -> str:
    """Perform OCR on an image file."""
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image, lang="eng")
    except Exception as e:
        logging.error("Error performing OCR on image: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process image file.")

def convert_pdf_to_images(pdf_path: str) -> List[str]:
    """Convert PDF pages to images for OCR."""
    try:
        return convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    except Exception as e:
        logging.error("Error converting PDF to images: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process PDF file.")

def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded PDF or image file."""
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        if file.filename.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)
            print(extracted_text, flush=True)
            return extracted_text
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            extracted_text = perform_ocr_on_image(file_path)
            print(extracted_text, flush=True)
            return extracted_text
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file type. Only PDF and images are allowed."
            )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def split_text_into_chunks(raw_text: str) -> List[str]:
    """Split raw text into manageable chunks."""
    return TEXT_SPLITTER.split_text(raw_text)


def store_chunks_in_chroma_by_title(text_chunks: List[str], title: str):
    """
    Store text chunks in a Chroma collection named after the manual's title.

    Args:
        text_chunks (List[str]): List of text chunks from the manual.
        title (str): The title of the manual.

    Returns:
        int: Number of chunks stored.
    """
    try:
        # Create or get a collection specific to the title
        manual_collection = vector_db.get_or_create_collection(name=title)

        # Generate embeddings for all text chunks
        embeddings = [EMBEDDING.embed_query(chunk) for chunk in text_chunks]

        # Generate unique IDs for each chunk
        ids = [f"{title}_{uuid.uuid4()}" for _ in text_chunks]

        # Add chunks with embeddings to the title-specific collection
        manual_collection.add(
            documents=text_chunks,  # Add the actual text
            embeddings=embeddings,  # Add the corresponding embeddings
            metadatas=[
                {"title": title} for _ in text_chunks
            ],  # Metadata for each chunk
            ids=ids,  # Unique IDs for each chunk
        )

        return len(text_chunks)
    except Exception as e:
        logging.error("Error storing chunks in Chroma: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Failed to store manual in the database."
        )


######################################################################
# Endpoints
######################################################################


#########################
### Testing Endpoints ###
#########################
@app.get(
    "/health",
    tags=["Utility"],
    summary="Health Check",
    description="Check the health status of the document management service and its connection to Chroma.",
)
async def health():
    """
    Returns a JSON object indicating that the service is healthy and providing information about the Chroma connection.
    """
    connection = str(vector_db.heartbeat())
    return {"status": "healthy", "connection": connection}


######################
### Main Endpoints ###
######################
@app.post(
    "/upload",
    tags=["Documents"],
    summary="Upload Files",
    description="Upload PDF or image files, extract text, and store them in Chroma DB."
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="Upload one or more files."),
    title: str = Form(..., description="The collection title for the files.")
):
    try:
        combined_text = ""
        for file in files:
            combined_text += extract_text_from_file(file) + "\n"

        # Split text into manageable chunks
        text_chunks = TEXT_SPLITTER.split_text(combined_text)

        # Store chunks in Chroma DB
        manual_collection = vector_db.get_or_create_collection(name=title)
        embeddings = [EMBEDDING.embed_query(chunk) for chunk in text_chunks]
        ids = [f"{title}_{uuid.uuid4()}" for _ in text_chunks]

        manual_collection.add(
            documents=text_chunks,
            embeddings=embeddings,
            metadatas=[{"title": title} for _ in text_chunks],
            ids=ids,
        )

        return {"message": f"Documents uploaded under title '{title}'.", "num_chunks": len(text_chunks)}
    except Exception as e:
        logging.error("Error uploading documents: %s", str(e))
        raise HTTPException(status_code=500, detail="Error processing documents.")


@app.get(
    "/manuals",
    tags=["Manuals"],
    summary="List Manuals",
    description="Retrieve all manual titles stored in the database.",
)
async def list_manuals():
    """
    Returns a list of all manual titles (collections) stored in the Chroma database.
    """
    try:
        # Get a list of all collections
        collections = vector_db.list_collections()

        # Extract collection names
        manual_titles = [collection.name for collection in collections]

        return {"manual_titles": manual_titles}
    except Exception as e:
        logging.error("Failed to retrieve manual titles: %s", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving manual titles.")


# @app.delete(
#     "/manual",
#     tags=["Manuals"],
#     summary="Delete Manual",
#     description="Delete all entries in the database for a given manual title.",
# )
# async def delete_manual(
#     title: str = Query(..., description="The title of the manual you want to delete."),
# ):
#     """
#     Deletes all documents associated with the provided manual title.
#     """
#     try:
#         result = collection.get()
#         all_ids = result["ids"]
#         all_metadatas = result["metadatas"]

#         ids_to_delete = [
#             doc_id
#             for doc_id, metadata in zip(all_ids, all_metadatas)
#             if metadata.get("title") == title
#         ]

#         if not ids_to_delete:
#             raise HTTPException(status_code=404, detail=f"Manual '{title}' not found.")

#         collection.delete(ids=ids_to_delete)

#         return {"message": f"Manual '{title}' successfully deleted."}
#     except Exception as e:
#         logging.error("Failed to delete manual: %s", str(e))
#         raise HTTPException(status_code=500, detail="Error deleting manual.")


# @app.get(
#     "/manual",
#     tags=["Manuals"],
#     summary="Get Manual",
#     description="Retrieve the full text content of all uploaded files for a given manual title.",
# )
# async def get_manual(
#     title: str = Query(
#         ..., description="The title of the manual you want to retrieve."
#     ),
# ):
#     """
#     Returns the full concatenated text of the specified manual.
#     """
#     try:
#         result = collection.get()
#         all_documents = result["documents"]
#         all_metadatas = result["metadatas"]

#         filtered_documents = [
#             doc
#             for doc, metadata in zip(all_documents, all_metadatas)
#             if metadata.get("title") == title
#         ]

#         if not filtered_documents:
#             raise HTTPException(status_code=404, detail=f"Manual '{title}' not found.")

#         full_text = " ".join(filtered_documents)
#         return {"title": title, "content": full_text}
#     except Exception as e:
#         logging.error("Failed to retrieve manual: %s", str(e))
#         raise HTTPException(status_code=500, detail="Error retrieving manual.")
