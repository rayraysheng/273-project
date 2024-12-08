from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import logging
import fitz as pymupdf
from fastapi.middleware.cors import CORSMiddleware
from chromadb import HttpClient
import uuid


# Initialize and configure
app = FastAPI()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
EMBEDDING = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Database configuration
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "http://chroma_db_service")
CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT", 8000))

# Initialize the HttpClient WITHOUT Settings
vector_db = HttpClient(
    host=CHROMA_DB_HOST,
    port=CHROMA_DB_PORT
)
collection = vector_db.get_or_create_collection(name="manuals")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# PDF processing
def extract_text_from_pdf(uploaded_files: List[UploadFile]) -> str:
    """Extract text from a list of uploaded PDF files."""
    try:
        text = ""
        for file in uploaded_files:
            with open(file.filename, "wb") as f:
                f.write(file.file.read())  # Save the file temporarily
            pdf_doc = pymupdf.open(file.filename)  # Open the PDF
            for page in pdf_doc:
                text += page.get_text()  # Extract text from each page
            os.remove(file.filename)  # Clean up the temporary file
        return text
    except Exception as e:
        logging.error("Error extracting text from PDF: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process PDF files.")

def split_text_into_chunks(raw_text: str) -> List[str]:
    """Split raw text into manageable chunks."""
    return TEXT_SPLITTER.split_text(raw_text)

def store_chunks_in_chroma(text_chunks: List[str], title: str):
    try:
        ids = [f"{title}_{uuid.uuid4()}" for _ in text_chunks]

        collection.add(
            documents=text_chunks,
            metadatas=[{"title": title} for _ in text_chunks],
            ids = ids
        )
        return len(text_chunks)
    except Exception as e:
        logging.error("Error storing chunks in Chroma: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to store manual in the database.")


######################################################################
# Endpoints
######################################################################

# Test routes
@app.get("/health")
async def health():
    connection = str(vector_db.heartbeat())
    return {
        "status": "healthy",
        "connection": connection
        }

@app.get("/manuals")
async def list_manuals():
    """
    Retrieve all manual titles stored in the database.
    """
    try:
        # Fetch all documents and their metadata from the collection
        result = collection.get()
        all_metadatas = result["metadatas"]

        # Extract unique titles from metadata
        titles = {metadata["title"] for metadata in all_metadatas if "title" in metadata}

        return {"manual_titles": list(titles)}
    except Exception as e:
        logging.error("Failed to retrieve manual titles: %s", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving manual titles.")

@app.get("/manual")
async def get_manual(title: str):
    """
    Retrieve the full text content of all uploaded files for a manual with the given title.
    """
    try:
        # Retrieve all documents and metadata
        result = collection.get()
        all_documents = result["documents"]
        all_metadatas = result["metadatas"]

        # Filter documents by title
        filtered_documents = [
            doc for doc, metadata in zip(all_documents, all_metadatas) if metadata.get("title") == title
        ]

        # Combine results into a single text
        if not filtered_documents:
            raise HTTPException(status_code=404, detail=f"Manual '{title}' not found.")
        
        full_text = " ".join(filtered_documents)
        return {"title": title, "content": full_text}
    except Exception as e:
        logging.error("Failed to retrieve manual: %s", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving manual.")


# Production routes
@app.post("/upload")
async def upload_manual(files: List[UploadFile], title: str = Form(...)):
    try:
        # Extract text from PDF files
        raw_text = extract_text_from_pdf(files)
        
        # Split text into chunks
        text_chunks = split_text_into_chunks(raw_text)
        
        # Store chunks in Chroma
        num_chunks = store_chunks_in_chroma(text_chunks, title)
        
        return {"message": f"Manual '{title}' successfully uploaded.", "num_chunks": num_chunks}
    except Exception as e:
        logging.error("Error uploading manual: %s", str(e))
        raise HTTPException(status_code=500, detail="Error uploading manual.")

@app.delete("/manual")
async def delete_manual(title: str):
    """
    Delete all entries in the database with the given title.
    """
    try:
        # Retrieve all documents and their metadata
        result = collection.get()
        all_ids = result["ids"]
        all_metadatas = result["metadatas"]

        # Find IDs of documents with the matching title
        ids_to_delete = [
            doc_id for doc_id, metadata in zip(all_ids, all_metadatas) if metadata.get("title") == title
        ]
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Manual '{title}' not found.")

        # Delete documents by IDs
        collection.delete(ids=ids_to_delete)

        return {"message": f"Manual '{title}' successfully deleted."}
    except Exception as e:
        logging.error("Failed to delete manual: %s", str(e))
        raise HTTPException(status_code=500, detail="Error deleting manual.")

