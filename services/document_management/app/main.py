from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
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

# Connect to Chroma DB
CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "http://chroma_db_service")
CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT", 8002))

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

#########################
### Testing Endpoints ###
#########################
@app.get("/health", tags=["Utility"], summary="Health Check", description="Check the health status of the document management service and its connection to Chroma.")
async def health():
    """
    Returns a JSON object indicating that the service is healthy and providing information about the Chroma connection.
    """
    connection = str(vector_db.heartbeat())
    return {"status": "healthy", "connection": connection}


@app.get("/manuals", tags=["Manuals"], summary="List Manuals", description="Retrieve all manual titles stored in the database.")
async def list_manuals():
    """
    Returns a list of all unique manual titles.
    """
    try:
        result = collection.get()
        all_metadatas = result["metadatas"]
        titles = {metadata["title"] for metadata in all_metadatas if "title" in metadata}
        return {"manual_titles": list(titles)}
    except Exception as e:
        logging.error("Failed to retrieve manual titles: %s", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving manual titles.")


@app.get("/manual", tags=["Manuals"], summary="Get Manual", description="Retrieve the full text content of all uploaded files for a given manual title.")
async def get_manual(
    title: str = Query(..., description="The title of the manual you want to retrieve.")
):
    """
    Returns the full concatenated text of the specified manual.
    """
    try:
        result = collection.get()
        all_documents = result["documents"]
        all_metadatas = result["metadatas"]

        filtered_documents = [doc for doc, metadata in zip(all_documents, all_metadatas) if metadata.get("title") == title]

        if not filtered_documents:
            raise HTTPException(status_code=404, detail=f"Manual '{title}' not found.")
        
        full_text = " ".join(filtered_documents)
        return {"title": title, "content": full_text}
    except Exception as e:
        logging.error("Failed to retrieve manual: %s", str(e))
        raise HTTPException(status_code=500, detail="Error retrieving manual.")

######################
### Main Endpoints ###
######################
@app.post("/upload", tags=["Manuals"], summary="Upload Manual", description="Upload PDF files and store them under a specified manual title.")
async def upload_manual(
    files: List[UploadFile] = File(..., description="One or more PDF files to upload."),
    title: str = Form(..., description="The title under which these PDF files should be stored.")
):
    """
    Uploads one or more PDF files, extracts their text, splits it into chunks, and stores it in Chroma under the given title.
    """
    try:
        raw_text = extract_text_from_pdf(files)
        text_chunks = split_text_into_chunks(raw_text)
        num_chunks = store_chunks_in_chroma(text_chunks, title)
        return {"message": f"Manual '{title}' successfully uploaded.", "num_chunks": num_chunks}
    except Exception as e:
        logging.error("Error uploading manual: %s", str(e))
        raise HTTPException(status_code=500, detail="Error uploading manual.")


@app.delete("/manual", tags=["Manuals"], summary="Delete Manual", description="Delete all entries in the database for a given manual title.")
async def delete_manual(
    title: str = Query(..., description="The title of the manual you want to delete.")
):
    """
    Deletes all documents associated with the provided manual title.
    """
    try:
        result = collection.get()
        all_ids = result["ids"]
        all_metadatas = result["metadatas"]

        ids_to_delete = [doc_id for doc_id, metadata in zip(all_ids, all_metadatas) if metadata.get("title") == title]

        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Manual '{title}' not found.")

        collection.delete(ids=ids_to_delete)

        return {"message": f"Manual '{title}' successfully deleted."}
    except Exception as e:
        logging.error("Failed to delete manual: %s", str(e))
        raise HTTPException(status_code=500, detail="Error deleting manual.")
