from fastapi import FastAPI, HTTPException
from .db import documents_collection
from .models import Document
from .crud import create_document, get_document, update_document, delete_document

app = FastAPI()

@app.post("/documents/", response_model=dict)
def create_document_endpoint(document: Document):
    document_id = create_document(documents_collection, document)
    return {"id": document_id}

@app.get("/documents/{document_id}", response_model=dict)
def read_document(document_id: str):
    document = get_document(documents_collection, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.put("/documents/{document_id}", response_model=dict)
def update_document_endpoint(document_id: str, updated_data: Document):
    success = update_document(documents_collection, document_id, updated_data.dict(exclude_unset=True))
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": "Document updated successfully"}

@app.delete("/documents/{document_id}", response_model=dict)
def delete_document_endpoint(document_id: str):
    success = delete_document(documents_collection, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": "Document deleted successfully"}
