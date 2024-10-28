from bson.objectid import ObjectId
from pymongo.collection import Collection
from .models import Document

def create_document(collection: Collection, document: Document):
    result = collection.insert_one(document.dict())
    return str(result.inserted_id)

def get_document(collection: Collection, document_id: str):
    document = collection.find_one({"_id": ObjectId(document_id)})
    if document:
        document["id"] = str(document["_id"])
        del document["_id"]
    return document

def update_document(collection: Collection, document_id: str, updated_data: dict):
    result = collection.update_one(
        {"_id": ObjectId(document_id)},
        {"$set": updated_data}
    )
    return result.modified_count > 0

def delete_document(collection: Collection, document_id: str):
    result = collection.delete_one({"_id": ObjectId(document_id)})
    return result.deleted_count > 0
