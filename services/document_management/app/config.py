import os
from dotenv import load_dotenv

load_dotenv()  

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
    DOCUMENT_MANAGEMENT_PORT = os.getenv("DOCUMENT_MANAGEMENT_PORT", "8001")
