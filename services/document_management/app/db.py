from pymongo import MongoClient
from .config import Config

database_name = 'tbd'

client = MongoClient(Config.DATABASE_URL)
db = client[database_name]  # Database name
documents_collection = db["documents"]  # Collection name
