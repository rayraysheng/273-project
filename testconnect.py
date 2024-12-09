import requests
import dotenv
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chromadb import HttpClient

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_HOST = "localhost"
CHROMA_DB_PORT = int(os.getenv("CHROMA_DB_PORT"))

def test_connection():
    client = HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
    try:
        client.ping()
        print("ChromaDB connection successful!")
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")

if __name__ == "__main__":
    test_connection()