from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def store_vector(text_chunks, collection_name):
    api_key = OPENAI_API_KEY
    embedding = OpenAIEmbeddings(api_key=api_key)

    vector_db = Chroma.from_texts(
        text_chunks,
        embedding=embedding,
        persist_directory="./data",
        collection_name=collection_name,
    )
    return vector_db


def inject_data(files, title):
    for file_path in files:
        content = read_file_content(file_path)
        text_chunks = get_text_chunks(content)
        store_vector(text_chunks, title)


# Example usage
files = ["file1.txt", "file2.txt"]
title = "example_manual"
inject_data(files, title)
