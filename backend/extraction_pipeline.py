from elastic_client import index_document
from embedding import embedding_text
import uuid

def load_documents():
    return [

    ]

def chunk_texts(text, max_tokens=200):
    return [text]

def build_index():
    docs = load_documents()
    for doc in docs:
        chunks = chunk_texts(doc)