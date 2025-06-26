from embedding import embedding_text
from elastic_client import search_similar
from llm_client import call_llm

def rag_pipeline(question: str):
    query_vector = embedding_text(question)
    docs = search_similar(question, query_vector)
    context = "\n".join(docs["_source"]["text"] for doc in docs)

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return call_llm(prompt)