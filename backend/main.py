from fastAPI import FastAPI, Request
from pydantic import BaseModel
from rag import rag_pipeline

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_query(query: Query):
    answer = rag_pipeline(query.question)
    return {"answer": answer}