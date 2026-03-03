from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import RAGPipeline

app = FastAPI()
rag = RAGPipeline("data/sample.txt")

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Question):
    answer = rag.query(q.question)
    return {"answer": answer}
