import requests
from app.config import OLLAMA_BASE_URL, LLM_MODEL

def generate_answer(context, question):
    prompt = f"""
    Use only the context below to answer the question.

    Context:
    {context}

    Question:
    {question}
    """

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )

    return response.json()["response"]
