import requests
import numpy as np
from app.config import OLLAMA_BASE_URL, EMBED_MODEL

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        vector = response.json()["embedding"]
        embeddings.append(vector)
    return np.array(embeddings).astype("float32")
