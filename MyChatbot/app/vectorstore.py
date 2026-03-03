import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []

    def add(self, vectors, texts):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_vector, top_k):
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        return [self.texts[i] for i in indices[0]]
