from app.chunking import chunk_text
from app.embedding import get_embeddings
from app.vectorstore import VectorStore
from app.generator import generate_answer
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

class RAGPipeline:
    def __init__(self, document_path):
        with open(document_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        embeddings = get_embeddings(chunks)

        self.vectorstore = VectorStore(embeddings.shape[1])
        self.vectorstore.add(embeddings, chunks)

    def query(self, question):
        question_embedding = get_embeddings([question])
        contexts = self.vectorstore.search(question_embedding, TOP_K)
        combined_context = "\n".join(contexts)
        return generate_answer(combined_context, question)
