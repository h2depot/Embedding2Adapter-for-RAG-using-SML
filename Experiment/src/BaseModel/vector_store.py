import faiss
import numpy as np
import ollama

from ..utils.config import (
    get_embedding_model,
    get_embedding_model_dim,
    get_ollama_host,
    get_retrieval_k,
)


class VectorStore:
    def __init__(self, chunks=None):
        self.embedding_model = get_embedding_model()
        self.dim = get_embedding_model_dim()
        self.index: faiss.Index | None = None
        self.chunks = list(chunks or [])
        self.client = ollama.Client(host=get_ollama_host())
        self.top_k = get_retrieval_k()
        self.db_init()
        print("VectorStore Initialized")

    def set_chunks(self, chunks: list[str]):
        self.chunks = list(chunks)
        self.db_init()
        self.embed_chunks()

    def embed_chunks(self):
        if not self.chunks:
            raise ValueError("chunks must not be empty.")
        response = self.client.embed(
            model=self.embedding_model,
            input=self.chunks,
            dimensions=self.dim,
        )
        vectors = np.array(response.get("embeddings"), dtype=np.float32)
        #print(f"Vectors shape: {vectors[0].shape}")
        self.db_insert(vectors)

    def search_query(self, query: str):
        response = self.client.embed(
            model=self.embedding_model,
            input=query,
            dimensions=self.dim,
        )
        query_vec = np.array(response.get("embeddings"), dtype=np.float32)
        distances, indices = self.db_search(query_vec)
        return [
            {
                "chunk": self.chunks[int(index)],
                "score": float(distance),
                "index": int(index),
            }
            for distance, index in zip(distances[0], indices[0])
            if index >= 0
        ]

    def db_init(self):
        self.index = faiss.IndexFlatIP(self.dim)

    def db_insert(self, vectors):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def db_search(self, query_vec):
        faiss.normalize_L2(query_vec)
        return self.index.search(query_vec, k=self.top_k)
