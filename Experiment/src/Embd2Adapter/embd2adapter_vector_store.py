import faiss
import numpy as np
from ..utils.config import get_retrieval_k
from .embd_model import Embd_Model

class Embd2Adapter_VectorStore:
    def __init__(self, embd_model: Embd_Model, chunks: list[str] | None = None):
        self.embd_model = embd_model
        self.dim = embd_model.dim
        self.index: faiss.Index | None = None
        self.chunks = chunks or []
        self.top_k = get_retrieval_k()
        self.db_init()
        print("VectorStore Initialized")

    def embedding_context(self, context: str | list[str]):
        return self.embd_model.embed(context)

    def set_chunks(self, chunks: list[str]):
        self.chunks = list(chunks)
        self.db_init()
        self.embed_chunks()

    def embed_chunks(self):
        if not self.chunks:
            raise ValueError("chunks must not be empty.")
        vectors = self.embd_model.embed(self.chunks)
        self.db_insert(vectors)

    def search_query(self, query: str):
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Vector store must contain embedded chunks before search.")
        query_vec = self.embd_model.embed(query)
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
