import numpy as np
import ollama

from ..utils.config import (
    get_embedding_model,
    get_embedding_model_dim,
    get_ollama_host,
)


class Embd_Model:
    """Own the embedding model configuration and Ollama client."""

    def __init__(self):
        self.model_name = get_embedding_model()
        self.dim = get_embedding_model_dim()
        self.client = ollama.Client(host=get_ollama_host())

    def embed(self, text: str | list[str]) -> np.ndarray:
        response = self.client.embed(
            model=self.model_name,
            input=text,
            dimensions=self.dim,
        )
        return np.asarray(response["embeddings"], dtype=np.float32)

    def unload(self) -> None:
        """Unload the embedding model from Ollama memory."""
        self.client.embed(
            model=self.model_name,
            input="",
            keep_alive=0,
            dimensions=self.dim,
        )
        print(f"Embedding model unloaded: {self.model_name}")
