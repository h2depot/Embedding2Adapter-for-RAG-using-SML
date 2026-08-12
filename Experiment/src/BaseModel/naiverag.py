import time

from ..utils.prompt import build_final_prompt

from ..utils.chunking import implement_chunking
from .llm import LLM
from .vector_store import VectorStore


class NaiveRAG:
    def __init__(self, source_path, encoding="utf-8"):
        start = time.perf_counter_ns()
        self.llm = LLM()
        self.chunks = implement_chunking(source_path, encoding)
        self.vector_store = VectorStore(self.chunks)
        self.vector_store.embed_chunks()
        elapsed_sec = (time.perf_counter_ns() - start) / 1_000_000_000
        print(f"NaiveRAG initialized successfully in {elapsed_sec} sec")

    def run(self, query: str):
        result = self.vector_store.search_query(query)
        context = self.construct_context(result)
        return self.llm.generate(build_final_prompt(context, query))

    def search(self, query: str):
        return self.vector_store.search_query(query)

    def get_final_prompt(self, query: str):
        result = self.vector_store.search_query(query)
        context = self.construct_context(result)
        return build_final_prompt(context, query)

    def construct_context(self, search_results: list[dict]) -> str:
        chunks = []
        seen = set()
        for result in search_results:
            chunk = result["chunk"].strip()
            if not chunk or chunk in seen:
                continue
            seen.add(chunk)
            chunks.append(chunk)
        return "\n\n".join(chunks)
