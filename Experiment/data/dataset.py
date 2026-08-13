from abc import ABC, abstractmethod
from pprint import pprint
import random

from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import HfApi

from ..src.utils.chunking import chunking_rawtext
from ..src.utils.config import get_chunk_overlap, get_chunk_size


class Dataset(ABC):
    """Common loader and normalized output interface for benchmark datasets."""

    RETRIEVAL_K = 5

    OUTPUT_FIELDS = (
        "query",
        "full_context",
        "gold_context",
        "distractor",
        "answer",
    )

    def __init__(self, info: dict):
        self.api = HfApi()
        self.info = info
        self.ds_name = info["ds_name"]
        self.config = info["config"]
        self.revision = info["revision"]
        self.split = info["split"]
        self.sampling = info["sampling"]
        self.eval_size = info["eval_size"]
        self.val_size = info["val_size"]
        self.dataset: HFDataset | None = None
        self.extracted_ds = self._empty_extracted_dataset()
        self.train_ds = self._empty_extracted_dataset()
        self.val_ds = self._empty_extracted_dataset()
        self.eval_ds = self._empty_extracted_dataset()

        self.dataset = self.load_sampled_dataset()

    @classmethod
    def _empty_extracted_dataset(cls):
        return {field: [] for field in cls.OUTPUT_FIELDS}

    def load_sampled_dataset(self) -> HFDataset:
        target_dataset = load_dataset(self.ds_name,self.config,revision=self.revision,split=self.split)
        sample_seed = self.sampling.get("seed")
        sample_size = self.sampling.get("size")
        if sample_size is not None:
            sample_size = min(sample_size, len(target_dataset))
            target_dataset = target_dataset.shuffle(seed=sample_seed).select(range(sample_size))
        return target_dataset

    def split_dataset(self):
        """Keep this benchmark's evaluation rows separate from its training rows."""
        self._convert_contexts_to_rag_chunks()
        row_count = len(self.extracted_ds["query"])
        if self.eval_size < 0 or self.val_size < 0:
            raise ValueError("eval_size and val_size must be non-negative.")
        if self.eval_size + self.val_size > row_count:
            raise ValueError(
                f"eval_size + val_size ({self.eval_size + self.val_size}) "
                f"exceeds the number of sampled rows ({row_count}) for "
                f"{self.ds_name}."
            )

        train_end = row_count - self.eval_size - self.val_size
        val_end = row_count - self.eval_size
        for field in self.OUTPUT_FIELDS:
            self.train_ds[field] = self.extracted_ds[field][:train_end]
            self.val_ds[field] = self.extracted_ds[field][train_end:val_end]
            self.eval_ds[field] = self.extracted_ds[field][val_end:]

        # The split datasets are the public representation after extraction.
        self.extracted_ds = self._empty_extracted_dataset()

    @staticmethod
    def _flatten_context(context) -> list[str]:
        """Normalize the benchmark-specific context shapes into text documents."""
        if isinstance(context, str):
            return [context]
        if isinstance(context, dict):
            context = context.get("sentences", context.get("content", []))

        documents = []
        for item in context:
            if isinstance(item, str):
                documents.append(item)
            else:
                documents.extend(str(text) for text in item)
        return documents

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    def _chunk_documents(self, documents: list[str]) -> list[str]:
        raw_text = "\n\n".join(text for text in documents if text and text.strip())
        if not raw_text:
            return []
        return chunking_rawtext(
            raw_text,
            chunk_size=get_chunk_size(),
            chunk_overlap=get_chunk_overlap(),
        )

    def _convert_contexts_to_rag_chunks(self):
        """Replace sentence-level contexts with the chunks seen by a RAG system."""
        for index, full_context in enumerate(self.extracted_ds["full_context"]):
            source_gold = self.extracted_ds["gold_context"][index]
            source_distractors = self.extracted_ds["distractor"][index]
            context_chunks = self._chunk_documents(self._flatten_context(full_context))
            gold_pairs = [
                (text, self._normalize_text(text))
                for text in source_gold
                if text and text.strip()
            ]
            normalized_chunks = [self._normalize_text(chunk) for chunk in context_chunks]
            gold_chunks = []
            for gold_text, normalized in gold_pairs:
                matching_chunk = next(
                    (
                        chunk
                        for chunk, normalized_chunk in zip(context_chunks, normalized_chunks)
                        if normalized in normalized_chunk
                    ),
                    None,
                )
                if matching_chunk is not None:
                    gold_chunks.append(matching_chunk)
                else:
                    # A very long supporting sentence can itself be split by FLC.
                    gold_chunks.extend(self._chunk_documents([gold_text]))

            gold_chunks = list(dict.fromkeys(gold_chunks))
            if len(gold_chunks) > self.RETRIEVAL_K:
                gold_chunks = random.sample(gold_chunks, k=self.RETRIEVAL_K)
            gold_set = set(gold_chunks)
            distractor_candidates = [
                chunk for chunk in context_chunks if chunk not in gold_set
            ]
            for document in self._flatten_context(source_distractors):
                distractor_candidates.extend(self._chunk_documents([document]))
            distractor_candidates = [
                chunk
                for chunk in dict.fromkeys(distractor_candidates)
                if chunk not in gold_set
            ]

            # Expose one uniform retrieval corpus for every benchmark.
            self.extracted_ds["full_context"][index] = list(
                dict.fromkeys(context_chunks + distractor_candidates)
            )

            distractor_count = (
                0
                if len(gold_chunks) >= self.RETRIEVAL_K
                else max(0, self.RETRIEVAL_K - len(gold_chunks))
            )
            self.extracted_ds["gold_context"][index] = gold_chunks
            self.extracted_ds["distractor"][index] = distractor_candidates[:distractor_count]


    def check_latest_revision(self):
        commits = self.api.list_repo_commits(
            repo_id=self.ds_name,
            repo_type="dataset",
        )

        print(f"--- dataset '{self.ds_name}' revision ---")
        for commit in commits[:5]:
            print(f"version commit ID: {commit.commit_id}")
            print(f"at               : {commit.created_at}")
            print(f"message          : {commit.title}")
            print("-" * 50)
        return commits[0]

    def show_top5(self):
        for name, dataset in (
            ("TRAIN_DATASET", self.train_ds),
            ("VALIDATION_DATASET", self.val_ds),
            ("EVALUATION_DATASET", self.eval_ds),
        ):
            print(f"{name}:")
            for index in range(min(5, len(dataset["query"]))):
                pprint({field: dataset[field][index] for field in self.OUTPUT_FIELDS})

    def show_top1_rawdata(self):
        if self.dataset is None:
            raise ValueError("Dataset is not loaded.")

        print("--- raw dataset summary ---")
        print(f"name     : {self.ds_name}")
        print(f"config   : {self.config}")
        print(f"split    : {self.split}")
        print(f"revision : {self.revision}")
        print(f"rows     : {len(self.dataset)}")
        print(f"columns  : {self.dataset.column_names}")
        print("\n--- features ---")
        pprint(self.dataset.features)
        print("\n--- top 1 raw row ---")
        pprint(self.dataset[0])
