from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...data.dataset_constructor import DatasetConstructor
from ..utils.config import get_global_seed, get_hypernet_info
from ..utils.evaluation import token_f1_score
from ..utils.prompt import build_final_prompt
from .embd_model import Embd_Model


class HyperNetTrainer:
    """Shared data preparation for the different HyperNet strategies.

    Subclasses own the adapter/controller setup, training loop, and generation
    because those operations depend on the conditioning strategy.
    """

    context_embedding_field = "context_embedding"
    query_embedding_field = "query_embedding"

    @staticmethod
    def _resolve_cache_path(configured_path: str) -> Path:
        cache_path = Path(configured_path)
        if cache_path.is_absolute():
            return cache_path
        repository_root = Path(__file__).resolve().parents[3]
        return repository_root / cache_path

    def __init__(
        self,
        dataset: DatasetConstructor,
        embd_model: Embd_Model,
    ):
        self.info = get_hypernet_info()
        self.model_id = self.info["model_id"]
        self.dataset = dataset
        self.embd_model = embd_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.initialize_models()

    def initialize_models(self):
        training_info = self.info["training"]
        use_bf16 = training_info.get("bf16", False) and self.device.type == "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.to(self.device)
        return model, tokenizer

    def prepare_datasets(self):
        cache_path = self._resolve_cache_path(
            self.info["training"]["embedding_cache_path"]
        )
        self.train_ds = self._prepare_dataset(
            self.dataset.train_ds,
            cache_path=cache_path,
            split_name="training",
        )
        self.val_ds = self._prepare_dataset(
            self.dataset.val_ds,
            cache_path=cache_path.with_name(
                f"{cache_path.stem}_validation{cache_path.suffix}"
            ),
            split_name="validation",
        )

    def _prepare_dataset(self, dataset, cache_path: Path, split_name: str):
        contexts = self.build_contexts(
            dataset["gold_context"],
            dataset["distractor"],
        )
        dataset["training_context"] = contexts
        dataset[self.context_embedding_field] = self.embed_contexts(
            contexts,
            cache_path=cache_path,
            split_name=split_name,
        )
        query_cache_path = cache_path.with_name(
            f"{cache_path.stem}_queries{cache_path.suffix}"
        )
        dataset[self.query_embedding_field] = self.embed_queries(
            dataset["query"],
            cache_path=query_cache_path,
            split_name=split_name,
        )
        return self.tokenize(dataset, split_name=split_name)

    def embed_queries(
        self,
        queries: list[str],
        cache_path: Path | None = None,
        split_name: str = "training",
    ) -> list[list[float]]:
        """Load cached query embeddings, or create and cache them."""
        if not queries:
            return []

        if cache_path is None:
            context_cache_path = self._resolve_cache_path(
                self.info["training"]["embedding_cache_path"]
            )
            cache_path = context_cache_path.with_name(
                f"{context_cache_path.stem}_queries{context_cache_path.suffix}"
            )
        cached = self._load_cache(cache_path, len(queries))
        if cached is not None:
            print(f"Loaded {split_name} query embeddings from {cache_path}")
            return cached

        embeddings = self._embed_batches(
            queries, desc=f"Embedding {split_name} queries"
        )
        self._save_cache(cache_path, embeddings)
        print(f"Saved {split_name} query embeddings to {cache_path}")
        return embeddings

    def build_contexts(
        self,
        gold_contexts: list[list[str]],
        distractor_contexts: list[list[str]],
    ) -> list[list[str]]:
        contexts, _ = self.build_contexts_with_attention_labels(
            gold_contexts,
            distractor_contexts,
        )
        return contexts

    def build_contexts_with_attention_labels(
        self,
        gold_contexts: list[list[str]],
        distractor_contexts: list[list[str]],
    ) -> tuple[list[list[str]], list[list[int]]]:
        if len(gold_contexts) != len(distractor_contexts):
            raise ValueError(
                "gold_contexts and distractor_contexts must have the same length."
            )
        rng = random.Random(get_global_seed())
        context_groups = []
        attention_labels = []
        for gold_context, distractors in zip(gold_contexts, distractor_contexts):
            labeled_contexts = [
                (context, 1) for context in gold_context
            ] + [
                (context, 0) for context in distractors
            ]
            rng.shuffle(labeled_contexts)
            context_groups.append([context for context, _ in labeled_contexts])
            attention_labels.append([label for _, label in labeled_contexts])
        return context_groups, attention_labels

    def build_training_contexts(self) -> list[list[str]]:
        return self.build_contexts(
            self.dataset.train_ds["gold_context"],
            self.dataset.train_ds["distractor"],
        )

    def embed_contexts(
        self,
        context_groups=None,
        cache_path: Path | None = None,
        split_name: str = "training",
    ):
        if context_groups is None:
            context_groups = self.build_training_contexts()
        empty_rows = [
            index for index, contexts in enumerate(context_groups) if not contexts
        ]
        if empty_rows:
            raise ValueError(
                "Each training row must contain at least one gold/distractor "
                f"context; empty row indices: {empty_rows[:10]}"
            )
        if not context_groups:
            return []

        flat_contexts = [
            context for contexts in context_groups for context in contexts
        ]
        if cache_path is None:
            cache_path = self._resolve_cache_path(
                self.info["training"]["embedding_cache_path"]
            )
        cached = self._load_cache(cache_path, len(context_groups))
        if cached is not None:
            print(f"Loaded {split_name} embeddings from {cache_path}")
            return cached

        embeddings = self._embed_batches(
            flat_contexts, desc=f"Embedding {split_name} contexts"
        )

        grouped_embeddings = []
        offset = 0
        for contexts in context_groups:
            next_offset = offset + len(contexts)
            grouped_embeddings.append(embeddings[offset:next_offset])
            offset = next_offset

        self._save_cache(cache_path, grouped_embeddings)
        print(f"Saved {split_name} embeddings to {cache_path}")
        return grouped_embeddings

    def _load_cache(self, path: Path, expected_rows: int):
        if not path.exists():
            return None
        cached = np.load(path, allow_pickle=True).item()
        embeddings = cached.get("embeddings")
        if (
            cached.get("model_name") == self.embd_model.model_name
            and cached.get("embedding_dim") == self.embd_model.dim
            and isinstance(embeddings, list)
            and len(embeddings) == expected_rows
        ):
            return embeddings
        print(f"Ignoring stale embedding cache: {path}")
        return None

    def _save_cache(self, path: Path, embeddings):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            path,
            {
                "model_name": self.embd_model.model_name,
                "embedding_dim": self.embd_model.dim,
                "seed": get_global_seed(),
                "embeddings": embeddings,
            },
            allow_pickle=True,
        )

    def _embed_batches(self, texts: list[str], desc: str) -> list[list[float]]:
        embeddings = []
        for start in tqdm(range(0, len(texts), 128), desc=desc, unit="batch"):
            batch = texts[start:start + 128]
            batch_embeddings = self.embd_model.embed(batch)
            expected_shape = (len(batch), self.embd_model.dim)
            if batch_embeddings.shape != expected_shape:
                raise ValueError(
                    f"Embedding model returned shape {batch_embeddings.shape}; "
                    f"expected {expected_shape}."
                )
            embeddings.extend(batch_embeddings.tolist())
        return embeddings

    # Compatibility with callers using the old, overly specific name.
    def embed_gold_contexts(self, context_groups=None):
        return self.embed_contexts(context_groups)

    def tokenize(
        self,
        dataset: dict,
        split_name: str = "training",
    ) -> HFDataset:
        required_fields = (
            "query",
            "training_context",
            "answer",
            self.context_embedding_field,
            self.query_embedding_field,
        )
        missing_fields = [field for field in required_fields if field not in dataset]
        if missing_fields:
            raise ValueError(
                f"Training dataset is missing required fields: {missing_fields}"
            )
        field_lengths = {field: len(dataset[field]) for field in required_fields}
        if len(set(field_lengths.values())) != 1:
            raise ValueError(f"Training dataset fields are misaligned: {field_lengths}")

        dataset = HFDataset.from_dict(
            {field: dataset[field] for field in required_fields}
        )
        max_length = self.info["tokenization"]["max_length"]
        eos_token_id = self.tokenizer.eos_token_id
        raw_sequence_lengths = []
        removed_token_counts = []

        def tokenize_batch(batch):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": build_final_prompt(
                            context="\n".join(training_context),
                            question=query,
                        ),
                    }],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for query, training_context in zip(
                    batch["query"], batch["training_context"]
                )
            ]
            prompt_tokens = self.tokenizer(
                prompts, padding=False, add_special_tokens=False
            )
            answer_tokens = self.tokenizer(
                batch["answer"], padding=False, add_special_tokens=False
            )

            input_ids = []
            attention_mask = []
            labels = []
            for prompt_ids, answer_ids in zip(
                prompt_tokens["input_ids"], answer_tokens["input_ids"]
            ):
                answer_ids = answer_ids + [eos_token_id]
                raw_length = len(prompt_ids) + len(answer_ids)
                removed_count = max(0, raw_length - max_length)
                raw_sequence_lengths.append(raw_length)
                removed_token_counts.append(removed_count)
                prompt_ids = prompt_ids[:max_length - len(answer_ids)]
                ids = prompt_ids + answer_ids
                input_ids.append(ids)
                attention_mask.append([1] * len(ids))
                labels.append([-100] * len(prompt_ids) + answer_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                self.context_embedding_field: batch[self.context_embedding_field],
                self.query_embedding_field: batch[self.query_embedding_field],
            }

        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name} dataset",
        )
        truncated_counts = [count for count in removed_token_counts if count > 0]
        tokenization_stats = {
            "total_rows": len(raw_sequence_lengths),
            "truncated_rows": len(truncated_counts),
            "truncated_ratio": (
                len(truncated_counts) / len(raw_sequence_lengths)
                if raw_sequence_lengths else 0.0
            ),
            "max_raw_sequence_length": max(raw_sequence_lengths, default=0),
            "total_removed_tokens": sum(truncated_counts),
            "max_removed_tokens": max(truncated_counts, default=0),
        }
        if split_name == "training":
            self.tokenization_stats = tokenization_stats
        else:
            self.validation_tokenization_stats = tokenization_stats
        print(f"{split_name.capitalize()} tokenization truncation stats:")
        for key, value in tokenization_stats.items():
            print(f"  {key}: {value}")
        return tokenized_dataset

    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        """Keep token ids instead of full-vocabulary logits during validation."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_validation_metrics(self, eval_prediction):
        """Compute answer token F1 from teacher-forced causal-LM predictions."""
        predictions, labels = eval_prediction
        scores = []
        for prediction_ids, label_ids in zip(predictions, labels):
            # A causal LM's token at position t predicts the label at t + 1.
            shifted_predictions = prediction_ids[:-1]
            shifted_labels = label_ids[1:]
            answer_mask = shifted_labels != -100
            predicted_answer_ids = shifted_predictions[answer_mask]
            gold_answer_ids = shifted_labels[answer_mask]

            predicted_answer = self.tokenizer.decode(
                predicted_answer_ids,
                skip_special_tokens=True,
            ).lower()
            gold_answer = self.tokenizer.decode(
                gold_answer_ids,
                skip_special_tokens=True,
            ).lower()
            scores.append(token_f1_score(predicted_answer, gold_answer))

        return {"token_f1": sum(scores) / len(scores) if scores else 0.0}
