from __future__ import annotations

import hashlib
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
        cache_path = Path(self.info["training"]["embedding_cache_path"])
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
        dataset[self.query_embedding_field] = self.embed_queries(dataset["query"])
        return self.tokenize(dataset, split_name=split_name)

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed each training query and keep it separate from context vectors."""
        if not queries:
            return []

        batch_size = 128
        embeddings = []
        for start in tqdm(
            range(0, len(queries), batch_size),
            desc="Embedding training queries",
            unit="batch",
        ):
            batch_queries = queries[start:start + batch_size]
            batch_embeddings = self.embd_model.embed(batch_queries)
            expected_shape = (len(batch_queries), self.embd_model.dim)
            if batch_embeddings.shape != expected_shape:
                raise ValueError(
                    f"Embedding model returned shape {batch_embeddings.shape}; "
                    f"expected {expected_shape}."
                )
            embeddings.extend(batch_embeddings.tolist())
        return embeddings

    def build_contexts(
        self,
        gold_contexts: list[list[str]],
        distractor_contexts: list[list[str]],
    ) -> list[list[str]]:
        if len(gold_contexts) != len(distractor_contexts):
            raise ValueError(
                "gold_contexts and distractor_contexts must have the same length."
            )
        rng = random.Random(get_global_seed())
        context_groups = []
        for gold_context, distractors in zip(gold_contexts, distractor_contexts):
            contexts = list(gold_context) + list(distractors)
            rng.shuffle(contexts)
            context_groups.append(contexts)
        return context_groups

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
        fingerprint = self._embedding_cache_fingerprint(
            context_groups, flat_contexts
        )
        if cache_path is None:
            cache_path = Path(self.info["training"]["embedding_cache_path"])
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True).item()
            if cached.get("fingerprint") == fingerprint:
                embeddings = cached.get("embeddings")
                if isinstance(embeddings, list):
                    print(f"Loaded {split_name} embeddings from {cache_path}")
                    return embeddings
            print(f"Ignoring stale embedding cache: {cache_path}")

        batch_size = 128
        embeddings = []
        for start in tqdm(
            range(0, len(flat_contexts), batch_size),
            desc=f"Embedding {split_name} contexts",
            unit="batch",
        ):
            batch_contexts = flat_contexts[start:start + batch_size]
            batch_embeddings = self.embd_model.embed(batch_contexts)
            expected_shape = (len(batch_contexts), self.embd_model.dim)
            if batch_embeddings.shape != expected_shape:
                raise ValueError(
                    f"Embedding model returned shape {batch_embeddings.shape}; "
                    f"expected {expected_shape}."
                )
            embeddings.extend(batch_embeddings.tolist())

        grouped_embeddings = []
        offset = 0
        for contexts in context_groups:
            next_offset = offset + len(contexts)
            grouped_embeddings.append(embeddings[offset:next_offset])
            offset = next_offset

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            cache_path,
            {
                "fingerprint": fingerprint,
                "model_name": self.embd_model.model_name,
                "embedding_dim": self.embd_model.dim,
                "seed": get_global_seed(),
                "embeddings": grouped_embeddings,
            },
            allow_pickle=True,
        )
        print(f"Saved {split_name} embeddings to {cache_path}")
        return grouped_embeddings

    # Compatibility with callers using the old, overly specific name.
    def embed_gold_contexts(self, context_groups=None):
        return self.embed_contexts(context_groups)

    def _embedding_cache_fingerprint(self, context_groups, flat_contexts):
        digest = hashlib.sha256()
        digest.update(self.embd_model.model_name.encode("utf-8"))
        digest.update(str(self.embd_model.dim).encode("ascii"))
        digest.update(str(get_global_seed()).encode("ascii"))
        digest.update(str([len(group) for group in context_groups]).encode("ascii"))
        for context in flat_contexts:
            encoded = context.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

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
