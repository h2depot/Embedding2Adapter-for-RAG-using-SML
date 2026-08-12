from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset
import hashlib
from pathlib import Path
import random
import torch
from types import MethodType
from tqdm.auto import tqdm
import numpy as np

from ....data.dataset_constructor import DatasetConstructor
from ...utils.config import get_hypernet_info, get_random_seed
from ...utils.prompt import build_final_prompt
from ..embd_model import Embd_Model
from .hypernet_models import HyperNetWrapper, HyperNetController


class HyperNetDataCollator:
    def __init__(self, tokenizer):
        self.token_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError(
                "Variable-length context embeddings currently require "
                "per_device_train_batch_size=1."
            )
        embeddings = torch.tensor(
            [feature.pop("embedding") for feature in features],
            dtype=torch.float32,
        )
        batch = self.token_collator(features)
        batch["embedding"] = embeddings
        return batch


class HyperNetTrainer:
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
        hypernetwork_info = self.info["hypernetwork"]
        self.hypernet = HyperNetController(
            device=self.device,
            embd_dim=self.embd_model.dim,
            projected_embd_dim=hypernetwork_info["projected_task_embedding_dim"],
            input_dim=self.model.config.hidden_size,
            reduction_factor=hypernetwork_info["reduction_factor"],
            task_hidden_dim=hypernetwork_info["task_hidden_dim"],
            num_layers=self.model.config.num_hidden_layers,
        ).to(device=self.device, dtype=next(self.model.parameters()).dtype)
        self.model = self.wrap_model(self.model)
        self.freeze_base_model()

        print("HyperNetTrainer Initialized!")
        print("(To train the hypernet, first call prepare_datasets() to embed the train datasets.")
        print("Then, call train() and the hypernet model training starts!)")


    def prepare_datasets(self):
        training_contexts = self.build_training_contexts()
        self.dataset.train_ds["training_context"] = training_contexts
        self.dataset.train_ds["embedding"] = self.embed_gold_contexts(
            training_contexts
        )
        self.train_ds = self.tokenize(self.dataset.train_ds)       


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

    def wrap_model(self, model):
        model.hypernet = self.hypernet
        for layer_id, layer in enumerate(model.model.layers):
            if not isinstance(layer, HyperNetWrapper):
                model.model.layers[layer_id] = HyperNetWrapper(
                    original_layer=layer,
                    hypernet=self.hypernet,
                    layer_id=layer_id,
                )

        original_forward = model.forward

        def forward_with_embedding(
            model_self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            logits_to_keep=0,
            embedding=None,
            **kwargs,
        ):
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                embedding=embedding,
                **kwargs,
            )

        model.forward = MethodType(forward_with_embedding, model)
        return model

    def freeze_base_model(self):
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, HyperNetController):
                module.requires_grad_(True)

    def build_contexts(
        self,
        gold_contexts: list[list[str]],
        distractor_contexts: list[list[str]],
    ) -> list[list[str]]:
        if len(gold_contexts) != len(distractor_contexts):
            raise ValueError(
                "gold_contexts and distractor_contexts must have the same length."
            )
        rng = random.Random(get_random_seed())
        context_groups = []
        for gold_context, distractors in zip(
            gold_contexts,
            distractor_contexts,
        ):
            contexts = list(gold_context) + list(distractors)
            rng.shuffle(contexts)
            context_groups.append(contexts)
        return context_groups

    def build_training_contexts(self) -> list[list[str]]:
        return self.build_contexts(
            self.dataset.train_ds["gold_context"],
            self.dataset.train_ds["distractor"],
        )

    def embed_gold_contexts(self, context_groups=None):
        if context_groups is None:
            context_groups = self.build_training_contexts()
        empty_rows = [
            index
            for index, contexts in enumerate(context_groups)
            if not contexts
        ]
        if empty_rows:
            raise ValueError(
                "Each training row must contain at least one gold/distractor "
                f"context; empty row indices: {empty_rows[:10]}"
            )

        if not context_groups:
            return []

        flat_contexts = [
            context
            for contexts in context_groups
            for context in contexts
        ]
        fingerprint = self._embedding_cache_fingerprint(
            context_groups,
            flat_contexts,
        )
        cache_path = Path(
            self.info["training"]["embedding_cache_path"]
        )
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True).item()
            if cached.get("fingerprint") == fingerprint:
                embeddings = cached.get("embeddings")
                if isinstance(embeddings, list):
                    print(f"Loaded training embeddings from {cache_path}")
                    return embeddings
            print(f"Ignoring stale embedding cache: {cache_path}")

        batch_size = 128
        embeddings = []
        for start in tqdm(
            range(0, len(flat_contexts), batch_size),
            desc="Embedding training contexts",
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
                "seed": get_random_seed(),
                "embeddings": grouped_embeddings,
            },
            allow_pickle=True,
        )
        print(f"Saved training embeddings to {cache_path}")
        return grouped_embeddings

    def _embedding_cache_fingerprint(self, context_groups, flat_contexts):
        digest = hashlib.sha256()
        digest.update(self.embd_model.model_name.encode("utf-8"))
        digest.update(str(self.embd_model.dim).encode("ascii"))
        digest.update(str(get_random_seed()).encode("ascii"))
        digest.update(str([len(group) for group in context_groups]).encode("ascii"))
        for context in flat_contexts:
            encoded = context.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def tokenize(self, dataset: dict) -> HFDataset:
        required_fields = (
            "query",
            "training_context",
            "answer",
            "embedding",
        )
        missing_fields = [field for field in required_fields if field not in dataset]
        if missing_fields:
            raise ValueError(
                f"Training dataset is missing required fields: {missing_fields}"
            )
        field_lengths = {
            field: len(dataset[field])
            for field in required_fields
        }
        if len(set(field_lengths.values())) != 1:
            raise ValueError(
                f"Training dataset fields are misaligned: {field_lengths}"
            )

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
                    batch["query"],
                    batch["training_context"],
                )
            ]
            prompt_tokens = self.tokenizer(
                prompts,
                padding=False,
                add_special_tokens=False,
            )
            answer_tokens = self.tokenizer(
                batch["answer"],
                padding=False,
                add_special_tokens=False,
            )

            input_ids = []
            attention_mask = []
            labels = []
            for prompt_ids, answer_ids in zip(
                prompt_tokens["input_ids"], answer_tokens["input_ids"]
            ):
                answer_ids = answer_ids + [eos_token_id]
                raw_sequence_length = len(prompt_ids) + len(answer_ids)
                removed_token_count = max(0, raw_sequence_length - max_length)
                raw_sequence_lengths.append(raw_sequence_length)
                removed_token_counts.append(removed_token_count)
                prompt_ids = prompt_ids[:max_length - len(answer_ids)]
                ids = prompt_ids + answer_ids

                input_ids.append(ids)
                attention_mask.append([1] * len(ids))
                labels.append([-100] * len(prompt_ids) + answer_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "embedding": batch["embedding"],
            }

        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing training dataset",
        )
        truncated_counts = [count for count in removed_token_counts if count > 0]
        self.tokenization_stats = {
            "total_rows": len(raw_sequence_lengths),
            "truncated_rows": len(truncated_counts),
            "truncated_ratio": (
                len(truncated_counts) / len(raw_sequence_lengths)
                if raw_sequence_lengths
                else 0.0
            ),
            "max_raw_sequence_length": max(raw_sequence_lengths, default=0),
            "total_removed_tokens": sum(truncated_counts),
            "max_removed_tokens": max(truncated_counts, default=0),
        }
        print("Tokenization truncation stats:")
        for key, value in self.tokenization_stats.items():
            print(f"  {key}: {value}")
        return tokenized_dataset

    def train(self):
        if not hasattr(self, "train_ds"):
            raise RuntimeError("Call prepare_datasets() before train().")
        training_info = self.info["training"]
        random_seed = get_random_seed()
        if training_info["per_device_train_batch_size"] != 1:
            raise ValueError(
                "Variable-length context embeddings currently require "
                "per_device_train_batch_size=1."
            )
        training_args = TrainingArguments(
            output_dir=training_info["output_dir"],
            num_train_epochs=training_info["num_train_epochs"],
            per_device_train_batch_size=training_info[
                "per_device_train_batch_size"
            ],
            gradient_accumulation_steps=training_info[
                "gradient_accumulation_steps"
            ],
            bf16=training_info.get("bf16", False),
            gradient_checkpointing=training_info.get(
                "gradient_checkpointing", False
            ),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=float(training_info["learning_rate"]),
            warmup_steps=training_info.get("warmup_steps", 0),
            logging_steps=training_info["logging_steps"],
            save_strategy=training_info["save_strategy"],
            save_total_limit=training_info.get("save_total_limit"),
            report_to=training_info["report_to"],
            seed=random_seed,
            data_seed=random_seed,
            remove_unused_columns=False,
            do_train=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_ds,
            data_collator=HyperNetDataCollator(self.tokenizer),
            processing_class=self.tokenizer,
        )
        self.trainer.train()
        self.trainer.save_model()
        output_dir = Path(training_info["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.hypernet.state_dict(),
            output_dir / "hypernet_state_dict.pt",
        )
        return self.trainer.state.log_history

    def generate_final_model(
        self,
        context: str | None,
        query: str,
        embedding=None,
    ) -> str:
        self.model.eval()
        tokenization_info = self.info["tokenization"]
        generation_info = self.info["generation"]

        prompt = (
            build_final_prompt(context=context, question=query)
            if context is not None
            else query
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=tokenization_info["max_length"],
        ).to(self.device)

        if embedding is None:
            embedding = self.embd_model.embed(
                context if context is not None else query
            )
        embedding = torch.as_tensor(
            embedding,
            dtype=torch.float32,
            device=self.device,
        )
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if (
            embedding.ndim != 2
            or embedding.shape[0] < 1
            or embedding.shape[1] != self.embd_model.dim
        ):
            raise ValueError(
                "embedding must have shape "
                f"({self.embd_model.dim},) or (n, {self.embd_model.dim}) "
                "with n >= 1."
            )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                embedding=embedding,
                **generation_info,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()
    

    def generate_hypernet(self, embedding, layer_id: int):
        self.hypernet.eval()
        with torch.no_grad():
            return self.hypernet(embedding, layer_id)

    def load_trained_hypernet(self, checkpoint_path):
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.hypernet.load_state_dict(state_dict)
        self.hypernet.eval()
        self.model.eval()


    
