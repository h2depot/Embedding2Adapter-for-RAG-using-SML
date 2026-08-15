import json
from pathlib import Path
from types import MethodType

import torch
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ..hypernet_trainer import HyperNetTrainer
from ...utils.config import get_global_seed
from ...utils.prompt import build_final_prompt
from .hypernet_crossattn_models import HyperNetController_CrossAttention, HyperNetWrapper_CrossAttention


class AttentionPretrainDataset:
    """Pair each query/context row with a deranged random context row."""

    def __init__(self, dataset, max_samples: int | None = None, seed: int = 33):
        sample_count = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        if sample_count < 2:
            raise ValueError("Attention pretraining requires at least two rows.")
        generator = torch.Generator().manual_seed(seed)
        self.dataset = dataset
        self.indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
        self.negative_indices = self.indices[1:] + self.indices[:1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]], self.dataset[self.negative_indices[index]]


class HyperNetCrossAttnCollator:
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
        context_embeddings = torch.tensor(
            [feature.pop("context_embedding") for feature in features],
            dtype=torch.float32,
        )
        query_embeddings = torch.tensor(
            [feature.pop("query_embedding") for feature in features],
            dtype=torch.float32,
        )
        attention_labels = torch.tensor(
            [feature.pop("attention_labels") for feature in features],
            dtype=torch.float32,
        )
        attention_loss_mask = torch.tensor(
            [feature.pop("attention_loss_mask") for feature in features],
            dtype=torch.float32,
        )
        batch = self.token_collator(features)
        batch["context_embds"] = context_embeddings
        batch["query_embd"] = query_embeddings
        batch["attention_labels"] = attention_labels
        batch["attention_loss_mask"] = attention_loss_mask
        return batch

class HyperNetCrossAttentionTrainer(HyperNetTrainer):

    def __init__(self, dataset, embd_model):
        super().__init__(dataset=dataset, embd_model=embd_model)
        hypernetwork_info = self.info["hypernetwork"]
        self.hypernet = HyperNetController_CrossAttention(
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
        print("HyperNetCrossAttentionTrainer Initialized!")

    def evaluate_attention(self, gold_mask, beta: float = 3.0):
        """Create a soft target distribution from binary gold labels."""
        gold_mask = torch.as_tensor(
            gold_mask,
            dtype=torch.float32,
            device=self.device,
        )
        if gold_mask.ndim not in (1, 2):
            raise ValueError("gold_mask must have shape (chunks,) or (batch, chunks).")
        if gold_mask.shape[-1] < 1:
            raise ValueError("gold_mask must contain at least one chunk.")
        if not torch.all((gold_mask == 0) | (gold_mask == 1)):
            raise ValueError("gold_mask must contain only binary values 0 and 1.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")

        return torch.softmax(beta * gold_mask, dim=-1)

    def _prepare_dataset(self, dataset, cache_path: Path, split_name: str):
        tokenized_dataset = super()._prepare_dataset(
            dataset,
            cache_path=cache_path,
            split_name=split_name,
        )
        _, gold_masks = self.build_contexts_with_attention_labels(
            dataset["gold_context"],
            dataset["distractor"],
        )
        attention_labels = [
            self.evaluate_attention(gold_mask).cpu().tolist()
            for gold_mask in gold_masks
        ]
        tokenized_dataset = tokenized_dataset.add_column(
            "attention_labels",
            attention_labels,
        )
        return tokenized_dataset.add_column(
            "attention_loss_mask",
            [float(any(gold_mask)) for gold_mask in gold_masks],
        )

    def wrap_model(self, model):
        model.hypernet = self.hypernet
        for layer_id, layer in enumerate(model.model.layers):
            if not isinstance(layer, HyperNetWrapper_CrossAttention):
                model.model.layers[layer_id] = HyperNetWrapper_CrossAttention(
                    original_layer=layer,
                    hypernet=self.hypernet,
                    layer_id=layer_id,
                )

        original_forward = model.forward
        attention_loss_weight = float(
            self.info["training"]["attention_loss_weight"]
        )

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
            context_embds=None,
            query_embd=None,
            **kwargs,
        ):
            attention_labels = kwargs.pop("attention_labels", None)
            attention_loss_mask = kwargs.pop("attention_loss_mask", None)
            if context_embds is None or query_embd is None:
                raise ValueError(
                    "Context and query embeddings must be passed to the model forward call."
                )
            hypernet_dtype = next(model_self.hypernet.parameters()).dtype
            target_device = (
                input_ids.device if input_ids is not None else inputs_embeds.device
            )
            context_embds = context_embds.to(
                device=target_device, dtype=hypernet_dtype
            )
            query_embd = query_embd.to(
                device=target_device, dtype=hypernet_dtype
            )
            pool = model_self.hypernet.embds_pool
            attention_score_modules = (
                pool.query_projector,
                pool.context_projector,
                pool.self_attention,
                pool.ffn_norm,
                pool.ffn,
                pool.gate,
            )
            attention_is_trainable = any(
                parameter.requires_grad
                for module in attention_score_modules
                for parameter in module.parameters()
            )
            if attention_labels is not None and attention_is_trainable:
                pooled_embedding, cross_attn_weights = (
                    model_self.hypernet.pool_embeddings(
                        context_embds,
                        query_embd,
                        return_attention_weights=True,
                    )
                )
            else:
                pooled_embedding = model_self.hypernet.pool_embeddings(
                    context_embds,
                    query_embd,
                    return_attention_weights=False,
                )
                cross_attn_weights = None
            outputs = original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                context_embds=context_embds,
                query_embd=query_embd,
                pooled_embedding=pooled_embedding,
                **kwargs,
            )
            if attention_labels is not None and attention_is_trainable:
                if attention_loss_mask is None:
                    raise ValueError(
                        "attention_loss_mask is required with attention_labels."
                    )
                predicted_attention = (
                    cross_attn_weights.float().mean(dim=1).squeeze(1)
                )
                predicted_attention = predicted_attention / (
                    predicted_attention.sum(dim=-1, keepdim=True)
                    .clamp_min(1e-8)
                )
                attention_labels = attention_labels.to(
                    device=predicted_attention.device,
                    dtype=predicted_attention.dtype,
                )
                attention_loss_mask = attention_loss_mask.to(
                    device=predicted_attention.device,
                    dtype=predicted_attention.dtype,
                )
                per_sample_attention_loss = -(
                    attention_labels
                    * predicted_attention.clamp_min(1e-8).log()
                ).sum(dim=-1)
                attention_loss = (
                    per_sample_attention_loss * attention_loss_mask
                ).sum() / attention_loss_mask.sum().clamp_min(1.0)
                outputs.loss = (
                    outputs.loss
                    + attention_loss_weight * attention_loss
                )
            return outputs

        model.forward = MethodType(forward_with_embedding, model)
        return model

    def freeze_base_model(self):
        self.model.requires_grad_(False)
        self.hypernet.requires_grad_(True)

    def train_attention(
        self,
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 8,
        max_samples: int = 10_000,
    ):
        """Pretrain AttentionPooling without running the causal LM."""
        if not hasattr(self, "train_ds") or not hasattr(self, "val_ds"):
            raise RuntimeError("Call prepare_datasets() before train_attention().")
        if num_epochs < 1 or gradient_accumulation_steps < 1:
            raise ValueError("num_epochs and gradient_accumulation_steps must be positive.")
        self.hypernet.requires_grad_(False)
        self.hypernet.embds_pool.requires_grad_(True)
        # Returned MHA weights include attention dropout.  Applying supervised
        # -log(p) to those weights would turn dropped gold positions into an
        # artificial -log(1e-8) penalty, so pretraining must be deterministic.
        self.hypernet.embds_pool.set_attention_dropout(0.0)
        self.hypernet.embds_pool.train()
        optimizer = torch.optim.AdamW(
            self.hypernet.embds_pool.parameters(), lr=learning_rate
        )
        generator = torch.Generator().manual_seed(get_global_seed())
        history = []
        attention_output_dir = Path(
            self.info["training"]["output_dir_cross_attn"]
        )
        attention_output_dir.mkdir(parents=True, exist_ok=True)
        history_path = attention_output_dir / "attention_pretrain_history.json"
        state_path = attention_output_dir / "attention_pretrain_state_dict.pt"
        self.attention_val_ds = AttentionPretrainDataset(
            self.val_ds, seed=get_global_seed() + 10_000
        )
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(num_epochs):
            total_loss = 0.0
            update_count = 0
            accumulated_steps = 0
            self.attention_train_ds = AttentionPretrainDataset(
                self.train_ds, max_samples=max_samples,
                seed=get_global_seed() + epoch,
            )
            indices = torch.randperm(len(self.attention_train_ds), generator=generator)
            progress = tqdm(indices.tolist(), desc=f"Attention pretrain {epoch + 1}/{num_epochs}")
            for step, index in enumerate(progress, start=1):
                row, negative_row = self.attention_train_ds[index]
                loss_mask = float(row["attention_loss_mask"])
                if loss_mask == 0.0:
                    continue
                positive_contexts = torch.as_tensor(
                    row["context_embedding"], device=self.device,
                    dtype=next(self.hypernet.parameters()).dtype,
                )
                negative_contexts = torch.as_tensor(
                    negative_row["context_embedding"], device=self.device,
                    dtype=positive_contexts.dtype,
                )
                contexts = torch.cat(
                    (positive_contexts, negative_contexts), dim=0
                ).unsqueeze(0)
                query = torch.as_tensor(
                    row["query_embedding"], device=self.device,
                    dtype=contexts.dtype,
                ).unsqueeze(0)
                target = torch.as_tensor(
                    row["attention_labels"], device=self.device,
                    dtype=torch.float32,
                )
                target = torch.cat(
                    (target, torch.zeros(len(negative_contexts), device=self.device))
                )
                _, weights = self.hypernet.pool_embeddings(
                    contexts, query, return_attention_weights=True
                )
                predicted = weights.float().squeeze(0).squeeze(1)
                loss = -(target * predicted.clamp_min(1e-8).log()).sum(dim=-1).mean()
                (loss / gradient_accumulation_steps).backward()
                total_loss += loss.item()
                update_count += 1
                accumulated_steps += 1
                if accumulated_steps == gradient_accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                progress.set_postfix(loss=total_loss / update_count)
            if accumulated_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if update_count == 0:
                raise RuntimeError("No rows with attention_loss_mask=1 were found.")
            validation = self.evaluate_attention_pretraining(self.attention_val_ds)
            epoch_result = {
                "epoch": epoch + 1,
                "attention_loss": total_loss / update_count,
                **validation,
            }
            history.append(epoch_result)
            history_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            torch.save(self.hypernet.embds_pool.state_dict(), state_path)
            tqdm.write(
                "Attention epoch result:\n"
                + json.dumps(epoch_result, ensure_ascii=False, indent=2)
            )
            self.hypernet.embds_pool.train()
        self.hypernet.embds_pool.eval()
        return history

    def evaluate_attention_pretraining(self, dataset=None):
        """Evaluate the fixed shuffled-context validation task."""
        if not hasattr(self, "val_ds"):
            raise RuntimeError("Call prepare_datasets() before attention evaluation.")
        if dataset is None:
            if not hasattr(self, "attention_val_ds"):
                self.attention_val_ds = AttentionPretrainDataset(
                    self.val_ds, seed=get_global_seed() + 10_000
                )
            dataset = self.attention_val_ds
        pool = self.hypernet.embds_pool
        was_training = pool.training
        pool.eval()
        per_head_losses = []
        mean_head_losses = []
        gold_masses = []
        top1_correct = []
        entropies = []
        self_attention_residual_ratios = []
        with torch.inference_mode():
            for row, negative_row in tqdm(dataset, desc="Attention validation"):
                if float(row["attention_loss_mask"]) == 0.0:
                    continue
                positive_contexts = torch.as_tensor(
                    row["context_embedding"], device=self.device,
                    dtype=next(self.hypernet.parameters()).dtype,
                )
                negative_contexts = torch.as_tensor(
                    negative_row["context_embedding"], device=self.device,
                    dtype=positive_contexts.dtype,
                )
                contexts = torch.cat(
                    (positive_contexts, negative_contexts), dim=0
                ).unsqueeze(0)
                query = torch.as_tensor(
                    row["query_embedding"], device=self.device,
                    dtype=positive_contexts.dtype,
                ).unsqueeze(0)
                original_target = torch.as_tensor(
                    row["attention_labels"], device=self.device,
                    dtype=torch.float32,
                )
                target = torch.cat((
                    original_target,
                    torch.zeros(len(negative_contexts), device=self.device),
                ))
                if hasattr(pool, "self_attention"):
                    projected_contexts = pool.context_projector(
                        positive_contexts.unsqueeze(0)
                    )
                    refined_contexts = pool.self_attention(projected_contexts)
                    self_attention_residual_ratios.append(
                        ((refined_contexts - projected_contexts).float().norm()
                         / projected_contexts.float().norm().clamp_min(1e-8)).item()
                    )
                _, weights = self.hypernet.pool_embeddings(
                    contexts, query, return_attention_weights=True
                )
                per_head = weights.float().squeeze(0).squeeze(1)
                predicted = per_head.mean(dim=0)
                per_head_losses.append(
                    -(target * per_head.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
                )
                mean_head_losses.append(
                    -(target * predicted.clamp_min(1e-8).log()).sum().item()
                )
                gold_mask = torch.zeros_like(target, dtype=torch.bool)
                gold_mask[:len(original_target)] = torch.isclose(
                    original_target, original_target.max()
                )
                gold_masses.append(predicted[gold_mask].sum().item())
                top1_correct.append(float(gold_mask[predicted.argmax()]))
                entropies.append(
                    (-(predicted.clamp_min(1e-8) * predicted.clamp_min(1e-8).log()).sum()
                     / torch.log(torch.tensor(predicted.numel(), device=self.device))).item()
                )
        if was_training:
            pool.train()
        if not per_head_losses:
            raise RuntimeError("No valid rows were found in the attention validation dataset.")
        metrics = {
            "val_attention_loss_per_head": sum(per_head_losses) / len(per_head_losses),
            "val_attention_loss_head_mean": sum(mean_head_losses) / len(mean_head_losses),
            "val_gold_mass": sum(gold_masses) / len(gold_masses),
            "val_top1_accuracy": sum(top1_correct) / len(top1_correct),
            "val_attention_entropy": sum(entropies) / len(entropies),
        }
        if self_attention_residual_ratios:
            metrics["val_self_attention_residual_ratio"] = (
                sum(self_attention_residual_ratios)
                / len(self_attention_residual_ratios)
            )
        return metrics

    def freeze_attention(self):
        """Freeze attention scoring, but train the pooled value path."""
        self.hypernet.requires_grad_(True)
        pool = self.hypernet.embds_pool
        pool.requires_grad_(False)
        pool.value_projector.requires_grad_(True)
        pool.output_norm.requires_grad_(True)
        pool.set_attention_dropout(0.0)
        pool.eval()

    def train(self):
        if not hasattr(self, "train_ds") or not hasattr(self, "val_ds"):
            raise RuntimeError("Call prepare_datasets() before train().")
        training_info = self.info["training"]
        random_seed = get_global_seed()
        if training_info["per_device_train_batch_size"] != 1:
            raise ValueError(
                "Variable-length context embeddings currently require "
                "per_device_train_batch_size=1."
            )
        training_args = TrainingArguments(
            output_dir=training_info["output_dir_cross_attn"],
            num_train_epochs=training_info["num_train_epochs"],
            per_device_train_batch_size=training_info[
                "per_device_train_batch_size"
            ],
            per_device_eval_batch_size=1,
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
            save_steps=training_info["save_steps"],
            eval_strategy=training_info["eval_strategy"],
            eval_steps=training_info["eval_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="token_f1",
            greater_is_better=True,
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
            eval_dataset=self.val_ds,
            data_collator=HyperNetCrossAttnCollator(self.tokenizer),
            processing_class=self.tokenizer,
            compute_metrics=self.compute_validation_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
        )
        self.trainer.train()
        self.trainer.save_model()
        output_dir = Path(training_info["output_dir_cross_attn"])
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.hypernet.state_dict(),
            output_dir / "hypernet_state_dict.pt",
        )
        hypernet_spec = self.calc_hypernet_spec()
        return self.trainer.state.log_history, hypernet_spec

    def calc_hypernet_spec(self):
        num_params = sum(p.numel() for p in self.hypernet.parameters())
        trainable_params = sum(
            p.numel() for p in self.hypernet.parameters() if p.requires_grad
        )
        model_size_bytes = sum(
            p.numel() * p.element_size()
            for p in self.hypernet.parameters()
        )
        print(f"Parameters: {num_params:,}")
        print(f"Trainable:  {trainable_params:,}")
        print(f"Size:       {model_size_bytes / 1024**2:.2f} MiB")
        return {
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
            "parameter_size_bytes": model_size_bytes,
            "parameter_size_mib": model_size_bytes / 1024**2,
        }
    
    def generate_final_model(self, context, query, context_embds=None, query_embd=None) -> str:
        self.model.eval()
        prompt = (
            build_final_prompt(context=context, question=query)
            if context is not None else query
        )
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=self.info["tokenization"]["max_length"],
        ).to(self.device)

        context_embds = torch.as_tensor(
            context_embds, dtype=torch.float32, device=self.device
        )
        query_embd = torch.as_tensor(
            query_embd, dtype=torch.float32, device=self.device
        )
        if context_embds.ndim == 2:
            context_embds = context_embds.unsqueeze(0)
        if query_embd.ndim == 1:
            query_embd = query_embd.unsqueeze(0)
        expected_dim = self.embd_model.dim
        if (
            context_embds.ndim != 3
            or context_embds.shape[0] != 1
            or context_embds.shape[1] < 1
            or context_embds.shape[2] != expected_dim
        ):
            raise ValueError(
                "context_embds must have shape "
                f"(n_contexts, {expected_dim}) or "
                f"(1, n_contexts, {expected_dim}), with n_contexts >= 1."
            )
        if query_embd.shape != (1, expected_dim):
            raise ValueError(
                f"query_embd must have shape ({expected_dim},) or "
                f"(1, {expected_dim})."
            )

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                context_embds=context_embds,
                query_embd=query_embd,
                **self.info["generation"],
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()

    def generate_hypernet(self, context_embds, query_embd, layer_id:int):
        with torch.no_grad():
            return self.hypernet(context_embds, query_embd, layer_id)

    def load_trained_hypernet(self, checkpoint_path):
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.hypernet.load_state_dict(state_dict)
        self.hypernet.eval()
        self.model.eval()
