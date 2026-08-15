from pathlib import Path
from types import MethodType

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ...utils.config import get_global_seed
from ...utils.prompt import build_final_prompt
from ..hypernet_trainer import HyperNetTrainer
from .hypernet_meanembds_models import (
    HyperNetController_MeanEmbds,
    HyperNetWrapper_MeanEmbds,
)


class HyperNetMeanEmbdsDataCollator:
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
        for feature in features:
            feature.pop("query_embedding", None)
        context_embeddings = torch.tensor(
            [feature.pop("context_embedding") for feature in features],
            dtype=torch.float32,
        )
        batch = self.token_collator(features)
        batch["embedding"] = context_embeddings
        return batch


class HyperNetMeanEmbdsTrainer(HyperNetTrainer):
    def __init__(self, dataset, embd_model):
        super().__init__(dataset=dataset, embd_model=embd_model)
        hypernetwork_info = self.info["hypernetwork"]
        self.hypernet = HyperNetController_MeanEmbds(
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

        print("HyperNetMeanEmbdsTrainer Initialized!")

    def wrap_model(self, model):
        model.hypernet = self.hypernet
        for layer_id, layer in enumerate(model.model.layers):
            if not isinstance(layer, HyperNetWrapper_MeanEmbds):
                model.model.layers[layer_id] = HyperNetWrapper_MeanEmbds(
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
        self.hypernet.requires_grad_(True)

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
            output_dir=training_info["output_dir_mean_embds"],
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
            data_collator=HyperNetMeanEmbdsDataCollator(self.tokenizer),
            processing_class=self.tokenizer,
            compute_metrics=self.compute_validation_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
        )
        self.trainer.train()
        self.trainer.save_model()
        output_dir = Path(training_info["output_dir_mean_embds"])
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

    def generate_final_model(
        self,
        context: str | None,
        query: str,
        embedding=None,
    ) -> str:
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

        if embedding is None:
            embedding = self.embd_model.embed(
                context if context is not None else query
            )
        embedding = torch.as_tensor(
            embedding, dtype=torch.float32, device=self.device
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
                **self.info["generation"],
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
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
