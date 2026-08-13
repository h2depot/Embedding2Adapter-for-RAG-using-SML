from pathlib import Path

from ...utils.recorder import Recorder
from .hypernet_crossattn_trainer import HyperNetCrossAttentionTrainer


class HyperNetCrossAttentionPipeline:
    def __init__(self, dataset_constructor, embd_model):
        self.embd_model = embd_model
        self.trainer = HyperNetCrossAttentionTrainer(dataset_constructor, embd_model)
        print("Hypernet with CrossAttention Loaded.")

    def train(self):
        self.embd_model.unload()
        log_history, hypernet_spec = self.trainer.train()
        Recorder().record_training_history(
            experiment="HyperNetTrainer-TypeCrossAttention",
            model_name=self.trainer.model_id,
            log_history=log_history,
            hypernet_spec=hypernet_spec,
        )
        return log_history, hypernet_spec

    def load_trained_hypernet(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = (
                Path(self.trainer.info["training"]["output_dir_cross_attn"])
                / "hypernet_state_dict.pt"
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"HyperNet checkpoint was not found: {checkpoint_path}")
        self.trainer.load_trained_hypernet(checkpoint_path)
        print(f"Loaded trained HyperNet from {checkpoint_path}")
        return checkpoint_path

    def generate(self, context, query, context_embeddings, query_embedding):
        return self.trainer.generate_final_model(
            context=context,
            query=query,
            context_embds=context_embeddings,
            query_embd=query_embedding,
        )
