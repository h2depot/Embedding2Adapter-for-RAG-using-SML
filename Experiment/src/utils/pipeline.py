from ...data.dataset_constructor import DatasetConstructor


class Pipeline:
    def __init__(self):
        self.dataset_constructor = DatasetConstructor()
        self._base_model_pipeline = None
        self._embd2adapter_pipeline = None

    def use_base_model(self):
        if self._base_model_pipeline is None:
            from ..BaseModel.BaseModel_pipeline import BaseModelPipeline

            self._base_model_pipeline = BaseModelPipeline(self.dataset_constructor)
        return self._base_model_pipeline

    def use_embd2adapter(self, method="mean_embds"):
        if self._embd2adapter_pipeline is None:
            from ..Embd2Adapter.Embd2Adapter_pipeline import Embd2AdapterPipeline

            self._embd2adapter_pipeline = Embd2AdapterPipeline(
                self.dataset_constructor
            )
        self._embd2adapter_pipeline.use_method(method)
        return self._embd2adapter_pipeline

    def _active_embd2adapter(self):
        return self._embd2adapter_pipeline or self.use_embd2adapter()

    def load_trained_hypernet(self, checkpoint_path=None):
        return self._active_embd2adapter().load_trained_hypernet(checkpoint_path)

    def evaluate_hotpotqa(self):
        return self._active_embd2adapter().evaluate_hotpotqa()

    def evaluate_multihoprag(self):
        return self._active_embd2adapter().evaluate_multihoprag()

    def evaluate_musique(self):
        return self._active_embd2adapter().evaluate_musique()

    def evaluate_2wikimultihop(self):
        return self._active_embd2adapter().evaluate_2wikimultihop()

    def experimentHyperNet(self):
        return self._active_embd2adapter().experimentHyperNet()
