from pathlib import Path

from tqdm.auto import tqdm

from ...data.dataset_constructor import DatasetConstructor
from .Type_MeanEmbedding.hypernet_trainer import HyperNetTrainer
from ..utils.evaluation import Evaluation
from ..utils.recorder import Recorder
from .embd_model import Embd_Model
from .embd2adapter_vector_store import Embd2Adapter_VectorStore

class Embd2AdapterPipeline:
    def __init__(self):
        self.dataset_constructor = DatasetConstructor()
        self.embd_model = Embd_Model()
        self.vector_store = Embd2Adapter_VectorStore(self.embd_model)
        self.hypernet_trainer = HyperNetTrainer(
            self.dataset_constructor,
            self.embd_model,
        )
        print("Embd2Adapter Pipline Initialized!")
        print("Experiment Setup Completed. check the datasets whether it is correct or not, and begin training your model.")

    def train_TypeMeanEmbd(self):
        self.embd_model.unload()
        recorder = Recorder()
        log_history = self.hypernet_trainer.train()
        recorder.record_training_history(
            experiment="HyperNetTrainer-TypeMeanEmbedding",
            model_name=self.hypernet_trainer.model_id,
            log_history=log_history,
        )
        return log_history

    def load_trained_hypernet(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = (
                Path(self.hypernet_trainer.info["training"]["output_dir"])
                / "hypernet_state_dict.pt"
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"HyperNet checkpoint was not found: {checkpoint_path}"
            )
        self.hypernet_trainer.load_trained_hypernet(checkpoint_path)
        print(f"Loaded trained HyperNet from {checkpoint_path}")
        return checkpoint_path

    def _evaluate(self, dataset_name: str, use_rag: bool):
        recorder = Recorder()
        eval_ds = self.dataset_constructor.datasets[dataset_name].eval_ds
        sample_count = len(eval_ds["query"])
        llm_ans = []
        oracle_contexts = self.hypernet_trainer.build_contexts(
            eval_ds["gold_context"],
            eval_ds["distractor"],
        )
        evaluation_rows = zip(
            eval_ds["query"],
            eval_ds["full_context"],
            oracle_contexts,
        )
        for query, full_context, oracle_context in tqdm(
            evaluation_rows,
            total=sample_count,
            desc=f"Evaluating {dataset_name} ({'RAG' if use_rag else 'No RAG'})",
            unit="question",
        ):
            if use_rag:
                self.vector_store.set_chunks(full_context)
                search_results = self.vector_store.search_query(query)
                contexts = [result["chunk"] for result in search_results]
            else:
                contexts = oracle_context
            if not contexts:
                raise ValueError(
                    f"Evaluation row for {dataset_name} has no contexts."
                )
            context = "\n".join(contexts)
            embedding = self.embd_model.embed(contexts)
            llm_ans.append(
                self.hypernet_trainer.generate_final_model(
                    context=context,
                    query=query,
                    embedding=embedding,
                )
            )
        evaluation = Evaluation(
            llm_ans=llm_ans,
            ds={"answer": list(eval_ds["answer"])},
        )
        result = evaluation.get_results()
        recorder.record(experiment="HyperNetTrainer", model_name=self.hypernet_trainer.model_id, dataset=dataset_name, rag="Naive RAG" if use_rag else "No RAG", em = result["exact_match"], f1=result["token_f1"])
        return result

    def _evaluate_both(self, dataset_name: str):
        return {
            "rag": self._evaluate(dataset_name, use_rag=True),
            "no_rag": self._evaluate(dataset_name, use_rag=False),
        }

    def evaluate_hotpotqa(self):
        return self._evaluate_both("hotpotqa/hotpot_qa")

    def evaluate_multihoprag(self):
        return self._evaluate_both("yixuantt/MultiHopRAG")

    def evaluate_musique(self):
        return self._evaluate_both("awinml/musique")

    def evaluate_2wikimultihop(self):
        return self._evaluate_both("framolfese/2WikiMultihopQA")

    def experimentHyperNet(self):
        return {
            "hotpotqa": self.evaluate_hotpotqa(),
            "multihoprag": self.evaluate_multihoprag(),
            "musique": self.evaluate_musique(),
            "2wikimultihop": self.evaluate_2wikimultihop(),
        }
