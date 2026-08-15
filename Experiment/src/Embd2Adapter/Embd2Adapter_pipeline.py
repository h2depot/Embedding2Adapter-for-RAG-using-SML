import gc
import time
import torch
from tqdm.auto import tqdm

from ...data.dataset_constructor import DatasetConstructor
from ..utils.evaluation import Evaluation
from ..utils.recorder import Recorder
from .embd_model import Embd_Model
from .embd2adapter_vector_store import Embd2Adapter_VectorStore


class Embd2AdapterPipeline:
    def __init__(self, dataset_constructor=None):
        self.dataset_constructor = dataset_constructor or DatasetConstructor()
        self.embd_model = Embd_Model()
        self.vector_store = Embd2Adapter_VectorStore(self.embd_model)
        self._method_name = None
        self._method_pipeline = None
        print("Embd2Adapter Pipline Initialized!")
        print(
            "Experiment Setup Completed. check the datasets whether it is "
            "correct or not, and begin training your model."
        )

    @property
    def hypernet_trainer(self):
        return self.use_method(self._method_name or "mean_embds").trainer

    def use_method(self, method="mean_embds"):
        aliases = {
            "mean_embds": "mean_embds",
            "mean_embeddings": "mean_embds",
            "query_diff_pooling": "query_diff_pooling",
            "query_diff": "query_diff_pooling",
        }
        method_name = aliases.get(method)
        if method_name is None:
            raise ValueError(f"Unknown Embd2Adapter method: {method}")
        if self._method_name == method_name:
            return self._method_pipeline

        self._release_method()

        if method_name == "mean_embds":
            from .Type_MeanEmbeddings.hypernet_meanembds_pipeline import (
                HyperNetMeanEmbdsPipeline,
            )
            method_pipeline = HyperNetMeanEmbdsPipeline
        else:
            from .Type_QueryDiffPooling.hypernet_querydiffpooling_pipeline import (
                HyperNetQueryDiffPoolingPipeline,
            )
            method_pipeline = HyperNetQueryDiffPoolingPipeline

        self._method_pipeline = method_pipeline(
            self.dataset_constructor,
            self.embd_model,
        )
        self._method_name = method_name
        return self._method_pipeline

    def _release_method(self):
        if self._method_pipeline is None:
            return

        method_pipeline = self._method_pipeline
        self._method_pipeline = None
        self._method_name = None
        del method_pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_method(self):
        self._release_method()

    def train(self):
        if not self._method_pipeline:
            raise ValueError("Please define the pipline method first!")
        return self._method_pipeline.train()

    def load_trained_hypernet(self, checkpoint_path=None):
        if not self._method_pipeline:
            raise ValueError("Please define the pipline method first!")
        return self._method_pipeline.load_trained_hypernet(checkpoint_path)

    def _evaluate(self, dataset_name: str, use_rag: bool):
        if not self._method_pipeline:
            raise ValueError("Please define the pipline method first!")
        recorder = Recorder()
        eval_ds = self.dataset_constructor.datasets[dataset_name].eval_ds
        sample_count = len(eval_ds["query"])
        llm_ans = []
        oracle_contexts = self._method_pipeline.trainer.build_contexts(
            eval_ds["gold_context"],
            eval_ds["distractor"],
        )
        evaluation_rows = zip(
            eval_ds["query"],
            eval_ds["full_context"],
            oracle_contexts,
        )
        if use_rag:
            start_time = time.perf_counter()
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
                raise ValueError(f"Evaluation row for {dataset_name} has no contexts.")
            context = "\n".join(contexts)
            context_embeddings = self.embd_model.embed(contexts)
            query_embedding = self.embd_model.embed(query)
            llm_ans.append(
                self._method_pipeline.generate(
                    context,
                    query,
                    context_embeddings,
                    query_embedding,
                )
            )
        if use_rag:
            elapsed_time = time.perf_counter() - start_time
        evaluation = Evaluation(
            llm_ans=llm_ans,
            ds={"answer": list(eval_ds["answer"])},
        )
        result = evaluation.get_results()
        recorder.record(
            experiment="HyperNetTrainer",
            model_name=self._method_pipeline.trainer.model_id,
            dataset=dataset_name,
            rag="Naive RAG" if use_rag else "No RAG",
            em=result["exact_match"],
            f1=result["token_f1"],
            elapsed_time=elapsed_time if use_rag else None,
        )
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
