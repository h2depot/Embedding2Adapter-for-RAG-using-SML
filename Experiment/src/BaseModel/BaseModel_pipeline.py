import random
import time

from tqdm.auto import tqdm

from ...data.dataset_constructor import DatasetConstructor
from ..utils.config import get_global_seed
from ..utils.prompt import build_final_prompt
from ..utils.evaluation import Evaluation
from ..utils.recorder import Recorder
from .llm import LLM
from .vector_store import VectorStore


class BaseModelPipeline:
    def __init__(self, dataset_constructor=None):
        self.dataset_constructor = dataset_constructor or DatasetConstructor()
        self.llm = LLM()
        self.vector_store = VectorStore()

    def _evaluate(self, dataset_name: str, use_rag: bool):
        recorder = Recorder()
        eval_ds = self.dataset_constructor.datasets[dataset_name].eval_ds
        rng = random.Random(get_global_seed())
        llm_ans = []
        oracle_contexts = []
        for gold_context, distractors in zip(
            eval_ds["gold_context"],
            eval_ds["distractor"],
        ):
            contexts = list(gold_context) + list(distractors)
            rng.shuffle(contexts)
            oracle_contexts.append(contexts)
        evaluation_rows = zip(
            eval_ds["query"],
            eval_ds["full_context"],
            oracle_contexts,
        )
        if use_rag:
            start_time = time.perf_counter()
        for query, full_context, oracle_context in tqdm(
            evaluation_rows,
            total=len(eval_ds["query"]),
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
            llm_ans.append(
                self.llm.generate(
                    build_final_prompt(
                        context=context,
                        question=query,
                    )
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
            experiment="BaseModel",
            model_name=self.llm.model_name,
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

    def experimentBaseModel(self):
        return {
            "hotpotqa": self.evaluate_hotpotqa(),
            "multihoprag": self.evaluate_multihoprag(),
            "musique": self.evaluate_musique(),
            "2wikimultihop": self.evaluate_2wikimultihop(),
        }
