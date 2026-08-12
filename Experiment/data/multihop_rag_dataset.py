from .dataset import Dataset
import random


class MultiHopRAGDataset(Dataset):
    def __init__(self, info):
        super().__init__(info)
        self.fact_pool = [
            evidence["fact"]
            for row in self.dataset
            for evidence in row["evidence_list"]
        ]
        self.extract_dataset()
        self.split_dataset()
        print("Dataset: MultiHopRAG Initialized!")

    def extract_dataset(self):
        if self.dataset is None:
            raise ValueError(">>>Dataset is not loaded>>>")
        for row in self.dataset:
            full_context = self.get_full_context(row)
            self.extracted_ds["query"].append(row["query"])
            self.extracted_ds["full_context"].append(full_context)
            gold_context = self.get_gold_context(row)
            self.extracted_ds["gold_context"].append(gold_context)
            self.extracted_ds["distractor"].append(self.get_distractor(gold_context))
            self.extracted_ds["answer"].append(row["answer"])

    def get_full_context(self, row) -> str:
        return "\n\n".join(
            evidence["fact"]
            for evidence in row["evidence_list"]
            if evidence["fact"] and evidence["fact"].strip()
        )

    def get_gold_context(self, row) -> list[str]:
        return [
            evidence["fact"]
            for evidence in row["evidence_list"]
            if evidence["fact"] and evidence["fact"].strip()
        ]

    def get_distractor(self, gold_context) -> list[str]:
        candidates = list(set(self.fact_pool) - set(gold_context))
        if not candidates:
            return []
        return random.sample(candidates, k=min(self.RETRIEVAL_K, len(candidates)))
