from .dataset import Dataset
import random


class MusiqueDataset(Dataset):
    def __init__(self, info):
        super().__init__(info)
        self.extract_dataset()
        self.split_dataset()
        print("Dataset: Musique Initialized!")

    def extract_dataset(self):
        if self.dataset is None:
            raise ValueError(">>>Dataset is not loaded>>>")
        for row in self.dataset:
            full_context = self.get_full_context(row)
            self.extracted_ds["query"].append(row["question"])
            self.extracted_ds["full_context"].append(full_context)
            self.extracted_ds["gold_context"].append(self.get_gold_context(row))
            self.extracted_ds["distractor"].append(self.get_distractor(row))
            self.extracted_ds["answer"].append(row["answer"])

    def get_full_context(self, row) -> str:
        return "\n\n".join(
            paragraph["paragraph_text"]
            for paragraph in row["paragraphs"]
            if paragraph["paragraph_text"] and paragraph["paragraph_text"].strip()
        )

    def get_gold_context(self, row) -> list[str]:
        if not row["answerable"]:
            return self.get_distractor(row)
        return [
            paragraph["paragraph_text"]
            for paragraph in row["paragraphs"]
            if paragraph["is_supporting"]
        ]

    def get_distractor(self, row) -> list[str]:
        candidates = [
            paragraph["paragraph_text"]
            for paragraph in row["paragraphs"]
            if not paragraph["is_supporting"]
        ]
        if not candidates:
            return []
        return random.sample(candidates, k=min(self.RETRIEVAL_K, len(candidates)))
