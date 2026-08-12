from .dataset import Dataset
import random


class TwoWikiMultihopQADataset(Dataset):
    def __init__(self, info):
        super().__init__(info)
        self.extract_dataset()
        self.split_dataset()
        print("Dataset: TwoWikiMultihopQA Initialized!")

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
            sentence
            for sentences in row["context"]["sentences"]
            for sentence in sentences
            if sentence and sentence.strip()
        )

    def get_gold_context(self, row) -> list[str]:
        context = dict(zip(row["context"]["title"], row["context"]["sentences"]))
        supporting_facts = zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
        )
        return [context[title][sent_id] for title, sent_id in supporting_facts]

    def get_distractor(self, row) -> list[str]:
        supporting_facts = set(zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
        ))
        candidates = [
            sentence
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
            for sent_id, sentence in enumerate(sentences)
            if (title, sent_id) not in supporting_facts
        ]
        if not candidates:
            return []
        return random.sample(candidates, k=min(self.RETRIEVAL_K, len(candidates)))
