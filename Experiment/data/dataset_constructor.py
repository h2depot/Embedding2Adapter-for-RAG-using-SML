from .hotpotqa_dataset import HotpotQADataset
from .musique_dataset import MusiqueDataset
from .two_wiki_multihopqa_dataset import TwoWikiMultihopQADataset
from .multihop_rag_dataset import MultiHopRAGDataset
from ..src.utils.config import get_dataset_info, get_random_seed
import random
from pprint import pprint

class DatasetConstructor:
    DATASET_NAME = {
        "hotpotqa/hotpot_qa": HotpotQADataset,
        "awinml/musique": MusiqueDataset,
        "framolfese/2WikiMultihopQA": TwoWikiMultihopQADataset,
        "yixuantt/MultiHopRAG": MultiHopRAGDataset,
    }

    OUTPUT_FIELDS = (
        "query",
        "full_context",
        "gold_context",
        "distractor",
        "answer",
    )

    def __init__(self):
        self.seed = get_random_seed()
        random.seed(self.seed)
        self.datasets = {}
        self.create_dataset()
        self.train_ds=self._empty_dataset()
        self.consolidate_dataset()
        print("Dataset Constructor Initialized!")


    def create_dataset(self):
        for info in get_dataset_info():
            dataset_class = self.DATASET_NAME[info["ds_name"]]
            self.datasets[info["ds_name"]] = dataset_class(info)

    @classmethod
    def _empty_dataset(cls):
        return {field: [] for field in cls.OUTPUT_FIELDS}

    def consolidate_dataset(self):
        rows = [
            {field: dataset.train_ds[field][index] for field in self.OUTPUT_FIELDS}
            for dataset in self.datasets.values()
            for index in range(len(dataset.train_ds["query"]))
        ]
        random.shuffle(rows)

        for field in self.OUTPUT_FIELDS:
            self.train_ds[field] = [row[field] for row in rows]

    def print_top5_data(self):
        datasets = [("TRAIN_DATASET", self.train_ds)]
        datasets.extend(
            (f"EVALUATION_DATASET ({name})", dataset.eval_ds)
            for name, dataset in self.datasets.items()
        )
        for name, dataset in datasets:
            print(f"{name}:")
            for index in range(min(5, len(dataset["query"]))):
                pprint({field: dataset[field][index] for field in self.OUTPUT_FIELDS})
