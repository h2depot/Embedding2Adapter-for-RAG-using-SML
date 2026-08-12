from pathlib import Path

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config" / "default_config.yaml"
NAIVERAG_CONFIG_PATH = EXPERIMENT_DIR / "config" / "naiverag.yaml"
EMBD2ADAPTER_CONFIG_PATH = EXPERIMENT_DIR / "config" / "embd2adapter_config.yaml"


def load_config(path: Path = NAIVERAG_CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_chunking_method():
    return load_config()["chunking"]["method"]


def get_chunk_size():
    return load_config()["chunking"]["chunk_size"]


def get_chunk_overlap():
    return load_config()["chunking"]["chunk_overlap"]


def get_ollama_host():
    return load_config()["ollama"]["host"]


def get_embedding_model():
    return load_config()["embedding"]["model_name"]


def get_embedding_model_dim():
    return load_config()["embedding"]["dimension"]


def get_retrieval_k():
    return load_config()["retrieval"]["top_k"]


def get_generating_model():
    return load_config()["llm"]["model_name"]


def get_generating_options():
    return load_config()["llm"]["options"]


def get_dataset_info():
    return load_config(DEFAULT_CONFIG_PATH)["dataset"]


def get_random_seed():
    return int(load_config(DEFAULT_CONFIG_PATH)["seed"])


def get_hypernet_info():
    return load_config(EMBD2ADAPTER_CONFIG_PATH)["hypernet"]
