from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    get_chunk_overlap,
    get_chunk_size,
    get_chunking_method,
)


def implement_chunking(source_path, encoding="utf-8"):
    raw_text = load_rawtext(source_path, encoding)
    chunks = chunking_rawtext(
        raw_text,
        get_chunking_method(),
        get_chunk_size(),
        get_chunk_overlap(),
    )
    print(f"Chunking completed with {len(chunks)} chunks")
    return chunks


def load_rawtext(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as file:
        return file.read()


def chunking_rawtext(raw_text, method, chunk_size=100, chunk_overlap=0):
    if method == "PLC":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
    elif method == "FLC":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif method == "RCC":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "、", " ", ""],
        )
    else:
        raise ValueError(f"Unsupported chunking method: {method}")
    return splitter.split_text(raw_text)
