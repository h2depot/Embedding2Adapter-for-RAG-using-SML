from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_chunk_overlap, get_chunk_size


def implement_chunking(source_path, encoding="utf-8"):
    raw_text = load_rawtext(source_path, encoding)
    chunks = chunking_rawtext(
        raw_text,
        get_chunk_size(),
        get_chunk_overlap(),
    )
    print(f"Chunking completed with {len(chunks)} chunks")
    return chunks


def load_rawtext(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as file:
        return file.read()


def chunking_rawtext(raw_text, chunk_size=100, chunk_overlap=0):
    """Split text with the experiment's fixed-length chunking strategy."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(raw_text)
