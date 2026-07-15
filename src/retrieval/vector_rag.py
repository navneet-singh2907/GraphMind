"""Generic semantic retrieval for GraphMind."""

from __future__ import annotations

import json
import logging
import os
import warnings

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.ingestion.loader import load_and_chunk
from src.utils.config import EMBEDDING_MODEL, LLM_MODEL, NEBIUS_API_KEY, NEBIUS_BASE_URL
from src.utils.nebius_embeddings import NebiusEmbeddings


logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "graphmind"
DATA_DIR = "data"
PROCESSED_VECTOR_PATH = os.path.join(DATA_DIR, "processed", "vector_documents.jsonl")
DEFAULT_VECTOR_BATCH_SIZE = 64

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are GraphMind, a helpful assistant for a connected knowledge base.
Answer using only the context below. Name relevant entities and cite source titles when useful.
If the context is insufficient, say: "I couldn't find this in the indexed documents."

Context:
{context}

Question: {question}

Answer:"""
)


def _get_embeddings():
    return NebiusEmbeddings()


def _get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=NEBIUS_API_KEY,
        openai_api_base=NEBIUS_BASE_URL,
        temperature=0.2,
    )


def _source_display_name(source_path: str) -> str:
    name = os.path.splitext(os.path.basename(source_path))[0]
    return name or source_path


def _chroma_value(value):
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_processed_vector_documents() -> list[Document]:
    if not os.path.exists(PROCESSED_VECTOR_PATH):
        return []

    documents = []
    with open(PROCESSED_VECTOR_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = {
                key: _chroma_value(value)
                for key, value in record.items()
                if key not in {"content", "text", "metadata"} and value is not None
            }
            for key, value in record.get("metadata", {}).items():
                if value is not None:
                    metadata[f"meta_{key}"] = _chroma_value(value)
            metadata["source"] = record.get("source_uri") or record.get("source_path", "unknown")
            documents.append(
                Document(page_content=record.get("content") or record.get("text", ""), metadata=metadata)
            )
    return documents


def _document_ids(documents: list[Document]) -> list[str]:
    return [str(doc.metadata.get("id") or f"doc_{index:06d}") for index, doc in enumerate(documents, 1)]


def build_vector_store(batch_size: int = DEFAULT_VECTOR_BATCH_SIZE):
    chunks = _load_processed_vector_documents()

    if not chunks:
        raw_dir = os.path.join(DATA_DIR, "raw")
        for dirpath, _, filenames in os.walk(raw_dir):
            for filename in filenames:
                if filename.lower().endswith((".md", ".txt", ".rst")):
                    filepath = os.path.join(dirpath, filename)
                    loaded = load_and_chunk(filepath)
                    for chunk in loaded:
                        chunk.metadata.update(
                            {
                                "source": os.path.relpath(filepath, raw_dir),
                                "title": os.path.splitext(filename)[0],
                                "source_type": os.path.splitext(filename)[1].lstrip("."),
                                "collection": os.path.relpath(dirpath, raw_dir),
                            }
                        )
                    chunks.extend(loaded)

    print(f"  Total chunks: {len(chunks)} - embedding with {EMBEDDING_MODEL}...")
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
    )
    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
    )
    ids = _document_ids(chunks)
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        vectorstore.add_documents(chunks[start:end], ids=ids[start:end])
        print(f"  Embedded {end}/{len(chunks)} documents")
    return vectorstore


def load_vector_store():
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
    )


def semantic_search(question: str, k: int = 8, filters: dict | None = None) -> list[Document]:
    search_kwargs: dict = {"k": k}
    if filters:
        search_kwargs["filter"] = filters
    return load_vector_store().as_retriever(search_kwargs=search_kwargs).invoke(question)


def answer_question_vector(question: str) -> dict:
    try:
        documents = semantic_search(question)
    except Exception as exc:
        return {
            "question": question,
            "answer": f"Semantic retrieval is unavailable: {exc}",
            "sources": [],
        }

    context_blocks = []
    sources: dict[str, dict] = {}
    for document in documents:
        source = document.metadata.get("source") or document.metadata.get("source_uri", "unknown")
        title = document.metadata.get("title") or _source_display_name(source)
        collection = document.metadata.get("collection", "")
        context_blocks.append(f"Source: {title} ({source})\n{document.page_content}")
        sources.setdefault(
            source,
            {
                "source": source,
                "title": title,
                "collection": collection,
                "source_type": document.metadata.get("source_type", ""),
            },
        )

    answer = (ANSWER_PROMPT | _get_llm() | StrOutputParser()).invoke(
        {"context": "\n\n".join(context_blocks), "question": question}
    )
    return {"question": question, "answer": answer, "sources": list(sources.values())}
