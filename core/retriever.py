"""
core/retriever.py
─────────────────────────────────────────────────────────────────────────────
RAG Retrieval Engine
Builds and queries a ChromaDB vector store from the knowledge base documents
(KB articles + resolved cases).

First-time setup: run  python core/retriever.py  to index all documents.
Subsequent runs:  the retriever loads the persisted index instantly.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = str(BASE_DIR / "vector_store" / "chroma_db")
KB_DIR           = str(BASE_DIR / "data" / "knowledge_base")
CASES_DIR        = str(BASE_DIR / "data" / "resolved_cases")

# ─── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")

# ─── Chunking config ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 600   # Larger chunks = more context per result
CHUNK_OVERLAP = 80    # Overlap prevents splitting mid-sentence


def _load_all_documents():
    """Load all Markdown files from KB and resolved_cases directories."""
    docs = []

    for directory in [KB_DIR, CASES_DIR]:
        if not Path(directory).exists():
            logger.warning("Directory not found, skipping: %s", directory)
            continue
        loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        loaded = loader.load()
        logger.info("Loaded %d documents from %s", len(loaded), directory)
        docs.extend(loaded)

    return docs


def build_vector_store() -> None:
    """
    Ingest all knowledge base documents into ChromaDB.
    Run this once before starting the API server.
    """
    logger.info("Building vector store from knowledge base...")
    docs = _load_all_documents()

    if not docs:
        raise RuntimeError(
            "No documents found. Add .md files to data/knowledge_base/ or data/resolved_cases/"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Created %d chunks from %d documents.", len(chunks), len(docs))

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    )
    vectorstore.persist()
    logger.info("✅ Vector store built and persisted at: %s", VECTOR_STORE_PATH)


def get_retriever(k: int = 5):
    """
    Load the persisted ChromaDB vector store and return a LangChain retriever.

    Args:
        k: Number of top documents to retrieve per query.

    Returns:
        A LangChain VectorStoreRetriever.
    """
    if not Path(VECTOR_STORE_PATH).exists():
        raise RuntimeError(
            "Vector store not found. Run  python core/retriever.py  first."
        )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_docs(query: str, k: int = 5) -> list[str]:
    """
    Convenience function: retrieve top-k document chunks for a query.

    Args:
        query: The condensed Dasher problem statement.
        k:     Number of results to return.

    Returns:
        List of document content strings.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    logger.info("Retrieved %d chunks for query: %s...", len(docs), query[:60])
    return [doc.page_content for doc in docs]


# ─── Run directly to build the index ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_vector_store()
    print("\n✅ Vector store ready. You can now start the API server.")
