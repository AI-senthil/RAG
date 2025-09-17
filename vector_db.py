import os
import glob
import uuid
from typing import List, Tuple, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    # Backwards compatibility with older package name
    from PyPDF2 import PdfReader  # type: ignore


DEFAULT_EMBEDDING_MODEL = "dunzhang/stella_en_1.5B_v5"
DEFAULT_COLLECTION_NAME = "Senthil_Profile"
DEFAULT_DB_DIR = "chroma_db"


def get_embedding_function(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return a Chroma-compatible embedding function using SentenceTransformers.

    Notes:
    - This will download and load the model on first use. The specified model is large
      and may require significant RAM and disk space.
    - To control device selection, set the environment variable SENTENCE_TRANSFORMERS_DEVICE
      to "cpu" or a CUDA device like "cuda".
    """

    # Allow users to override device via env var if desired
    # chroma will pass texts -> List[float] to this callable.
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def get_or_create_collection(
    persist_directory: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
):
    """Create (or open) a persistent Chroma collection with the given embedding model."""
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)
    embedding_fn = get_embedding_function(embedding_model_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _read_pdf_texts(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text per page from a PDF file.

    Returns a list of dicts with fields: page_number, text
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(getattr(reader, "pages", [])):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_number": i + 1, "text": text})
    return pages


def _chunk_text(
    text: str,
    max_words: int = 800,
    overlap_words: int = 200,
) -> List[str]:
    """Simple word-based chunking with overlap to balance recall & context length."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def ingest_pdfs_to_chroma(
    pdf_folder_path: str,
    persist_directory: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
) -> Tuple[chromadb.Client, chromadb.Collection]:
    """Walk a folder for PDFs, chunk their text, and store in a persistent Chroma DB.

    - pdf_folder_path: Directory to recursively search for .pdf files
    - Returns: (client, collection)
    """

    client, collection = get_or_create_collection(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )

    pdf_paths = sorted(
        glob.glob(os.path.join(pdf_folder_path, "**", "*.pdf"), recursive=True)
    )
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found under folder: {os.path.abspath(pdf_folder_path)}"
        )

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for pdf_path in pdf_paths:
        page_texts = _read_pdf_texts(pdf_path)
        for page in page_texts:
            page_number = page["page_number"]
            page_text = page["text"].strip()
            if not page_text:
                continue
            chunks = _chunk_text(page_text)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{os.path.basename(pdf_path)}-p{page_number}-c{idx}-{uuid.uuid4().hex[:8]}"
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": os.path.abspath(pdf_path),
                        "page": page_number,
                        "chunk": idx,
                    }
                )
                ids.append(doc_id)

                # Flush in batches to avoid high memory peak
                if len(documents) >= batch_size:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    documents, metadatas, ids = [], [], []

    # Final flush
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return client, collection


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
) -> Dict[str, Any]:
    """Query a Chroma collection and return documents, metadatas, ids, distances."""
    return collection.query(
        query_texts=[query_text],
        n_results=n_results,
        # Note: valid include items per Chroma version may exclude "ids".
        # We request documents, metadatas, and distances which are sufficient for RAG.
        include=["documents", "metadatas", "distances"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a folder of PDFs into a persistent Chroma DB."
    )
    parser.add_argument("pdf_folder", type=str, help="Path to folder containing PDFs")
    parser.add_argument(
        "--db", dest="db_dir", type=str, default=DEFAULT_DB_DIR, help="DB directory"
    )
    parser.add_argument(
        "--name",
        dest="collection_name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Collection name",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name",
    )
    parser.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=128,
        help="Batch size for adding documents",
    )

    args = parser.parse_args()

    print(
        f"Ingesting PDFs from '{args.pdf_folder}' into '{args.db_dir}' collection '{args.collection_name}' using model '{args.model_name}'..."
    )
    ingest_pdfs_to_chroma(
        pdf_folder_path=args.pdf_folder,
        persist_directory=args.db_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.model_name,
        batch_size=args.batch_size,
    )
    print("Done.")
