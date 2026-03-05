#!/usr/bin/env python3
"""
ingest.py - Document Ingestion Script

Reads all .txt and .md files from ./data recursively, chunks them,
creates embeddings using sentence-transformers, and stores in ChromaDB.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

COLLECTION_NAME = "documents"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def find_documents(data_dir: Path) -> list[Path]:
    """Recursively find all .txt and .md files in the data directory."""
    documents = []
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print("Please create it and add some .txt or .md files.")
        sys.exit(1)
    
    for ext in ["*.txt", "*.md"]:
        documents.extend(data_dir.rglob(ext))
    
    return sorted(documents)


def main():
    print("=" * 60)
    print("Document Ingestion Script")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Chroma DB path: {CHROMA_DB_PATH}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
    print()

    # Find documents
    documents = find_documents(DATA_DIR)
    if not documents:
        print("No .txt or .md files found in the data directory.")
        print("Please add some documents and run again.")
        sys.exit(1)
    
    print(f"Found {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc.relative_to(DATA_DIR)}")
    print()

    # Load embedding model
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.\n")

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created collection '{COLLECTION_NAME}'.\n")

    # Process documents
    total_chunks = 0
    all_chunks = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []

    for doc_path in documents:
        try:
            content = doc_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read {doc_path}: {e}")
            continue
        
        chunks = chunk_text(content)
        if not chunks:
            continue
        
        # Relative path for metadata
        rel_path = str(doc_path.relative_to(DATA_DIR))
        
        print(f"Processing '{rel_path}': {len(chunks)} chunk(s)")
        
        # Generate embeddings
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{rel_path}::chunk_{i}"
            all_chunks.append(chunk)
            all_embeddings.append(embedding.tolist())
            all_metadatas.append({
                "source": rel_path,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            all_ids.append(chunk_id)
            total_chunks += 1

    # Add to collection
    if all_chunks:
        print(f"\nAdding {total_chunks} chunks to ChromaDB...")
        collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print("Done!")

    print("\n" + "=" * 60)
    print(f"Ingestion complete: {total_chunks} chunks indexed from {len(documents)} file(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
