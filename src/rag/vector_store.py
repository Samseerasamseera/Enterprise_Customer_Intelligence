import chromadb
from sentence_transformers import SentenceTransformer

from src.rag.document_loader import create_document_chunks


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "customer_intelligence"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store():

    print("=" * 60)
    print("CHROMADB VECTOR STORE")
    print("=" * 60)

    # Load document chunks
    chunks = create_document_chunks()

    print(
        f"Chunks received: {len(chunks)}"
    )

    # Load embedding model
    print(
        f"Loading embedding model: {EMBEDDING_MODEL}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Create ChromaDB client
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Prepare data
    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        f"{chunk['source']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    # Generate embeddings
    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    # Store in ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Documents stored: {collection.count()}"
    )

    print(
        f"Vector database: {CHROMA_PATH}"
    )

    print("=" * 60)
    print("VECTOR STORE CREATED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    create_vector_store()