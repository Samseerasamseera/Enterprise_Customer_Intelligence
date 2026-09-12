import os

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "customer_intelligence"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "openai/gpt-oss-20b"


# ============================================================
# VALIDATE GROQ API KEY
# ============================================================

if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please add GROQ_API_KEY to your .env file."
    )


# ============================================================
# INITIALIZE CLIENTS
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=3
):
    """
    Retrieve the most relevant document chunks
    from ChromaDB.
    """

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_documents = []

    for i in range(
        len(results["documents"][0])
    ):

        retrieved_documents.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "distance": results["distances"][0][i]
            }
        )

    return retrieved_documents


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    retrieved_documents
):
    """
    Generate an answer using the retrieved
    document context.
    """

    context_parts = []

    for document in retrieved_documents:

        context_parts.append(
            f"Source: {document['source']}\n"
            f"{document['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are an enterprise customer intelligence assistant.

Answer the user's question using ONLY the
provided context.

If the answer cannot be found in the context,
say that the information is not available
in the knowledge base.

Do not invent facts.

Keep the answer clear, concise, and business-focused.

Context:
{context}

User Question:
{query}

Answer:
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def ask_question(
    query,
    top_k=3
):
    """
    Complete RAG pipeline:
    Query → Retrieval → Context → LLM Answer
    """

    retrieved_documents = retrieve_documents(
        query,
        top_k=top_k
    )

    answer = generate_answer(
        query,
        retrieved_documents
    )

    return answer, retrieved_documents


# ============================================================
# TEST RAG SYSTEM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ENTERPRISE CUSTOMER INTELLIGENCE - RAG")
    print("=" * 60)

    query = input(
        "\nEnter your question: "
    )

    print(
        "\nSearching knowledge base..."
    )

    answer, documents = ask_question(
        query,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for i, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"{i}. {document['source']} "
            f"(chunk {document['chunk_id']})"
        )

    print("=" * 60)
    print("RAG PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)