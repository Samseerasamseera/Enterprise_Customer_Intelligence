import os


KNOWLEDGE_BASE_PATH = "data/knowledge_base"


def load_documents():
    """
    Load all text documents from the knowledge base.
    """

    documents = []

    for filename in os.listdir(KNOWLEDGE_BASE_PATH):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                KNOWLEDGE_BASE_PATH,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            documents.append(
                {
                    "source": filename,
                    "text": text
                }
            )

    return documents


def chunk_text(
    text,
    chunk_size=500,
    chunk_overlap=100
):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def create_document_chunks():

    documents = load_documents()

    all_chunks = []

    for document in documents:

        chunks = chunk_text(
            document["text"]
        )

        for i, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": i
                }
            )

    return all_chunks


if __name__ == "__main__":

    print("=" * 60)
    print("DOCUMENT LOADER")
    print("=" * 60)

    documents = load_documents()

    print(
        f"Documents loaded: {len(documents)}"
    )

    for document in documents:

        print(
            f"- {document['source']}"
        )

    chunks = create_document_chunks()

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )

    print("\nSample chunk:")
    print("-" * 60)
    print(chunks[0]["text"])
    print("-" * 60)

    print("\nDocument loading completed successfully!")