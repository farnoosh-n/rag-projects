from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


# =========================================================
# 1. LLM
# =========================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# =========================================================
# 2. LOAD PDF
# =========================================================

PDF_PATH = "Elsevior_paper.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded pages: {len(documents)}")


# =========================================================
# 3. SPLIT DOCUMENTS INTO CHUNKS
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Created chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

# =========================================================
# 4. CREATE EMBEDDINGS AND RERANKER
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)



# =========================================================
# 5. CREATE FAISS VECTOR STORE
# =========================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created.")


# =========================================================
# 6. CREATE BM25 INDEX
# =========================================================

tokenized_chunks = [
    document.page_content.lower().split()
    for document in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created.")

# =========================================================
# 7. GRAPH STATE
# =========================================================

class GraphState(TypedDict):
    original_question: str
    question: str
    documents: list
    relevant: str
    generation: str
    retry_count: int


# =========================================================
# 8. RETRIEVE NODE
#    FAISS Top 20 -> Reranker -> Top 5
# =========================================================

def retrieve(state: GraphState):
    question = state["question"]


    print("\n")
    print("=" * 50)
    print("RETRIEVE")
    print("=" * 50)

    print("Search query:")
    print(question)

    # Step 1: FAISS retrieves a larger candidate set.
    # FAISS is fast and is used for candidate retrieval.
    faiss_results = vectorstore.similarity_search(
        question,
        k=20
    )

    for rank, document in enumerate(faiss_results, start=1):

        chunk_id = document.metadata["chunk_id"]

    # STEP 2: BM25 KEYWORD SEARCH
    tokenized_query = question.lower().split()

    bm25_results = bm25.get_top_n(
        tokenized_query,
        chunks,
        n=20
    )

    for rank, document in enumerate(bm25_results, start=1):
        chunk_id = document.metadata["chunk_id"]

    # STEP 3: CHECK WHETHER ABSTRACT CHUNK WAS FOUND
    faiss_ids = {
    document.metadata["chunk_id"]
    for document in faiss_results
    }

    bm25_ids = {
        document.metadata["chunk_id"]
        for document in bm25_results
    }

    print("\n")
    print("=" * 50)
    print("RETRIEVAL ANALYSIS")
    print("=" * 50)

    print(f"FAISS Chunk IDs:")
    print(sorted(faiss_ids))

    print(f"\nBM25 Chunk IDs:")
    print(sorted(bm25_ids))

    print("\nIs Chunk 1 retrieved?")

    print(f"FAISS: {1 in faiss_ids}")
    print(f"BM25 : {1 in bm25_ids}")

    
# =========================================================
# 9. RRF HYBRID RETRIEVAL
# =========================================================

    rrf_scores = {}

    k = 60

    # FAISS ranks
    for rank, document in enumerate(faiss_results, start=1):
        chunk_id = document.metadata["chunk_id"]

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += 1 / (k + rank)


    # BM25 ranks
    for rank, document in enumerate(bm25_results, start=1):
        chunk_id = document.metadata["chunk_id"]

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += 1 / (k + rank)


    # Sort by RRF score
    ranked_chunk_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    # Select top 20
    hybrid_ids = ranked_chunk_ids[:20]


    # Convert IDs back to Documents
    chunk_lookup = {
        chunk.metadata["chunk_id"]: chunk
        for chunk in chunks
    }

    combined_documents = [
        chunk_lookup[chunk_id]
        for chunk_id in hybrid_ids
    ]

# =========================================================
# 10. CROSS-ENCODER RERANKING
# =========================================================

    pairs = [
        [question, document.page_content]
        for document in combined_documents
    ]

    reranker_scores = reranker.predict(pairs)

    reranked_results = list(
        zip(combined_documents, reranker_scores)
    )

    reranked_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_documents = [
        document
        for document, score in reranked_results[:5]
    ]

# =========================================================
# 11. CONTEXT EXPANSION
# Top 1 and Top 2 -> Previous + Current + Next
# Top 3, Top 4, Top 5 -> Current only
# =========================================================

    context_documents = []

    for rank, document in enumerate(top_documents):

        chunk_id = document.metadata["chunk_id"]

        # Top 1 and Top 2
        if rank < 2:

            previous_chunk_id = chunk_id - 1
            next_chunk_id = chunk_id + 1

            # Previous chunk
            if previous_chunk_id in chunk_lookup:
                context_documents.append(
                    chunk_lookup[previous_chunk_id]
                )

            # Current chunk
            context_documents.append(document)

            # Next chunk
            if next_chunk_id in chunk_lookup:
                context_documents.append(
                    chunk_lookup[next_chunk_id]
                )

        # Top 3, Top 4, Top 5
        else:

            context_documents.append(document)

    return {
    "documents": context_documents
    }
    

# =========================================================
# 12. RUN RETRIEVAL TEST
# =========================================================

user_question = input("\nEnter your question: ")

state = {
    "question": user_question
}

result = retrieve(state)


# =========================================================
# 13. FINAL RETRIEVED DOCUMENTS
# =========================================================

for rank, document in enumerate(
    result["documents"],
    start=1
):

    chunk_id = document.metadata["chunk_id"]

# =========================================================
# 14. FINAL RETRIEVED DOCUMENTS
# =========================================================

context = "\n\n".join(
    document.page_content
    for document in result["documents"]
)
prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{user_question}

Answer:
"""

response = llm.invoke(prompt)

print("\n")
print("=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)

print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)
