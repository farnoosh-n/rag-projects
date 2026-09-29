# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, LangGraph, FAISS, BM25, Cross-Encoder reranking, and Ollama.

The project also includes a separate evaluation component using Ragas to evaluate the quality of the RAG pipeline.

---

## Project Structure

```text
rag-chatbot/
│
├── rag-chatbot/
│   ├── main.py
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── .gitignore
│   
│
├── ragas-evaluation/
│   ├── ragas_dataset.py
│   ├── test_ragas_qwen.py
│   ├── ragas_evaluation_report.pdf
│   
│
└── README.md
```

## Separate Python Environments

The RAG chatbot and the Ragas evaluation are maintained as two separate components with independent Python environments.

They were separated because the dependency requirements of the two environments were not fully compatible. Keeping them isolated avoids dependency conflicts while allowing the RAG system and its evaluation workflow to be developed independently.

The `ragas-evaluation` component uses a prepared dataset containing questions, generated answers, retrieved contexts, and reference answers for evaluation.

---

# RAG Chatbot

## RAG Pipeline

The chatbot uses a hybrid retrieval and reranking pipeline.

The main components are:

1. PDF document loading
2. Text chunking
3. Hugging Face embeddings
4. FAISS semantic retrieval
5. BM25 keyword retrieval
6. Reciprocal Rank Fusion (RRF)
7. Cross-Encoder reranking
8. Context expansion
9. Qwen 2.5 3B generation through Ollama

The PDF is split into chunks using a chunk size of 1000 characters with an overlap of 200 characters. Each chunk is also assigned a `chunk_id`. 

The project uses `BAAI/bge-base-en-v1.5` for embeddings and `BAAI/bge-reranker-v2-m3` as the Cross-Encoder reranker.

### Hybrid Retrieval

FAISS is initially used to retrieve the top 20 semantic candidates, while BM25 retrieves the top 20 keyword-based candidates.

The results from both retrieval methods are combined using Reciprocal Rank Fusion (RRF), and the top 20 combined candidates are selected for reranking.

### Reranking

The combined candidates are scored using the Cross-Encoder:

```text
BAAI/bge-reranker-v2-m3
```

The top 5 documents after reranking are selected.

### Context Expansion

For the top two reranked chunks, the system also includes the previous and next chunks.

This produces additional surrounding context for the final answer generation.

The remaining top three chunks are included without expansion.

### Generation

The final context is passed to Qwen 2.5 3B through Ollama.

The model is configured with:

```text
Model: qwen2.5:3b
Temperature: 0
```

---

# Ragas Evaluation

The project includes a separate evaluation pipeline based on Ragas.

The evaluation uses:

```text
LLM:        qwen2.5:3b
Embedding:  nomic-embed-text
Provider:   Ollama
Samples:    10
```

The evaluation script independently calculates four metrics for each question:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

The evaluation is performed one sample at a time and the results are then printed as a final summary.

---

# Evaluation Metrics

### 1. Faithfulness

Measures whether the generated answer is supported by the retrieved context.

A higher score indicates that the answer is more consistent with the provided context.

### 2. Answer Relevancy

Measures how relevant the generated answer is to the user's question.

### 3. Context Precision

Measures how relevant the retrieved context is to the question and reference answer.

### 4. Context Recall

Measures whether the retrieved context contains the information needed to answer the question according to the reference answer.

---

# Evaluation Results

The Ragas evaluation was performed on 10 questions.

| Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---:|---:|---:|---:|
| Q1 | 1.000 | 0.971 | 1.000 | 1.000 |
| Q2 | 1.000 | 0.975 | 1.000 | 1.000 |
| Q3 | 0.000 | 0.681 | 0.000 | 0.000 |
| Q4 | 1.000 | 0.000 | 1.000 | 1.000 |
| Q5 | 0.000 | 0.973 | 1.000 | 1.000 |
| Q6 | 0.000 | 0.969 | 1.000 | 1.000 |
| Q7 | 1.000 | 0.000 | 1.000 | 1.000 |
| Q8 | 1.000 | 0.988 | 1.000 | 1.000 |
| Q9 | 0.000 | 0.779 | 1.000 | 1.000 |
| Q10 | 0.000 | 0.000 | 1.000 | 1.000 |

The complete raw evaluation output is available in:

`ragas_evaluation_report.pdf`

---

## Observations

The evaluation results show different behavior across the four metrics.

### Context Recall

Context Recall was `1.0` for all 10 questions.

This indicates that, according to the evaluation, the retrieved contexts contained the information required by the reference answers for all questions.

### Context Precision

Context Precision was approximately `1.0` for 9 of the 10 questions.

Question 3 received a Context Precision score of `0.0`.

### Faithfulness

Faithfulness varied across the questions.

Questions 1, 2, 4, 7, and 8 received a score of `1.0`, while Questions 3, 5, 6, 9, and 10 received a score of `0.0`.

### Answer Relevancy

Answer Relevancy also varied across the questions.

Several questions received high scores, including Q1, Q2, Q5, Q6, and Q8, while Q4, Q7, and Q10 received a score of `0.0`.

---

## Interpreting the Results

The four metrics evaluate different aspects of a RAG system, so they should not be treated as a single overall quality score.

For example, a question can have high Context Recall and Context Precision while receiving a low Faithfulness or Answer Relevancy score.

This can be seen in the evaluation results. For example, Q4 has perfect Context Precision and Context Recall, as well as Faithfulness of `1.0`, but its Answer Relevancy score is `0.0`.

Similarly, Q5 and Q6 have high Context Precision, Context Recall, and Answer Relevancy, but their Faithfulness scores are `0.0`.

These differences show why evaluating both retrieval and generation is useful when analyzing a RAG system.

---

## Possible Sources of Variation

The evaluation uses Qwen 2.5 3B locally as the evaluator LLM.

The variation in Faithfulness and Answer Relevancy may therefore be influenced by how the evaluator model interprets the generated answers, questions, and retrieved contexts.

However, the current results alone are not sufficient to establish this as the definitive cause.

Further experiments with different evaluator models or repeated evaluations would be needed to investigate this possibility.

---

# Evaluation Report

The complete output from the evaluation script is included in:

```text
ragas-evaluation/ragas_evaluation_report.pdf
```

The report contains the evaluation results for all 10 questions and all four Ragas metrics.

---

# Technologies

- Python
- LangChain
- LangGraph
- FAISS
- BM25
- Sentence Transformers
- Cross-Encoder
- Ragas
- Ollama
- Qwen 2.5 3B
- Hugging Face Embeddings

---

# Purpose

This project demonstrates the development of a RAG-based question-answering system together with a separate evaluation workflow.

The project focuses on both sides of a RAG application:

**Retrieval and generation**

and

**Evaluation of retrieval and generated answers**

The evaluation component is kept separate from the main chatbot environment because of dependency compatibility between the two environments.