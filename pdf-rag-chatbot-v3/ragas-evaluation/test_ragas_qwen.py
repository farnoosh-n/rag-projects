import asyncio

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas_dataset import dataset


# ============================================================
# 1. Ollama OpenAI-compatible client
# ============================================================

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)


# ============================================================
# 2. Qwen evaluator
# ============================================================

evaluator_llm = llm_factory(
    "qwen2.5:3b",
    provider="openai",
    client=client,
)


# ============================================================
# 3. Local embedding
# ============================================================

evaluator_embeddings = embedding_factory(
    provider="openai",
    model="nomic-embed-text",
    client=client,
)


# ============================================================
# 4. Metrics
# ============================================================

faithfulness = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

context_precision = ContextPrecision(
    llm=evaluator_llm
)

context_recall = ContextRecall(
    llm=evaluator_llm
)


# ============================================================
# 5. Evaluate one sample
# ============================================================

async def evaluate_sample(sample, index):

    question = sample["question"]
    answer = sample["answer"]
    contexts = sample["contexts"]
    ground_truth = sample["ground_truth"]

    print("\n" + "=" * 70)
    print(f"QUESTION {index}")
    print("=" * 70)

    print("Question:")
    print(question)

    print("\nEvaluating Faithfulness...")

    faithfulness_score = await faithfulness.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    print("Faithfulness:", faithfulness_score)

    print("\nEvaluating Answer Relevancy...")

    answer_relevancy_score = await answer_relevancy.ascore(
        user_input=question,
        response=answer,
    )

    print("Answer Relevancy:", answer_relevancy_score)

    print("\nEvaluating Context Precision...")

    context_precision_score = await context_precision.ascore(
        user_input=question,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )

    print("Context Precision:", context_precision_score)

    print("\nEvaluating Context Recall...")

    context_recall_score = await context_recall.ascore(
        user_input=question,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )

    print("Context Recall:", context_recall_score)

    return {
        "question": question,
        "faithfulness": faithfulness_score,
        "answer_relevancy": answer_relevancy_score,
        "context_precision": context_precision_score,
        "context_recall": context_recall_score,
    }


# ============================================================
# 6. Main
# ============================================================

async def main():

    print("\n" + "=" * 70)
    print("STARTING RAGAS EVALUATION WITH QWEN")
    print("=" * 70)

    print(f"Number of samples: {len(dataset)}")
    print("LLM       : qwen2.5:3b")
    print("Embedding : nomic-embed-text")
    print("Provider  : Ollama")
    print()

    results = []

    for i, sample in enumerate(dataset, start=1):

        result = await evaluate_sample(
            sample,
            i
        )

        results.append(result)

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\nQuestion {i}")
        print(f"Faithfulness      : {result['faithfulness']}")
        print(f"Answer Relevancy  : {result['answer_relevancy']}")
        print(f"Context Precision : {result['context_precision']}")
        print(f"Context Recall    : {result['context_recall']}")


if __name__ == "__main__":
    asyncio.run(main())