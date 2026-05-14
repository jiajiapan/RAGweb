"""RAG pipeline evaluation using SQuAD dataset to establish baseline metrics."""

import json
import os
import re
import sys
import tempfile
from collections import Counter
from time import perf_counter

from jinja2 import Template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.propressing import embedding  # noqa: E402
from backend.search import retrieve  # noqa: E402

EVAL_TEMPLATE = Template(
    """\
Instructions: Answer the question based ONLY on the provided Context. \
Give a short, specific answer using exact words from the Context. \
Do not explain or elaborate.

Context:
{% for doc in documents %}
---
{{ doc }}
{% endfor %}
---

Question: {{ query }}
Answer:"""
)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(prediction: str, ground_truths: list[str]) -> float:
    norm_pred = normalize_text(prediction)
    for gt in ground_truths:
        if normalize_text(gt) == norm_pred:
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    norm_pred = normalize_text(prediction)
    pred_tokens = norm_pred.split()

    best_f1 = 0.0
    for gt in ground_truths:
        norm_gt = normalize_text(gt)
        gt_tokens = norm_gt.split()

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(gt_tokens) if gt_tokens else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

    return best_f1


def load_squad(split: str = "validation", max_samples: int = 100):
    from datasets import load_dataset

    squad = load_dataset("squad", split=split)
    if max_samples:
        squad = squad.select(range(min(max_samples, len(squad))))
    return squad


def create_context_pdf(contexts: list[str], output_path: str):
    import fitz

    doc = fitz.open()
    for ctx in contexts:
        page = doc.new_page()
        rect = fitz.Rect(72, 72, 540, 800)
        page.insert_textbox(rect, ctx, fontsize=11)
    doc.save(output_path)
    doc.close()


def run_evaluation(max_samples: int = 100, top_k: int = 5):
    print(f"Loading SQuAD validation set ({max_samples} samples)...")
    squad = load_squad(max_samples=max_samples)

    unique_contexts = list(dict.fromkeys(squad["context"]))
    print(f"Unique contexts: {len(unique_contexts)}")

    pdf_path = os.path.join(tempfile.gettempdir(), "squad_eval.pdf")
    print("Creating PDF from contexts...")
    create_context_pdf(unique_contexts, pdf_path)

    print("Indexing documents into LanceDB...")
    t0 = perf_counter()
    vs_name = embedding(pdf_path)
    index_time = perf_counter() - t0
    print(f"Indexed in {index_time:.1f}s, vector store: {vs_name}")

    from backend.query import query_deepseek

    predictions = []
    ground_truths_list = []
    retrieve_times = []
    llm_times = []

    print(f"\nRunning QA evaluation on {len(squad)} questions...")
    for i, item in enumerate(squad):
        question = item["question"]
        answers = item["answers"]["text"]

        t0 = perf_counter()
        docs = retrieve(vs_name, question, k=25, rerank=True, top_k=top_k)
        retrieve_times.append(perf_counter() - t0)

        prompt = EVAL_TEMPLATE.render(documents=docs, query=question)

        try:
            t0 = perf_counter()
            response = list(query_deepseek(prompt))
            llm_times.append(perf_counter() - t0)
            answer = response[-1] if response else ""
        except Exception as e:
            llm_times.append(0)
            print(f"  [WARN] LLM error on question #{i}: {e}")
            answer = ""

        predictions.append(answer)
        ground_truths_list.append(answers)

        if (i + 1) % 25 == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(squad)}] "
                f"avg retrieve: {sum(retrieve_times[-25:]) / len(retrieve_times[-25:]):.2f}s, "
                f"avg llm: {sum(llm_times[-25:]) / len(llm_times[-25:]):.2f}s"
            )

    em_scores = [compute_exact_match(p, g) for p, g in zip(predictions, ground_truths_list, strict=False)]
    f1_scores = [compute_f1(p, g) for p, g in zip(predictions, ground_truths_list, strict=False)]

    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100
    avg_retrieve_time = sum(retrieve_times) / len(retrieve_times)
    avg_llm_time = sum(llm_times) / len(llm_times)

    results = {
        "dataset": "SQuAD v1.1 (validation)",
        "samples": len(predictions),
        "exact_match_percent": round(avg_em, 2),
        "f1_percent": round(avg_f1, 2),
        "avg_retrieve_time_s": round(avg_retrieve_time, 2),
        "avg_llm_time_s": round(avg_llm_time, 2),
        "total_index_time_s": round(index_time, 1),
        "config": {
            "top_k": top_k,
            "embed_model": os.getenv("EMB_MODEL", "default"),
            "rerank_model": os.getenv("RERANK_MODEL", "default"),
        },
    }

    print(f"\n{'=' * 55}")
    print("RAG BASELINE METRICS — SQuAD v1.1")
    print(f"{'=' * 55}")
    print(f"  Samples:          {results['samples']}")
    print(f"  Exact Match (EM): {results['exact_match_percent']}%")
    print(f"  F1 Score:         {results['f1_percent']}%")
    print(f"  Avg retrieve:     {results['avg_retrieve_time_s']}s")
    print(f"  Avg LLM call:     {results['avg_llm_time_s']}s")
    print(f"  Index time:       {results['total_index_time_s']}s")
    print(f"{'=' * 55}")

    output_file = "baseline_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on SQuAD")
    parser.add_argument("--samples", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks passed to LLM")
    args = parser.parse_args()
    run_evaluation(max_samples=args.samples, top_k=args.top_k)
