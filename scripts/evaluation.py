"""evaluate.py — Compute generation quality metrics on pipeline inference results.

Reads the JSON output from pipeline.py (run_inference) and computes:
  - BLEU (corpus-level, 1-4 grams with Chen & Cherry smoothing)
  - ROUGE-1, ROUGE-2, ROUGE-L  (F1, macro-averaged)
  - BERTScore  (P, R, F1 using microsoft/deberta-xlarge-mnli by default)
  - LLM-as-judge  (GPT-4o via OpenAI API — set OPENAI_API_KEY to enable)
"""

import argparse
import json
import os
import string
import sys
from typing import Any, Dict, List, Optional

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from openai import OpenAI


# ── Text normalisation ──────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


# ── BLEU ───────────────────────────────────────────────────────────────────

def compute_bleu(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Corpus-level BLEU-1 through BLEU-4 with smoothing."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    references = [[_tokenize(r["reference"])] for r in records]
    hypotheses = [_tokenize(r["prediction"]) for r in records]
    smoother = SmoothingFunction().method3

    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        scores[f"bleu_{n}"] = corpus_bleu(
            references, hypotheses, weights=weights, smoothing_function=smoother
        )
    return scores


# ── ROUGE ──────────────────────────────────────────────────────────────────

def compute_rouge(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L — macro-averaged F1."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    accum: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0
    for r in records:
        scores = scorer.score(r["reference"], r["prediction"])
        for key in accum:
            accum[key] += scores[key].fmeasure
        n += 1
    return {k: v / n for k, v in accum.items()} if n else accum


# ── BERTScore ──────────────────────────────────────────────────────────────

def compute_bertscore(
    records: List[Dict[str, Any]],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> Dict[str, float]:
    """Macro-averaged BERTScore P, R, F1.

    Uses DeBERTa-xlarge-mnli by default (top performer on BERTScore leaderboard).
    Pass model_type="distilbert-base-uncased" for a lighter-weight alternative.
    """
    predictions = [r["prediction"] for r in records]
    references = [r["reference"] for r in records]
    P, R, F1 = bert_score_fn(predictions, references, model_type=model_type, verbose=False)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


# ── LLM-as-judge ───────────────────────────────────────────────────────────

_LLM_JUDGE_SYSTEM = """\
You are an expert evaluator for question-answering systems.
You will be given a question, a reference (ground-truth) answer, and a model prediction.
Score the prediction on a scale of 1-5 using the following rubric:

5 — Fully correct and complete; matches the reference in substance.
4 — Mostly correct with minor omissions or slight wording differences.
3 — Partially correct; captures some key information but misses important details.
2 — Mostly incorrect; contains a grain of truth but is largely wrong or irrelevant.
1 — Completely wrong, hallucinated, or refuses to answer.

Respond with ONLY a JSON object: {"score": <int>, "reason": "<one sentence>"}
"""

_LLM_JUDGE_USER = """\
Question: {question}

Reference answer: {reference}

Model prediction: {prediction}
"""


def compute_llm_judge(
    records: List[Dict[str, Any]],
    model: str = "gpt-4o",
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """LLM-as-judge scoring via OpenAI API.

    Requires the OPENAI_API_KEY environment variable.

    Args:
        records:      Inference records from pipeline.py.
        model:        OpenAI model ID to use as judge.
        max_examples: Cap the number of examples sent to the judge (cost control).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  Skipping LLM-as-judge: OPENAI_API_KEY not set.")
        return {}

    client = OpenAI(api_key=api_key)
    subset = records if max_examples is None else records[:max_examples]
    per_example = []

    for i, r in enumerate(subset):
        user_msg = _LLM_JUDGE_USER.format(
            question=r["question"],
            reference=r["reference"],
            prediction=r["prediction"],
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            score = int(parsed["score"])
            reason = parsed.get("reason", "")
        except Exception as exc:
            print(f"  LLM judge error on example {r.get('example_index')}: {exc}")
            score = -1
            reason = f"error: {exc}"

        print(f"  [{i + 1}/{len(subset)}] score={score}  {reason[:80]}")
        per_example.append({"example_index": r.get("example_index"), "score": score, "reason": reason})

    valid_scores = [e["score"] for e in per_example if e["score"] >= 1]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else None
    return {"avg_score": avg, "per_example": per_example}


# ── Aggregate metrics for one file ─────────────────────────────────────────

def evaluate_file(
    predictions_path: str,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    llm_judge_model: str = "gpt-4o",
    llm_judge_max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    with open(predictions_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    mode = records[0].get("mode", "unknown") if records else "unknown"
    dataset = records[0].get("dataset", "") if records else ""
    n = len(records)

    print(f"\n{'='*60}")
    print(f"Evaluating: {predictions_path}")
    print(f"  mode={mode}  examples={n}")
    print(f"{'='*60}")

    output: Dict[str, Any] = {
        "source_file": str(predictions_path),
        "mode": mode,
        "dataset": dataset,
        "num_examples": n,
    }

    print("\n[BLEU]")
    bleu = compute_bleu(records)
    for k, v in bleu.items():
        print(f"  {k}: {v:.4f}")
    output["bleu"] = bleu

    print("\n[ROUGE]")
    rouge = compute_rouge(records)
    for k, v in rouge.items():
        print(f"  {k}: {v:.4f}")
    output["rouge"] = rouge

    print("\n[BERTScore]")
    bscore = compute_bertscore(records, model_type=bertscore_model)
    for k, v in bscore.items():
        print(f"  {k}: {v:.4f}")
    output["bertscore"] = bscore

    print("\n[LLM-as-judge]")
    judge = compute_llm_judge(records, model=llm_judge_model, max_examples=llm_judge_max_examples)
    if judge and judge.get("avg_score") is not None:
        print(f"  avg_score: {judge['avg_score']:.3f}")
    output["llm_judge"] = judge

    return output


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate REAL-MM-RAG pipeline predictions (BLEU, ROUGE, BERTScore, LLM-judge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "predictions",
        nargs="+",
        help="One or more prediction JSON files from pipeline.py run_inference().",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the metrics summary JSON.",
    )
    parser.add_argument(
        "--bertscore-model",
        default="microsoft/deberta-xlarge-mnli",
        help="Model type for BERTScore. Use 'distilbert-base-uncased' to save memory.",
    )
    parser.add_argument(
        "--llm-judge-model",
        default="gpt-4o",
        help="OpenAI model for LLM-as-judge (default: gpt-4o).",
    )
    parser.add_argument(
        "--llm-judge-max-examples",
        type=int,
        default=None,
        help="Limit LLM-judge to first N examples per file (cost control).",
    )
    args = parser.parse_args()

    all_results = []
    for path in args.predictions:
        if not os.path.exists(path):
            print(f"File not found: {path}", file=sys.stderr)
            continue
        result = evaluate_file(
            predictions_path=path,
            bertscore_model=args.bertscore_model,
            llm_judge_model=args.llm_judge_model,
            llm_judge_max_examples=args.llm_judge_max_examples,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        header = f"{'Mode':<20} {'BLEU-4':>8} {'ROUGE-L':>8} {'BERTScore':>10} {'LLM-judge':>10}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            mode = r["mode"]
            bleu4 = r["bleu"].get("bleu_4") if r.get("bleu") else None
            rougeL = r["rouge"].get("rougeL") if r.get("rouge") else None
            bs_f1 = r["bertscore"].get("bertscore_f1") if r.get("bertscore") else None
            judge_avg = r["llm_judge"].get("avg_score") if r.get("llm_judge") else None

            def _fmt(v: Optional[float]) -> str:
                return f"{v:.4f}" if v is not None else "   N/A  "

            print(f"{mode:<20} {_fmt(bleu4):>8} {_fmt(rougeL):>8} {_fmt(bs_f1):>10} {_fmt(judge_avg):>10}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nMetrics saved → {args.output}")


if __name__ == "__main__":
    main()
