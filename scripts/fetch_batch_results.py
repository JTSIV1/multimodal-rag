import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from together import Together
from dotenv import load_dotenv

load_dotenv()


SCORE_KEYS = [
    "accuracy_score",
    "relevance_score",
    "groundedness_score",
    "faithfulness_score",
]

SCORE_LABELS = {
    "accuracy_score":     "ACCURACY",
    "relevance_score":    "RELEVANCE",
    "groundedness_score": "GROUNDEDNESS",
    "faithfulness_score": "FAITHFULNESS",
}

LABEL_WIDTH = max(len(v) for v in SCORE_LABELS.values()) + 2  # 14

def fetch_output(client: Together, batch_id: str) -> str:
    batch = client.batches.retrieve(batch_id)
    print(f"Batch {batch_id} Status  : {batch.status}")
    if batch.progress is not None:
        print(f"Batch {batch_id} Progress: {batch.progress:.1f}%")

    if batch.status != "COMPLETED":
        if batch.status in ("FAILED", "EXPIRED", "CANCELLED"):
            print(f"Batch {batch_id} did not complete successfully.")
            sys.exit(1)
        print(f"Batch {batch_id} not finished yet — check back later.")
        sys.exit(0)

    if not batch.output_file_id:
        print(f"No output_file_id on completed batch {batch_id}.")
        sys.exit(1)

    print(f"Downloading output file {batch.output_file_id} for batch {batch_id}...")
    response = client.files.content(batch.output_file_id)
    return response.read().decode("utf-8")


def _prompt_manual_scores(custom_id: str, raw_content: str) -> dict | None:
    """Print the raw response and ask the user to enter scores manually."""
    print("\n" + "─" * 60)
    print(f"MALFORMED RESPONSE  —  {custom_id}")
    print("Raw content:")
    print(f"  {raw_content}")
    print()
    inp = input(
        "  Enter scores as accuracy,relevance,groundedness,faithfulness [1-5]\n"
        "  or press Enter to skip: "
    ).strip()

    if not inp:
        print("  Skipped.")
        return None

    try:
        parts = [float(x.strip()) for x in inp.split(",")]
        if len(parts) != 4:
            raise ValueError
        result = dict(zip(SCORE_KEYS, parts))
        return result
    except ValueError:
        print("  Could not parse — skipped.")
        return None


def parse_scores(content: str):
    scores_by_dataset = defaultdict(lambda: defaultdict(list))
    n_errors = n_parse_failures = n_ok = 0

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)

        if record.get("error"):
            n_errors += 1
            print(f"  API error on {record.get('custom_id')}: {record['error']}")
            continue

        custom_id = record.get("custom_id", "")
        parts = custom_id.split("_")
        dataset = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"

        scores = None
        try:
            content_str = record["response"]["body"]["choices"][0]["message"]["content"].strip()
            
            if content_str.startswith("```"):
                content_str = content_str.split("\n", 1)[-1]
                if content_str.endswith("```"):
                    content_str = content_str[:-3].strip()
                    
            scores = json.loads(content_str)
        except (KeyError, IndexError, json.JSONDecodeError):
            raw = ""
            try:
                raw = record["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
            scores = _prompt_manual_scores(custom_id, raw)
            if scores is None:
                n_parse_failures += 1
                continue

        for key in SCORE_KEYS:
            val = scores.get(key)
            if val is not None:
                try:
                    scores_by_dataset[dataset][key].append(float(val))
                except (TypeError, ValueError):
                    pass

        n_ok += 1

    return scores_by_dataset, n_ok, n_errors, n_parse_failures


def _ci(vals: list) -> tuple[float, float]:
    arr = np.array(vals)
    lo, hi = stats.t.interval(0.95, len(arr) - 1, loc=arr.mean(), scale=stats.sem(arr))
    return lo, hi


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _format_overall_line(key: str, vals: list) -> str:
    arr = np.array(vals)
    mean, std = arr.mean(), arr.std()
    lo, hi = _ci(vals)
    label = f"{SCORE_LABELS[key]:<{LABEL_WIDTH}}"
    return f"{label}: {mean:.4f} ± {std:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])"


def _format_dataset_line(key: str, vals: list) -> str:
    arr = np.array(vals)
    label = f"{SCORE_LABELS[key]:<{LABEL_WIDTH}}"
    return f"  {label}: {arr.mean():.4f} ± {arr.std():.4f}"


def build_report(scores_by_dataset: dict, n_ok: int, n_errors: int, n_parse_failures: int) -> str:
    lines = []
    sep = "=" * 41

    lines += [sep, "      LLM-AS-A-JUDGE EVALUATION REPORT", sep, ""]

    all_scores = defaultdict(list)
    for ds in scores_by_dataset.values():
        for key, vals in ds.items():
            all_scores[key].extend(vals)

    total = n_ok + n_errors + n_parse_failures
    lines.append("--- OVERALL METRICS ---")
    lines.append(f"Total Samples: {total}  (ok={n_ok}, errors={n_errors}, skipped={n_parse_failures})")
    for key in SCORE_KEYS:
        vals = all_scores.get(key, [])
        if vals:
            lines.append(_format_overall_line(key, vals))
    lines.append("")

    lines.append("--- METRICS BY DATASET ---")
    for dataset in sorted(scores_by_dataset):
        ds = scores_by_dataset[dataset]
        n = len(next(iter(ds.values()), []))
        lines.append(f"Dataset: {dataset} (N={n})")
        for key in SCORE_KEYS:
            vals = ds.get(key, [])
            if vals:
                lines.append(_format_dataset_line(key, vals))
        lines.append("")

    lines.append("--- SIGNIFICANCE TESTS (Performance Across Datasets) ---")
    lines.append("Using Kruskal-Wallis H-test (non-parametric ANOVA) to test if scores differ by dataset.")

    if len(scores_by_dataset) > 1:
        dataset_order = sorted(scores_by_dataset)
        for key in SCORE_KEYS:
            groups = [np.array(scores_by_dataset[d].get(key, [])) for d in dataset_order]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                stat, p_val = stats.kruskal(*groups)
                sig = _sig_label(p_val)
                label = SCORE_LABELS[key]
                lines.append(f"{label}: H-stat={stat:.4f}, p-value={p_val:.4e} [{sig}]")
        lines.append("(* p<0.05, ** p<0.01, *** p<0.001, ns = not significant)")
    else:
        lines.append("Only one dataset found. Skipping across-dataset significance tests.")

    return "\n".join(lines) + "\n"


def print_report(scores_by_dataset: dict, n_ok: int, n_errors: int, n_parse_failures: int):
    print(build_report(scores_by_dataset, n_ok, n_errors, n_parse_failures))


def save_report(scores_by_dataset: dict, n_ok: int, n_errors: int, n_parse_failures: int, path: str):
    report = build_report(scores_by_dataset, n_ok, n_errors, n_parse_failures)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch and parse Together AI batch evaluation jobs.")
    parser.add_argument("batch_ids", nargs="+", help="One or more Together AI batch job IDs")
    parser.add_argument("--save", "-s", metavar="PATH", help="Save the combined raw output JSONL to this path")
    parser.add_argument("--report", "-r", metavar="PATH", help="Save a formatted .txt evaluation report to this path")
    args = parser.parse_args()

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    all_raw_content = []
    
    for batch_id in args.batch_ids:
        print(f"--- Processing Batch: {batch_id} ---")
        raw_content = fetch_output(client, batch_id)
        all_raw_content.append(raw_content)
        print()

    combined_content = "\n".join(all_raw_content)

    if args.save:
        with open(args.save, "w") as f:
            f.write(combined_content)
        print(f"Combined raw output saved to {args.save}")

    scores_by_dataset, n_ok, n_errors, n_parse_failures = parse_scores(combined_content)

    print_report(scores_by_dataset, n_ok, n_errors, n_parse_failures)

    if args.report:
        save_report(scores_by_dataset, n_ok, n_errors, n_parse_failures, args.report)


if __name__ == "__main__":
    main()
