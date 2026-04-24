"""
Custom Fine-tuning Data Generation Pipeline
============================================
Generates new (question, answer, image_path) triplets from your own PDF
documents using the LOCAL Qwen3-VL-8B model (via the project's VLM class)
as the teacher — no OpenAI API key required.

Output format exactly mirrors the REAL-MM-RAG qas.jsonl produced by
scripts/get_data.py, so MMRAGPipeline.ingest() and run_inference() work
on custom data without any modification.

Output structure (drop into data/raw/ like any other dataset):
    <out_dir>/
        pages/       ← rendered PNG per page
        qas.jsonl    ← (question, answer, image_path, ...) records
        pages.jsonl  ← page metadata

Usage (run from repo root):
    python scripts/generate_custom_data.py \\
        --input_dir  ./my_pdfs \\
        --out_dir    data/raw/CustomDocs/train \\
        --doc_type   tech_slides \\
        --num_questions 3 \\
        --skip_first_pages 1

    # Then plug in exactly like REAL-MM-RAG data:
    #   pipeline.ingest("data/raw/CustomDocs/train/qas.jsonl")
    #   pipeline.run_inference("data/raw/CustomDocs/train/qas.jsonl", ...)

Requirements (in addition to existing requirements.txt):
    pip install pymupdf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Allow running from repo root or scripts/ ─────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from vlm import VLM  # project's existing VLM class


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert academic QA dataset curator. "
    "Generate knowledge-seeking question-answer pairs grounded strictly "
    "in the visual content of the document page image provided. "
    "Rules: "
    "(1) Questions must require understanding, not simple lookup. "
    "(2) Answers must be fully grounded in the page — no external knowledge. "
    "(3) Answers should be 2-4 sentences and reference visual elements "
    "(tables, figures, equations) when present. "
    "(4) Do NOT say 'this slide' or 'this image' — answer as factual knowledge. "
    "(5) Return ONLY a valid JSON array, no markdown fences, no explanation."
)

def make_user_prompt(num_questions: int, doc_type: str) -> str:
    return (
        f"Analyze this document page and generate {num_questions} diverse, "
        f"knowledge-seeking question-answer pairs.\n"
        f"Document type: {doc_type}\n\n"
        f"Return a JSON array only:\n"
        f'[\n  {{"question": "<question>", "answer": "<detailed answer>"}}\n]'
    )


# ── PDF rendering ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: Path, pages_dir: Path, dpi: int = 150) -> list[Path]:
    """Render each PDF page to a PNG. Returns list of saved image paths."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Run: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    saved = []
    doc_stem = pdf_path.stem.replace(" ", "_")

    for i in range(len(doc)):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        img_path = pages_dir / f"{doc_stem}_{i + 1}.png"
        pix.save(str(img_path))
        saved.append(img_path)

    doc.close()
    log.info(f"  Rendered {len(saved)} pages from {pdf_path.name}")
    return saved


# ── VLM inference ─────────────────────────────────────────────────────────────

def generate_qa_pairs(
    vlm: VLM,
    image_path: Path,
    doc_type: str,
    num_questions: int,
) -> list[dict]:
    """Call the local VLM on a page image and return parsed QA pairs."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": make_user_prompt(num_questions, doc_type)},
            ],
        },
    ]

    response = vlm.generate(messages, sampling_params={"max_new_tokens": 1024})
    raw = response.get("text", "").strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        pairs = json.loads(raw)
        assert isinstance(pairs, list)
        return pairs
    except Exception as e:
        log.warning(f"    JSON parse failed for {image_path.name}: {e}")
        log.debug(f"    Raw output was: {raw[:300]}")
        return []


# ── Quality filter ────────────────────────────────────────────────────────────

SEARCH_STYLE_KEYWORDS = [
    "what is written", "what does it say", "what is the title",
    "list the items on", "what text appears", "what is shown on",
]

def quality_filter(pair: dict) -> bool:
    q = pair.get("question", "").strip()
    a = pair.get("answer", "").strip()
    if len(q) < 25 or len(a) < 50:
        return False
    if any(kw in q.lower() for kw in SEARCH_STYLE_KEYWORDS):
        return False
    return True


def is_content_rich(img_path: Path, min_kb: float = 30.0) -> bool:
    """Skip near-blank pages — tiny PNG = mostly white."""
    return img_path.stat().st_size / 1024 >= min_kb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate custom fine-tuning QA data using local Qwen3-VL-8B"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing input PDF files")
    parser.add_argument("--out_dir", default="data/raw/CustomDocs/train",
                        help="Output dir — mirrors get_data.py structure (default: data/raw/CustomDocs/train)")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--checkpoint_path", default="./Qwen3-VL-8B-Instruct",
                        help="Local model checkpoint path (default: ./Qwen3-VL-8B-Instruct)")
    parser.add_argument("--num_questions", type=int, default=3,
                        help="QA pairs to generate per page (default: 3)")
    parser.add_argument("--doc_type", default="tech_slides",
                        choices=["tech_slides", "tech_report", "financial_report", "academic_paper"],
                        help="Document type label (default: tech_slides)")
    parser.add_argument("--skip_first_pages", type=int, default=1,
                        help="Skip first N pages per PDF, e.g. title/TOC (default: 1)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for PDF page rendering (default: 150)")
    args = parser.parse_args()

    from tqdm import tqdm

    # ── Setup ──
    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        log.error(f"No PDF files found in {input_dir}")
        return
    log.info(f"Found {len(pdf_files)} PDF(s) in {input_dir}")

    # ── Load the local VLM (same class used everywhere else in the project) ──
    log.info(f"Loading VLM: {args.model_id} from {args.checkpoint_path}")
    vlm = VLM(args.model_id, args.checkpoint_path)

    skip_pages = set(range(1, args.skip_first_pages + 1))

    # ── Process all PDFs ──
    all_qas:   list[dict] = []
    all_pages: list[dict] = []
    example_index = 0

    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        log.info(f"Processing: {pdf_path.name}")
        image_paths = pdf_to_images(pdf_path, pages_dir, dpi=args.dpi)

        for idx, img_path in enumerate(tqdm(image_paths, desc=f"  {pdf_path.stem}", leave=False)):
            page_num = idx + 1

            # Page metadata — matches get_data.py pages.jsonl format exactly
            all_pages.append({
                "dataset":       f"Custom_{args.doc_type}",
                "split":         "train",
                "example_index": example_index,
                "doc_id":        pdf_path.stem,
                "page_number":   page_num,
                "image_path":    str(img_path),
            })

            if page_num in skip_pages:
                log.info(f"    Skipping page {page_num} (title/TOC)")
                example_index += 1
                continue

            if not is_content_rich(img_path):
                log.info(f"    Skipping page {page_num} (likely blank)")
                example_index += 1
                continue

            pairs = generate_qa_pairs(vlm, img_path, args.doc_type, args.num_questions)
            kept  = [p for p in pairs if quality_filter(p)]
            log.info(f"    Page {page_num}: {len(kept)}/{len(pairs)} pairs kept")

            for pair in kept:
                # QA record — matches get_data.py qas.jsonl format exactly
                all_qas.append({
                    "dataset":       f"Custom_{args.doc_type}",
                    "split":         "train",
                    "example_index": example_index,
                    "question":      pair["question"],
                    "answer":        pair["answer"],
                    "image_path":    str(img_path),
                    "doc_id":        pdf_path.stem,
                    "page_number":   page_num,
                })
                example_index += 1

    # ── Write output ──
    qas_path   = out_dir / "qas.jsonl"
    pages_path = out_dir / "pages.jsonl"

    with open(qas_path, "w", encoding="utf-8") as f:
        for r in all_qas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(pages_path, "w", encoding="utf-8") as f:
        for r in all_pages:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== Done ===")
    print(f"QA pairs generated : {len(all_qas)}")
    print(f"Pages processed    : {len(all_pages)}")
    print(f"Output qas.jsonl   : {qas_path}")
    print(f"Output pages dir   : {pages_dir}")
    print(f"\nPlug into your pipeline:")
    print(f"  pipeline.ingest('{qas_path}')")
    print(f"  pipeline.run_inference('{qas_path}', output_path='results/custom_rag.json')")


if __name__ == "__main__":
    main()
