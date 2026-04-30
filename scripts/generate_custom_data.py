"""
Custom Fine-tuning Data Generation Pipeline (Image/PNG Edition)
===============================================================
Generates new (question, answer, image_path) triplets from a folder of PNGs
using a larger teacher model.

The model is downloaded to Colab's ephemeral storage (/content/) to 
avoid hitting Google Drive storage quotas.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath('/content/drive/MyDrive/multimodal-rag'))

try:
    from scripts.vlm import VLM
except ImportError:
    logger.error("Could not import VLM from scripts.vlm. Ensure you are running this from the repository root.")
    sys.exit(1)

def extract_json_array(text: str) -> list:
    """Attempts to cleanly extract a JSON array from the model's text output."""
    match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from output:\n{text}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from directory of images")

    parser.add_argument("--input_dir", type=Path, default=Path("./accepted"), help="Directory containing source PNGs")
    parser.add_argument("--out_dir", type=Path, default=Path("./accepted"), help="Where to save qas.jsonl")
    parser.add_argument("--doc_type", type=str, default="tech_slides", choices=["financial_report", "tech_report", "tech_slides", "lecture_slides"])
    parser.add_argument("--num_questions", type=int, default=3, help="Number of QAs per image")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="HuggingFace model ID for teacher")
    parser.add_argument("--local_dir", type=str, default="/content/tmp_teacher_model", help="Colab local path to avoid GDrive limits")
    parser.add_argument("--resume", action="store_true", help="Skip images already present in qas.jsonl")

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qas_path = args.out_dir / "qas.jsonl"

    # Define the instruction block
    prompt_instruction = ""
    if args.doc_type == "financial_report":
        prompt_instruction = "Focus on extracting specific numerical figures, year-over-year changes, table data, and financial metrics."
    elif args.doc_type == "tech_report":
        prompt_instruction = "Focus on technical architecture, system specs, methodology, diagrams, and specific technical claims."
    elif args.doc_type == "tech_slides":
        prompt_instruction = "Focus on the high-level bullet points, slide titles, key takeaways, and flowcharts presented."
    elif args.doc_type == "lecture_slides":
        prompt_instruction = "Focus on key concepts, definitions, equations, diagrams, and the main pedagogical points of the slide."

    # ── Resume support: collect image paths and example_index already on disk ──
    done_images = set()
    next_example_index = 0
    if args.resume and qas_path.exists():
        with open(qas_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_images.add(rec.get("image_path"))
                    next_example_index = max(next_example_index, rec.get("example_index", -1) + 1)
                except json.JSONDecodeError:
                    continue
        logger.info(f"Resume: {len(done_images)} images already processed; resuming from example_index {next_example_index}.")

    # ── Process Input Images ──
    image_files = list(args.input_dir.glob("*.png")) + list(args.input_dir.glob("*.jpg"))
    if not image_files:
        logger.error(f"No PNG or JPG files found in {args.input_dir}")
        return

    # ── Initialize Model ──
    logger.info(f"Loading teacher model: {args.model_id}")
    logger.info(f"Downloading to ephemeral storage: {args.local_dir} (Protects GDrive Space)")

    vlm = VLM(
        model_id=args.model_id,
        checkpoint_path=args.local_dir,
        force_transformers=False
    )

    example_index = next_example_index
    total_kept = 0

    # Open the output file in append mode and flush after each image so a
    # disconnect doesn't lose progress. fsync is overkill here; flush is enough
    # to push the line out of Python's buffer to the OS.
    mode = "a" if args.resume else "w"
    with open(qas_path, mode, encoding="utf-8") as out_f:
        for img_path in tqdm(image_files, desc="Processing Images"):
            img_path_str = str(img_path)
            if img_path_str in done_images:
                continue

            doc_id = img_path.stem

            prompt = prompt = (
              f"You are given a single page (slide) from a lecture deck. The slide may contain text, "
              f"figures, equations, tables, and diagrams.\n\n"
              f"Your task is to produce {args.num_questions} question-answer pairs that an interested "
              f"learner might ask to test their understanding of the topic shown on this slide. The "
              f"questions must be self-contained and answerable by someone who can see the slide but "
              f"who is not aware that any 'slide', 'page', 'document', 'figure', 'dialogue', 'example', "
              f"or 'lecture' is being referenced. Write questions about the SUBJECT MATTER itself, not about the slide as an artifact.  Under absoultely no cicumstances use references such as 'in the dialouge' 'on the slide' 'in this slide' etc, or question containing an explicit reference to a slide. All of the questions should be questions about the topic the slide is on, but should in no way reference the slide. For example, a question containing 'on the slide discussing ...' is absolutely forbidden under all circumstances.\n\n"
              f"Substituting words for slide is absolutely forbidden. An example of such a workaround which is not allowed is the following: 'What is the primary focus of the content presented in the document?'. The preceding question is an example of a question which is not about the subject matter on a document but rather about the document itself. Only ever write question about the subject matter and concepts without reference to the document in the question, this is absolutely crucial.\n"
              f"If you cannot make a relevant question without explicitly referencing the slide, write your question as 'NO QUESTION'\n"
              f"Each question must:\n"
              f"- Refer to the actual concepts, techniques, theorems, names, or quantities shown\n"
              f"- Be answerable in 1-3 sentences from information uniquely supported by this slide\n"
              f"- Use specific terms (e.g. 'Euclid's proof', 'softmax', 'transformer', 'a CNN with 3 layers') "
              f"rather than generic ones ('this proof', 'the model', 'the architecture')\n"
              f"- Where useful, combine information from multiple parts of the slide\n\n"
              f"{prompt_instruction}\n\n"
              f"Examples of GOOD questions for a slide explaining Euclid's proof of infinite primes:\n"
              f"- How does Euclid's proof construct a new number to derive a contradiction?\n"
              f"- Why must the constructed number Q have a prime factor larger than P?\n"
              f"- What assumption does Euclid's proof of the infinitude of primes start from?\n\n"
              f"Examples of BAD questions (DO NOT WRITE QUESTIONS LIKE THESE):\n"
              f"- What is the main topic being introduced on this slide? [refers to the slide as an artifact]\n"
              f"- What proof is being discussed in the dialogue? [refers to slide framing]\n"
              f"- What can you learn from this figure? [refers to a visual element]\n"
              f"- What is the title? [trivial / about the slide, not the content]\n"
              f"- How does the model work? [too generic, no specific name]\n\n"
              f"Return the output EXCLUSIVELY as a valid JSON array of objects. Each object must have "
              f"exactly two string keys: \"question\" and \"answer\". Do not include markdown code "
              f"fences, do not include any text before or after the JSON array."
          )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path_str},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            logger.info(f"Generating QAs for {img_path.name}...")
            try:
                response = vlm.generate(messages)
                qa_pairs = extract_json_array(response["text"])
            except Exception as e:
                logger.error(f"Inference failed for {img_path.name}: {e}")
                qa_pairs = []

            kept = [p for p in qa_pairs if isinstance(p, dict) and "question" in p and "answer" in p]
            logger.info(f"  -> Kept {len(kept)}/{len(qa_pairs)} pairs")

            for pair in kept:
                record = {
                    "dataset":       f"Custom_{args.doc_type}",
                    "split":         "test",
                    "example_index": example_index,
                    "question":      pair["question"],
                    "answer":        pair["answer"],
                    "image_path":    img_path_str,
                    "doc_id":        doc_id,
                    "page_number":   None,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                example_index += 1
                total_kept += 1

            out_f.flush() 

    print(f"\n=== Done ===")
    print(f"QA pairs generated this run : {total_kept}")
    print(f"Output saved to             : {qas_path}")

if __name__ == "__main__":
    main()