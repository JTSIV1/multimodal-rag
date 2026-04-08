("""Download and export REAL-MM-RAG datasets from Hugging Face.

Saves page images to `data/raw/<dataset_short>/<split>/pages/` and
creates `pages.jsonl` and `qas.jsonl` metadata files for each dataset.

Usage:
	python scripts/get_data.py --out_dir data/raw --datasets ibm-research/REAL-MM-RAG_FinReport

The script tries to be robust to a few common dataset schemas used on HF.
It can optionally run Tesseract OCR (if `pytesseract` and `tesseract` are available)
to produce `ocr_text` for each saved page image.
""")

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional
from tqdm import tqdm

try:
	from datasets import load_dataset
except Exception as e:
	print("ERROR: the `datasets` library is required. Install with `pip install datasets`.")
	raise

try:
	from PIL import Image
except Exception:
	print("ERROR: the `Pillow` library is required. Install with `pip install pillow`.")
	raise

try:
	import pytesseract
	TESSERACT_AVAILABLE = True
except Exception:
	pytesseract = None  # type: ignore
	TESSERACT_AVAILABLE = False


COMMON_DOC_ID_FIELDS = ["doc_id", "document_id", "docname", "document", "id"]
COMMON_PAGE_FIELDS = ["page", "page_index", "page_number", "page_id", "pageno"]
COMMON_QUESTION_FIELDS = ["question", "query", "prompt", "question_text"]
COMMON_ANSWER_FIELDS = ["answer", "answers", "label", "gt", "ground_truth"]


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def find_first_field(example: Dict[str, Any], candidates: Iterable[str]) -> Optional[Any]:
	for c in candidates:
		if c in example and example[c] is not None:
			return example[c]
	return None


def pil_from_value(v: Any) -> Optional[Image.Image]:
	if v is None:
		return None
	# HuggingFace ImageFeature typically stores a dict with 'path'
	if isinstance(v, dict) and "path" in v:
		try:
			return Image.open(v["path"]).convert("RGB")
		except Exception:
			pass
	# If it's already a PIL image
	try:
		if isinstance(v, Image.Image):
			return v.convert("RGB")
	except Exception:
		pass
	# If it's raw bytes
	if isinstance(v, (bytes, bytearray)):
		try:
			return Image.open(io.BytesIO(v)).convert("RGB")
		except Exception:
			pass
	# If it's a string path or URL
	if isinstance(v, str):
		if os.path.exists(v):
			try:
				return Image.open(v).convert("RGB")
			except Exception:
				pass
		# try HTTP fetch for URLs
		if v.startswith("http://") or v.startswith("https://"):
			try:
				import requests

				resp = requests.get(v, timeout=20)
				resp.raise_for_status()
				return Image.open(io.BytesIO(resp.content)).convert("RGB")
			except Exception:
				pass
	# Arrow ImageFeature may expose bytes under 'bytes' key
	if isinstance(v, dict) and "bytes" in v:
		try:
			return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
		except Exception:
			pass
	return None


def save_image(img: Image.Image, out_path: str) -> None:
	ensure_dir(os.path.dirname(out_path))
	img.save(out_path, format="PNG")


def export_dataset(
	hf_id: str,
	out_dir: str,
	max_examples: Optional[int] = None,
	run_ocr: bool = False,
) -> None:
	ds = load_dataset(hf_id)
	short = hf_id.split("/")[-1]
	base = os.path.join(out_dir, short)
	ensure_dir(base)

	for split_name, subset in ds.items():
		print(f"Processing {hf_id} split={split_name} (size={len(subset)})")
		pages_meta: List[Dict[str, Any]] = []
		qas_meta: List[Dict[str, Any]] = []
		split_dir = os.path.join(base, split_name)
		pages_dir = os.path.join(split_dir, "pages")
		ensure_dir(pages_dir)

		# detect image columns
		img_cols = [k for k, v in subset.features.items() if "image" in k.lower() or v.__class__.__name__.lower() == "image"]
		if not img_cols:
			# fallback: any column named 'image' or containing 'page'
			img_cols = [k for k in subset.features.keys() if "image" in k.lower() or "page" in k.lower()]
		if not img_cols:
			print(f"  WARNING: could not find image column for {hf_id} {split_name}; skipping images.")

		count = 0
		for i, ex in tqdm(enumerate(subset), total=len(subset)):
			if max_examples is not None and count >= max_examples:
				break

			# get document/page identifiers heuristically
			doc_id = find_first_field(ex, COMMON_DOC_ID_FIELDS)
			page_no = find_first_field(ex, COMMON_PAGE_FIELDS)

			# try to get image
			img = None
			if img_cols:
				for col in img_cols:
					if col in ex:
						img = pil_from_value(ex[col])
						if img is not None:
							img_col = col
							break
			else:
				img_col = None

			image_path = None
			if img is not None:
				fname = f"{doc_id or 'doc'}_{page_no if page_no is not None else i}.png"
				# sanitize fname
				fname = fname.replace(os.sep, "_")
				dest = os.path.join(pages_dir, fname)
				try:
					save_image(img, dest)
					image_path = os.path.relpath(dest)
				except Exception as e:
					print(f"  ERROR saving image for example {i}: {e}")

			# optional OCR
			ocr_text = None
			if run_ocr and image_path is not None:
				if not TESSERACT_AVAILABLE:
					print("  WARNING: pytesseract/tesseract not available; skipping OCR.")
				else:
					try:
						ocr_text = pytesseract.image_to_string(Image.open(os.path.join(os.getcwd(), image_path)))
						# save OCR text file
						if ocr_text:
							txt_path = os.path.join(pages_dir, fname + ".txt")
							with open(txt_path, "w", encoding="utf-8") as f:
								f.write(ocr_text)
					except Exception as e:
						print(f"  WARNING: OCR failed for {image_path}: {e}")

			page_meta = {
				"dataset": hf_id,
				"split": split_name,
				"example_index": i,
				"doc_id": doc_id,
				"page_number": page_no,
				"image_path": image_path,
			}
			if ocr_text:
				page_meta["ocr_text"] = ocr_text

			pages_meta.append(page_meta)

			# QA fields
			question = find_first_field(ex, COMMON_QUESTION_FIELDS)
			answer = find_first_field(ex, COMMON_ANSWER_FIELDS)
			if question is not None and answer is not None:
				qas_meta.append(
					{
						"dataset": hf_id,
						"split": split_name,
						"example_index": i,
						"question": question,
						"answer": answer,
						"image_path": image_path,
						"doc_id": doc_id,
						"page_number": page_no,
					}
				)

			count += 1

		# write metadata
		pages_path = os.path.join(split_dir, "pages.jsonl")
		qas_path = os.path.join(split_dir, "qas.jsonl")
		with open(pages_path, "w", encoding="utf-8") as f:
			for item in pages_meta:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")

		with open(qas_path, "w", encoding="utf-8") as f:
			for item in qas_meta:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")

		print(f"  Wrote {len(pages_meta)} pages and {len(qas_meta)} QA entries to {split_dir}")


def parse_args():
	p = argparse.ArgumentParser(description="Download REAL-MM-RAG datasets and export pages + metadata")
	p.add_argument(
		"--datasets",
		type=str,
		default=",".join([
			"ibm-research/REAL-MM-RAG_FinReport",
			"ibm-research/REAL-MM-RAG_TechReport",
			"ibm-research/REAL-MM-RAG_TechSlides",
		]),
		help="Comma-separated dataset ids on Hugging Face",
	)
	p.add_argument("--out_dir", type=str, default="data/raw", help="Output base directory")
	p.add_argument("--max_examples", type=int, default=None, help="Max examples per split (for quick testing)")
	p.add_argument("--run_ocr", action="store_true", help="Run Tesseract OCR on saved pages (optional)")
	return p.parse_args()


def main():
	args = parse_args()
	dataset_ids = [s.strip() for s in args.datasets.split(",") if s.strip()]
	for did in dataset_ids:
		try:
			export_dataset(did, args.out_dir, max_examples=args.max_examples, run_ocr=args.run_ocr)
		except Exception as e:
			print(f"ERROR processing {did}: {e}")


if __name__ == "__main__":
	main()

