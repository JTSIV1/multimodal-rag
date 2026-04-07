# multimodal-rag

## Environment Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pull REAL-MM-RAG data

See scripts/get_data.py for the helper to pull the REAL-MM-RAG dataset.

- Default datasets from hf: `ibm-research/REAL-MM-RAG_FinReport`, `ibm-research/REAL-MM-RAG_TechReport`, `ibm-research/REAL-MM-RAG_TechSlides`

Test run download (only 10 examples per split):

```bash
python3 scripts/get_data.py --out_dir data/raw --max_examples 10
```

Full download (all default datasets):

```bash
python3 scripts/get_data.py --out_dir data/raw
```

By default the script saves page images and writes two metadata files per split:

- `data/raw/<dataset_short>/<split>/pages/` — PNG page images (one file per page)
- `data/raw/<dataset_short>/<split>/pages.jsonl` — page-level metadata (image path, doc id, page number, optional OCR)
- `data/raw/<dataset_short>/<split>/qas.jsonl` — QA triples linking questions/answers to page images

**Optional OCR**:
To download OCR text saved alongside images (for baselines), install `tesseract` binary from homebrew.

Homebrew (macOS):

```bash
brew install tesseract
```

Pull and extract metadata with OCR:

```bash
python3 scripts/get_data.py --out_dir data/raw --run_ocr
```
