from typing import List, Optional
import json
import os
import glob
from pathlib import Path
import torch
from tqdm import tqdm
import chromadb

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SSBERT = True
except Exception:
    _HAS_SSBERT = False

from transformers import AutoTokenizer, AutoModel
import numpy as np


class OCRRetriever:
    """Retriever that indexes OCR-extracted text files and performs vector search.

    By default this tries to use `sentence-transformers` for embeddings and
    falls back to `transformers` + mean pooling if not available.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: Optional[str] = None,
        device: Optional[str] = None,
        per_page: bool = True,
    ):
        self.model_name = model_name
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else
                ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = device

        # Default to saving embeddings inside the `scripts/` folder as requested
        if persist_dir is None:
            scripts_dir = Path(__file__).parent
            default_dir = scripts_dir / "embeddings_ocr"
            self.persist_dir = str(default_dir)
        else:
            self.persist_dir = persist_dir

        os.makedirs(self.persist_dir, exist_ok=True)

        # initialize embedding backend
        if _HAS_SSBERT:
            try:
                self.embedder = SentenceTransformer(self.model_name)
                self.embed_type = "sbert"
            except Exception:
                self.embed_type = "transformers"
        else:
            self.embed_type = "transformers"

        if self.embed_type == "transformers":
            # AutoModel fallback with mean pooling
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

        # setup chromadb
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="ocr_doc_patches",
            metadata={"hnsw:space": "cosine"},
        )

        self.per_page = per_page

    def _embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)

        if self.embed_type == "sbert":
            embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs

        # transformers fallback
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs, return_dict=True)
                last_hidden = out.last_hidden_state
                mask = inputs.get("attention_mask")
                if mask is None:
                    emb = last_hidden.mean(dim=1)
                else:
                    mask = mask.unsqueeze(-1)
                    summed = (last_hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1e-9)
                    emb = summed / denom
                all_embs.append(emb.cpu().numpy())

        return np.vstack(all_embs)

    def ingest_dataset(self, qas_jsonl_path: str, pages_root: Optional[str] = None, batch_size: int = 256, chunk_size_chars: Optional[int] = None):
        qas_path = Path(qas_jsonl_path)
        if not qas_path.exists():
            raise FileNotFoundError(f"qas.jsonl not found: {qas_jsonl_path}")

        default_pages_root = qas_path.parent / "pages"
        if pages_root is None:
            pages_root = str(default_pages_root)

        with open(qas_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        buf_ids: List[str] = []
        buf_texts: List[str] = []
        buf_metas: List[dict] = []

        for entry in tqdm(data, desc="Indexing OCR pages"):
            img_rel = entry.get("image_path")

            text_path = None
            # Try several strategies to resolve text file path
            if img_rel:
                # absolute path candidate
                if os.path.isabs(img_rel):
                    cand = img_rel + ".txt"
                    if os.path.exists(cand):
                        text_path = cand
                    else:
                        # maybe image path already has extension, keep that and add .txt
                        cand2 = os.path.splitext(img_rel)[0] + os.path.splitext(img_rel)[1] + ".txt"
                        if os.path.exists(cand2):
                            text_path = cand2

                # relative to cwd
                if text_path is None and os.path.exists(os.path.join(os.getcwd(), img_rel + ".txt")):
                    text_path = os.path.join(os.getcwd(), img_rel + ".txt")

                # relative to pages folder
                if text_path is None and os.path.exists(os.path.join(pages_root, os.path.basename(img_rel) + ".txt")):
                    text_path = os.path.join(pages_root, os.path.basename(img_rel) + ".txt")

                # sometimes text file is name.png.txt
                if text_path is None:
                    cand = os.path.join(pages_root, os.path.basename(img_rel)) + ".txt"
                    if os.path.exists(cand):
                        text_path = cand

                # try replacing extension with .txt
                if text_path is None:
                    cand = os.path.join(pages_root, os.path.splitext(os.path.basename(img_rel))[0] + ".txt")
                    if os.path.exists(cand):
                        text_path = cand

            # fallback: try to locate by doc_id and page_number
            if text_path is None:
                doc_id = entry.get("doc_id")
                page_no = entry.get("page_number")
                if doc_id is not None and page_no is not None:
                    pattern = os.path.join(pages_root, f"*{doc_id}*_{page_no}*.txt")
                    matches = glob.glob(pattern)
                    if matches:
                        text_path = matches[0]

            # as last resort, try basename search across pages_root
            if text_path is None and img_rel:
                basename = os.path.basename(img_rel)
                matches = list(Path(pages_root).rglob(basename + "*.txt")) if os.path.exists(pages_root) else []
                if matches:
                    text_path = str(matches[0])

            if text_path is None or not os.path.exists(text_path):
                print(f"  WARNING: Could not resolve text for entry {entry.get('example_index')} (doc {entry.get('doc_id')})")
                continue

            try:
                with open(text_path, "r", encoding="utf-8", errors="ignore") as tf:
                    page_text = tf.read().strip()
            except Exception as e:
                print(f"  ERROR opening text {text_path}: {e}")
                continue

            if not page_text:
                continue

            texts_to_index = []
            if chunk_size_chars and chunk_size_chars > 0:
                start = 0
                chunk_id = 0
                while start < len(page_text):
                    chunk = page_text[start:start + chunk_size_chars]
                    texts_to_index.append((chunk_id, chunk))
                    chunk_id += 1
                    start += chunk_size_chars
            else:
                texts_to_index = [(0, page_text)]

            for chunk_idx, txt in texts_to_index:
                vid = f"page_{entry.get('doc_id','unk')}_{entry.get('example_index','0')}_c{chunk_idx}"
                metadata = {
                    "doc_id": str(entry.get("doc_id", "")),
                    "text_path": text_path,
                    "question": entry.get("question", ""),
                    "answer": entry.get("answer", ""),
                }
                buf_ids.append(vid)
                buf_texts.append(txt)
                buf_metas.append(metadata)

                if len(buf_ids) >= batch_size:
                    try:
                        embs = self._embed_texts(buf_texts)
                        self.collection.add(ids=buf_ids, embeddings=embs.tolist(), metadatas=buf_metas)
                    except Exception as e:
                        print(f"  ERROR adding batch to DB: {e}")
                    buf_ids.clear()
                    buf_texts.clear()
                    buf_metas.clear()

        # flush remaining
        if buf_ids:
            try:
                embs = self._embed_texts(buf_texts)
                self.collection.add(ids=buf_ids, embeddings=embs.tolist(), metadatas=buf_metas)
            except Exception as e:
                print(f"  ERROR adding final batch to DB: {e}")

    def search(self, query: str, top_k: int = 10):
        with torch.no_grad():
            q_emb = self._embed_texts([query])[0]

        results = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)

        out = []
        # results may return lists nested per-query
        ids_list = results.get("ids", [])
        metadatas_list = results.get("metadatas", [])
        distances_list = results.get("distances", [])

        if ids_list:
            for i, _ in enumerate(ids_list[0] if isinstance(ids_list[0], list) else ids_list):
                rid = ids_list[0][i] if isinstance(ids_list[0], list) else ids_list[i]
                meta = metadatas_list[0][i] if isinstance(metadatas_list[0], list) else metadatas_list[i]
                dist = distances_list[0][i] if isinstance(distances_list[0], list) else distances_list[i]
                similarity = 1.0 - dist if isinstance(dist, (int, float)) else None
                out.append((rid, similarity, meta))

        return out


if __name__ == "__main__":
    # Discover datasets under repository's data/raw folder and ingest
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "raw" / "scripts"
    ocr = OCRRetriever()

    if not data_root.exists():
        print(f"data/raw not found at expected location: {data_root}")
    else:
        print(f"Scanning {data_root} for datasets...")
        for dataset_dir in sorted(data_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for split_dir in sorted(dataset_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                qas_path = split_dir / "qas.jsonl"
                if qas_path.exists():
                    print(f"Ingesting {qas_path}")
                    try:
                        ocr.ingest_dataset(str(qas_path))
                    except Exception as e:
                        print(f"Failed to ingest {qas_path}: {e}")
