from typing import List
import json
import os
import glob
import torch
from PIL import Image
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
import chromadb
from tqdm import tqdm
from pathlib import Path

# Class to retrive documents from info in data/ for RAG to pass as context to Qwen3VL
class RAGRetreiver():
    def __init__(
        self,
        model_name: str = "athrael-soju/colqwen3.5-4.5B-v3",
        persist_dir: str = "embeddings/",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        per_page: bool = True,
    ):
        
        self.model = ColQwen3_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa"
        )

        self.processor = ColQwen3_5Processor.from_pretrained(model_name)

        # record device string for tensor/device operations
        self.device = device
        self.persist_dir = persist_dir
        self.per_page = per_page

        # init chromadb with persistent storage
        # embeddings are stored on disk so you can embed once and reload later.
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # create/get collection
        self.collection = self.client.get_or_create_collection(
            name="doc_patches",
            metadata={"hnsw:space": "cosine"},
        )
    
    # Function to embed all documents for later retrieval
    def ingest_dataset(self, qas_jsonl_path: str, pages_root: str | None = None, batch_size: int = 256):
        """Ingest a QA JSONL file produced by `scripts/get_data.py`.

        qas_jsonl_path: path to qas.jsonl
        pages_root: optional root folder to resolve image paths (defaults to parent 'pages' folder)
        """
        qas_path = Path(qas_jsonl_path)
        if not qas_path.exists():
            raise FileNotFoundError(f"qas.jsonl not found: {qas_jsonl_path}")

        # default pages folder (sibling 'pages' folder next to qas.jsonl)
        default_pages_root = qas_path.parent / "pages"
        if pages_root is None:
            pages_root = str(default_pages_root)

        # load QA metadata
        with open(qas_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # Buffers for batch adding
        buf_ids: List[str] = []
        buf_embs: List[List[float]] = []
        buf_metas: List[dict] = []

        for entry in tqdm(data, desc="Indexing Pages"):
            img_rel = entry.get("image_path")

            img_path = None
            # Try several strategies to resolve the image path
            if img_rel:
                # absolute path
                if os.path.isabs(img_rel) and os.path.exists(img_rel):
                    img_path = img_rel
                # relative to current cwd
                elif os.path.exists(os.path.join(os.getcwd(), img_rel)):
                    img_path = os.path.join(os.getcwd(), img_rel)
                # relative to the pages folder next to qas.jsonl
                elif os.path.exists(os.path.join(pages_root, os.path.basename(img_rel))):
                    img_path = os.path.join(pages_root, os.path.basename(img_rel))
                # maybe the image_path already contains the full data/raw path
                elif os.path.exists(img_rel):
                    img_path = img_rel
            # fallback: try to locate by doc_id and page_number
            if img_path is None:
                doc_id = entry.get("doc_id")
                page_no = entry.get("page_number")
                if doc_id is not None and page_no is not None:
                    pattern = os.path.join(pages_root, f"*{doc_id}*_{page_no}.*")
                    matches = glob.glob(pattern)
                    if matches:
                        img_path = matches[0]

            if img_path is None:
                # As a last resort, try to find by basename across pages_root
                if img_rel:
                    basename = os.path.basename(img_rel)
                    matches = list(Path(pages_root).rglob(basename)) if os.path.exists(pages_root) else []
                    if matches:
                        img_path = str(matches[0])

            if img_path is None or not os.path.exists(img_path):
                print(f"  WARNING: Could not resolve image for entry {entry.get('example_index')} (doc {entry.get('doc_id')})")
                continue

            # Embed the image
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  ERROR opening image {img_path}: {e}")
                continue

            with torch.no_grad():
                batch = self.processor.process_images([image]).to(self.device)
                embd = self.model(**batch)[0].cpu().float().numpy()
                if embd.ndim == 3:
                    embd = embd[0]  # [num_patches, dim]

            metadata = {
                "doc_id": str(entry.get("doc_id", "")),
                "image_path": img_path,
                "question": entry.get("question", ""),
                "answer": entry.get("answer", "")
            }

            # Two ingestion modes:
            # - per_page: aggregate patch embeddings to a single vector per page (compact, recommended)
            # - per_patch: store each patch as a separate vector (larger, more fine-grained retrieval)
            # Prefix IDs with the dataset slug so entries from different datasets
            # never collide in the shared ChromaDB collection.
            dataset_slug = str(entry.get("dataset", "unk")).split("/")[-1]
            ex_idx = entry.get('example_index', '0')

            if self.per_page:
                # mean-pool across patches to get a single page embedding
                page_emb = embd.mean(axis=0)
                vid = f"{dataset_slug}_{ex_idx}"
                buf_ids.append(vid)
                buf_embs.append(page_emb.tolist())
                buf_metas.append(metadata)
            else:
                # store every patch separately
                num_patches = embd.shape[0]
                ids = [f"{dataset_slug}_{ex_idx}_p{i}" for i in range(num_patches)]
                for i in range(num_patches):
                    buf_ids.append(ids[i])
                    buf_embs.append(embd[i].tolist())
                    buf_metas.append(metadata)

            # Flush batch to ChromaDB periodically to avoid building huge in-memory lists
            if len(buf_ids) >= batch_size:
                try:
                    self.collection.add(ids=buf_ids, embeddings=buf_embs, metadatas=buf_metas)
                except Exception as e:
                    print(f"  ERROR adding batch to DB: {e}")
                buf_ids.clear()
                buf_embs.clear()
                buf_metas.clear()

        # flush remaining
        if buf_ids:
            try:
                self.collection.add(ids=buf_ids, embeddings=buf_embs, metadatas=buf_metas)
            except Exception as e:
                print(f"  ERROR adding final batch to DB: {e}")

    def search(self, query, num_patches=1000, top_k=10):
        # Embed the query
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            query_embd = self.model(**batch)[0].cpu().float().numpy() # [num_query_tokens, 128]

        # Retrivew top patches for each query token
        token_results = self.collection.query(
            query_embeddings=query_embd.tolist(),
            n_results=num_patches # get many patches to filter down with MaxSim
        )

        # MaxSim Scoring for patches to top-k
        doc_scores = {}
        for token_idx, distances in enumerate(token_results['distances']):
            for i, dist in enumerate(distances):
                meta = token_results['metadatas'][token_idx][i]
                doc_id = meta['doc_id']
                similarity = 1.0 - dist # Convert distance to similarity
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'score_vec': [0.0] * len(query_embd), 'path': meta['image_path']}
                
                # Keep the best similarity for this token
                doc_scores[doc_id]['score_vec'][token_idx] = max(doc_scores[doc_id]['score_vec'][token_idx], similarity)

        # Sum the max similarities for final ranking
        ranked_docs = sorted(
            [(doc_id, sum(info['score_vec']), info['path']) for doc_id, info in doc_scores.items()],
            key=lambda x: x[1],
        )

        return ranked_docs[:top_k]



if __name__ == "__main__":
    rag = RAGRetreiver()

    # Try to discover datasets under the repository's data/raw folder
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "raw"
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
                        rag.ingest_dataset(str(qas_path))
                    except Exception as e:
                        print(f"Failed to ingest {qas_path}: {e}")

    # example search
    results = rag.search("What was IBM's revenue growth in Q4 2006?")
    for doc_id, score, path in results:
        print(f"Found Doc {doc_id} (Score: {score:.4f}) at {path}")
