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

from qdrant_client import QdrantClient, models

# Class to retrive documents from info in data/ for RAG to pass as context to Qwen3VL
class RAGRetreiver():
    def __init__(
        self,
        model_name: str = "athrael-soju/colqwen3.5-4.5B-v3",
        persist_dir: str = "embeddings/",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
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

        # Initialize Qdrant (local disk persistence)
        self.client = QdrantClient(path=persist_dir)
        
        # CHANGE THIS LINE to force a new collection schema
        self.collection_name = "colqwen_multi_pages" 

        # Check if collection exists; if not, create it with MultiVectorConfig
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=128,  # ColQwen late-interaction vectors are 128-dimensional
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            )
    
    # Function to embed all documents for later retrieval using Qdrant
    def ingest_dataset(self, qas_jsonl_path: str, pages_root: str | None = None):
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

            # embd is already a numpy array, just convert it to a list
            multi_vector = embd.tolist()
            
            # Generate a unique integer ID (Qdrant requires int or UUID)
            doc_id_int = abs(hash(img_path)) % (10 ** 8)
            doc_id_string = metadata["doc_id"]

            # Upsert as a SINGLE point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id_int,
                        vector=multi_vector,
                        payload={"path": img_path, "doc_id": doc_id_string}
                    )
                ]
            )

            
    def search(self, query: str, top_k: int = 3):
        # Embed the query
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            query_embd = self.model(**batch)[0].cpu().float().numpy()

        # query_embd is already a numpy array, just convert the first element to a list
        query_multi_vector = query_embd.tolist()

        # Let Qdrant do the heavy lifting
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_multi_vector,
            limit=top_k
        )

        # Format the output to match what pipeline.py expects
        ranked_docs = []
        for point in search_results.points:
            ranked_docs.append((
                point.payload["doc_id"], 
                point.score, 
                point.payload["path"]
            ))

        return ranked_docs



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
