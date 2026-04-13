"""MMRAGPipeline: assembles a retriever + VLM for REAL-MM-RAG QA inference.

Three retrieval modes:
  - "image_rag" : visual retrieval (ColQwen3.5) → top-k image pages → VLM
  - "ocr_rag"   : text retrieval (sentence-transformers on OCR .txt) → context text → VLM
  - "no_rag"    : no retrieval; VLM receives only the question
                  (during evaluate(), the oracle image from the QA entry is passed
                   so this mode acts as an upper-bound / oracle baseline)

Inference results are saved as JSON so that evaluate.py can load them and compute
BLEU, ROUGE, BERTScore, and LLM-as-judge scores independently.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from vlm import VLM
from rag import RAGRetreiver
from ocr import OCRRetriever

VALID_MODES = ("image_rag", "ocr_rag", "no_rag")


class MMRAGPipeline:
    def __init__(
        self,
        mode: str,
        vlm_model_id: str,
        vlm_checkpoint_path: str,
        image_retriever_model_id: str = "athrael-soju/colqwen3.5-v1",
        image_retriever_persist_dir: str = "embeddings/",
        ocr_retriever_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        ocr_retriever_persist_dir: Optional[str] = None,
        top_k: int = 3,
        device: Optional[str] = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.top_k = top_k

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available() else
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
                "cpu"
            )
        self.device = device

        print(f"Initializing MMRAGPipeline | mode={mode} | device={device}")

        self.vlm = VLM(vlm_model_id, vlm_checkpoint_path)

        self.retriever: Optional[Any] = None
        if mode == "image_rag":
            self.retriever = RAGRetreiver(
                model_name=image_retriever_model_id,
                persist_dir=image_retriever_persist_dir,
                device=device,
            )
        elif mode == "ocr_rag":
            self.retriever = OCRRetriever(
                model_name=ocr_retriever_model_name,
                persist_dir=ocr_retriever_persist_dir,
                device=device,
            )

   
    # Ingestion
    def ingest(self, qas_jsonl_path: str, pages_root: Optional[str] = None) -> None:
        if self.retriever is None:
            print("Skipping ingest: no retriever in 'no_rag' mode.")
            return
        self.retriever.ingest_dataset(qas_jsonl_path, pages_root=pages_root)

    # Message builders
    def _build_image_rag_message(self, query: str) -> dict:
        """Retrieve top-k page images via visual RAG and build a VLM message."""
        raw = self.retriever.search(query, num_patches=1000, top_k=self.top_k * 5)
        results = sorted(raw, key=lambda x: x[1], reverse=True)[: self.top_k]

        content: List[dict] = []
        for _, _score, img_path in results:
            if img_path and os.path.exists(img_path):
                content.append({
                    "type": "image", 
                    "image": img_path
                })

        if not content:
            print("  WARNING: no valid images found in image RAG results.")

        content.append({"type": "text", "text": query})
        return {"role": "user", "content": content}

    def _build_ocr_rag_message(self, query: str) -> dict:
        """Retrieve top-k OCR text chunks and inject them as context into the prompt."""
        results = self.retriever.search(query, top_k=self.top_k)

        context_parts: List[str] = []
        for i, (_rid, _sim, meta) in enumerate(results):
            text_path = meta.get("text_path", "")
            if text_path and os.path.exists(text_path):
                try:
                    with open(text_path, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read().strip()
                    if text:
                        context_parts.append(f"[Document {i + 1}]\n{text}")
                except Exception as exc:
                    print(f"  WARNING: could not read {text_path}: {exc}")

        if context_parts:
            context = "\n\n".join(context_parts)
            prompt = (
                "Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        else:
            print("  WARNING: no OCR text retrieved; falling back to question only.")
            prompt = query

        return {"role": "user", "content": [{"type": "text", "text": prompt}]}

    def _build_no_rag_message(self, query: str, image_path: Optional[str] = None) -> dict:
        """Build a direct VLM message with an optional oracle image."""
        content: List[dict] = []
        if image_path and os.path.exists(image_path):
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": query})
        return {"role": "user", "content": content}

    
    # Inference
    def generate(
        self,
        query: str,
        image_path: Optional[str] = None,
        sampling_params: Optional[dict] = None,
    ) -> str:
        """Run a single query through the pipeline and return the generated text.

        Args:
            query:           The question / prompt.
            image_path:      Oracle image path — only used in 'no_rag' mode.
            sampling_params: Forwarded to VLM.generate().
        """
        if sampling_params is None:
            sampling_params = {"max_new_tokens": 512}

        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a question answering assistant for corporate applications. Respond in one sentence using all available information"}]}]

        if self.mode == "image_rag":
            # add system prompt
            messages.append(self._build_image_rag_message(query))
        elif self.mode == "ocr_rag":
            messages.append(self._build_ocr_rag_message(query))
        else:
            messages.append(self._build_no_rag_message(query, image_path))

        response = self.vlm.generate(messages, sampling_params)
        return response["text"]

    def run_inference(
        self,
        qas_jsonl_path: str,
        output_path: str,
        max_examples: Optional[int] = None,
        sampling_params: Optional[dict] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run inference over a REAL-MM-RAG QA split and save predictions to JSON.

        The saved JSON is a list of records, each containing:
            example_index, doc_id, question, reference, prediction

        In 'no_rag' mode the oracle image from each QA entry is forwarded to the
        VLM, making this a strong upper-bound baseline.

        Args:
            qas_jsonl_path: Path to qas.jsonl.
            output_path:    Where to write the predictions JSON.
            max_examples:   Limit inference to the first N examples (None = all).
            sampling_params:Forwarded to VLM.generate().

        Returns:
            The list of prediction records (also written to output_path).
        """
        if sampling_params is None:
            sampling_params = {"max_new_tokens": 512}

        with open(qas_jsonl_path, "r", encoding="utf-8") as fh:
            examples = [json.loads(line) for line in fh]

        if max_examples is not None:
            examples = examples[:max_examples]

        n_total = len(examples)
        print(f"\nRunning inference on {n_total} examples | mode={self.mode}")
        print(f"  dataset : {qas_jsonl_path}")
        print(f"  output  : {output_path}\n")

        records: List[Dict[str, Any]] = []
        for i, ex in enumerate(examples):
            question = ex["question"]
            reference = ex["answer"]
            oracle_img = ex.get("image_path")

            try:
                prediction = self.generate(
                    query=question,
                    image_path=oracle_img,
                    sampling_params=sampling_params,
                )
            except Exception as exc:
                print(f"  ERROR on example {ex.get('example_index')}: {exc}")
                prediction = ""

            records.append(
                {
                    "example_index": ex.get("example_index"),
                    "doc_id": ex.get("doc_id"),
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "mode": self.mode,
                    "dataset": ex.get("dataset", ""),
                }
            )

            if verbose:
                print(f"[{i + 1}/{n_total}] {question[:80]}...")
                print(f"  Pred: {prediction[:120]!r}")


        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)
        print(f"\nSaved {len(records)} predictions → {output_path}")

        return records


# Example: run inference for all three modes on FinReport test split
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    print(f"Repo root: {repo_root}")
    data_root = repo_root / "data" / "raw"
    results_dir = repo_root / "results"

    VLM_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    VLM_CHECKPOINT = str(repo_root / "Qwen3-VL-2B-Instruct")

    DATASET = "REAL-MM-RAG_FinReport"
    qas_path = str(data_root / DATASET / "test" / "qas.jsonl")

    SAMPLING = {"max_new_tokens": 4096}
    MAX_EXAMPLES = None  # set to an int to limit during development

    # ---- Image RAG ---------------------------------------------------------
    image_pipeline = MMRAGPipeline(
        mode="image_rag",
        vlm_model_id=VLM_MODEL_ID,
        vlm_checkpoint_path=VLM_CHECKPOINT,
        image_retriever_persist_dir=str(repo_root / "scripts" /"embeddings"),
        top_k=3,
    )
    image_pipeline.ingest(qas_path)
    image_pipeline.run_inference(
        qas_jsonl_path=qas_path,
        output_path=str(results_dir / f"image_rag_{DATASET}.json"),
        max_examples=MAX_EXAMPLES,
        sampling_params=SAMPLING,
    )

    # ---- OCR RAG -----------------------------------------------------------
    ocr_pipeline = MMRAGPipeline(
        mode="ocr_rag",
        vlm_model_id=VLM_MODEL_ID,
        vlm_checkpoint_path=VLM_CHECKPOINT,
        ocr_retriever_persist_dir=str(repo_root / "scripts" / "embeddings_ocr"),
        top_k=3,
    )
    ocr_pipeline.ingest(qas_path)
    ocr_pipeline.run_inference(
        qas_jsonl_path=qas_path,
        output_path=str(results_dir / f"ocr_rag_{DATASET}.json"),
        max_examples=MAX_EXAMPLES,
        sampling_params=SAMPLING,
    )

    # ---- No-RAG (oracle image baseline) ------------------------------------
    no_rag_pipeline = MMRAGPipeline(
        mode="no_rag",
        vlm_model_id=VLM_MODEL_ID,
        vlm_checkpoint_path=VLM_CHECKPOINT,
    )
    no_rag_pipeline.run_inference(
        qas_jsonl_path=qas_path,
        output_path=str(results_dir / f"no_rag_{DATASET}.json"),
        max_examples=MAX_EXAMPLES,
        sampling_params=SAMPLING,
    )
