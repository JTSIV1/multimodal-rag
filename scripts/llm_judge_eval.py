import json
import base64
import os
from together import Together
from dotenv import load_dotenv

load_dotenv()

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

import imghdr

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        data = image_file.read()
    fmt = imghdr.what(None, h=data) or "jpeg"
    mime = f"image/{fmt}"
    b64 = base64.b64encode(data).decode('utf-8')
    return mime, b64

# The rubric defined above
SYSTEM_PROMPT = """
You are an expert AI evaluator grading the performance of a Retrieval-Augmented Generation (RAG) system. 
You will be provided with a Question, a Ground Truth Answer, the Generated Answer, and the Retrieved Context (which may be text or images).

Your task is to evaluate the Generated Answer across four categories on a scale of 1 to 5. 

### Categories:
1. Answer Accuracy: How well does the Generated Answer match the Ground Truth? 
   - 1: Completely incorrect or contradicts the ground truth.
   - 3: Partially correct but missing key details.
   - 5: Perfectly matches the ground truth semantics and facts.
2. Answer Relevance: How directly does the Generated Answer address the Question?
   - 1: Off-topic or fails to answer the question.
   - 3: Answers the question but includes unnecessary filler.
   - 5: Directly and concisely answers the question.
3. Groundedness: To what extent are the claims in the Generated Answer supported by the Retrieved Context?
   - 1: None of the claims are found in the context.
   - 3: Some claims are supported, but others are missing from the context.
   - 5: Every claim in the answer is explicitly backed by the context.
4. Faithfulness: Does the Generated Answer avoid hallucinations and remain true to the context?
   - 1: Hallucinates external facts or directly contradicts the context.
   - 3: Mostly faithful, but makes minor logical leaps not supported by the context.
   - 5: Strictly adheres to the context with zero hallucinations or contradictions.

### Output Format
You must output ONLY valid JSON in the following format. Do not include markdown blocks or any other text.
{
  "accuracy_reason": "<brief justification>",
  "accuracy_score": <int>,
  "relevance_reason": "<brief justification>",
  "relevance_score": <int>,
  "groundedness_reason": "<brief justification>",
  "groundedness_score": <int>,
  "faithfulness_reason": "<brief justification>"
  "faithfulness_score": <int>,
}
"""

_EXPERIMENTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
)

def _resolve_path(p: str) -> str:
    """Resolve a path that may be relative to the experiments/ directory."""
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    cand = os.path.normpath(os.path.join(_EXPERIMENTS_DIR, p))
    return cand if os.path.exists(cand) else p


def _read_ocr_text(img_path: str) -> str:
    """Find and read the OCR .txt file corresponding to an image path."""
    img_path = _resolve_path(img_path)
    if not img_path:
        return "No context found."
    candidates = [
        img_path + ".txt",
        os.path.splitext(img_path)[0] + ".txt",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip() or "No context found."
            except Exception:
                pass
    return "No context found."


# def create_batch_file(model_id, results_file, output_batch_file, retrieval_files=None, is_vision=False, no_context=False):
#     """Create a JSONL batch file for Together AI evaluation.

#     retrieval_files: list of retrieval JSON paths to merge, or None to derive
#                      context directly from the result record (ideal baselines).
#     no_context:      if True, include a placeholder indicating no retrieval was done.
#     """
#     with open(results_file, 'r') as f:
#         results = [json.loads(line) for line in f]

#     # Merge retrieval records from one or more files; key by (dataset, example_index)
#     retrieval_dict = {}
#     if retrieval_files:
#         for rf in retrieval_files:
#             with open(rf) as f:
#                 for item in json.load(f):
#                     retrieval_dict[(item['dataset'], item['example_index'])] = item

#     with open(output_batch_file, 'w') as out_f:
#         for res in results:
#             idx = res['example_index']
#             dataset = res['dataset']
#             retrieval_data = retrieval_dict.get((dataset, idx), {})

#             user_content = [
#                 {"type": "text", "text": f"Question: {res['question']}"},
#                 {"type": "text", "text": f"Ground Truth: {res['ground_truth']}"},
#                 {"type": "text", "text": f"Generated Answer: {res['generated_answer']}"},
#             ]

#             if no_context:
#                 user_content.append({"type": "text", "text": "Retrieved Context: None (no retrieval was performed)"})
#             elif is_vision:
#                 user_content.append({"type": "text", "text": "Retrieved Context (Images Follow):"})
#                 if retrieval_files:
#                     img_paths = retrieval_data.get('retrieved_images', [])
#                 else:
#                     # Ideal baseline: the ground-truth image was given directly
#                     intended = res.get('intended_img', '')
#                     img_paths = [intended] if intended else []
#                 for img_path in img_paths:
#                     resolved = _resolve_path(img_path)
#                     if os.path.exists(resolved):
#                         mime, b64 = encode_image(resolved)
#                         user_content.append({
#                             "type": "image_url",
#                             "image_url": {"url": f"data:{mime};base64,{b64}"}
#                         })
#             else:
#                 if retrieval_files:
#                     context_text = retrieval_data.get('retrieved_prompt', 'No context found.')
#                 else:
#                     # Ideal text baseline: read OCR of the ground-truth image
#                     context_text = _read_ocr_text(res.get('intended_img', ''))
#                 user_content.append({"type": "text", "text": f"Retrieved Context: {context_text}"})

#             request = {
#                 "custom_id": f"eval_{dataset}_{idx}",
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": model_id,
#                     "messages": [
#                         {"role": "system", "content": SYSTEM_PROMPT},
#                         {"role": "user", "content": user_content}
#                     ],
#                     "temperature": 0.7,
#                 }
#             }
#             out_f.write(json.dumps(request) + '\n')

#     print(f"  Written {len(results)} requests → {output_batch_file}")

def create_batch_file(model_id, results_file, output_batch_file, retrieval_files=None, is_vision=False, no_context=False):
    """Create a JSONL batch file for Together AI evaluation.

    retrieval_files: list of retrieval JSON paths to merge, or None to derive
                     context directly from the result record (ideal baselines).
    no_context:      if True, include a placeholder indicating no retrieval was done.
    """
    MAX_FILE_BYTES = 95_000_000

    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]

    # Merge retrieval records from one or more files; key by (dataset, example_index)
    retrieval_dict = {}
    if retrieval_files:
        for rf in retrieval_files:
            with open(rf) as f:
                for item in json.load(f):
                    retrieval_dict[(item['dataset'], item['example_index'])] = item

    # Prepare file naming and chunking logic
    base_name, ext = os.path.splitext(output_batch_file)
    file_index = 0
    current_file_path = f"{base_name}_{file_index}{ext}"
    out_f = open(current_file_path, 'w', encoding='utf-8')
    current_byte_size = 0

    generated_files = [current_file_path]

    for res in results:
        idx = res['example_index']
        dataset = res['dataset']
        retrieval_data = retrieval_dict.get((dataset, idx), {})

        user_content = [
            {"type": "text", "text": f"Question: {res['question']}"},
            {"type": "text", "text": f"Ground Truth: {res['ground_truth']}"},
            {"type": "text", "text": f"Generated Answer: {res['generated_answer']}"},
        ]

        if no_context:
            user_content.append({"type": "text", "text": "Retrieved Context: None (no retrieval was performed)"})
        elif is_vision:
            user_content.append({"type": "text", "text": "Retrieved Context (Images Follow):"})
            if retrieval_files:
                img_paths = retrieval_data.get('retrieved_images', [])
            else:
                # Ideal baseline: the ground-truth image was given directly
                intended = res.get('intended_img', '')
                img_paths = [intended] if intended else []
            for img_path in img_paths:
                resolved = _resolve_path(img_path)
                if os.path.exists(resolved):
                    mime, b64 = encode_image(resolved)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })
        else:
            if retrieval_files:
                context_text = retrieval_data.get('retrieved_prompt', 'No context found.')
            else:
                # Ideal text baseline: read OCR of the ground-truth image
                context_text = _read_ocr_text(res.get('intended_img', ''))
            user_content.append({"type": "text", "text": f"Retrieved Context: {context_text}"})

        request = {
            "custom_id": f"eval_{dataset}_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.7,
            }
        }
        
        # Serialize and encode to get accurate byte size
        json_line = json.dumps(request) + '\n'
        line_bytes = json_line.encode('utf-8')
        line_byte_size = len(line_bytes)

        # Check if adding this line exceeds the 95MB limit
        if current_byte_size + line_byte_size > MAX_FILE_BYTES:
            out_f.close()
            file_index += 1
            current_file_path = f"{base_name}_{file_index}{ext}"
            out_f = open(current_file_path, 'w', encoding='utf-8')
            current_byte_size = 0
            generated_files.append(current_file_path)
            
        out_f.write(json_line)
        current_byte_size += line_byte_size

    out_f.close()

    if file_index == 0:
        print(f"  Written {len(results)} requests → {base_name}_0{ext}")
    else:
        print(f"  Written {len(results)} requests → chunked across {file_index + 1} files ({base_name}_0{ext} to {base_name}_{file_index}{ext})")

    return generated_files


# ── Configuration ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    MODEL_ID = "google/gemma-4-31B-it"


    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    _results_base = os.path.join(_scripts_dir, "..", "experiments", "results")
    BATCHES_DIR = os.path.join(_scripts_dir, "batches")
    os.makedirs(BATCHES_DIR, exist_ok=True)

    EXPERIMENTS = [
        # dict(
        #     name="baseline_no_rag",
        #     results=f"{_results_base}/baseline_no_rag/baseline_no_rag_results.jsonl",
        #     retrieval_files=None,
        #     is_vision=False,
        #     no_context=True,
        # ),
        dict(
            name="baseline_ideal_image",
            results=f"{_results_base}/baseline_ideal_rag_image/baseline_ideal_rag_image_results.jsonl",
            retrieval_files=None,
            is_vision=True,
            no_context=False,
        ),
        dict(
            name="baseline_ideal_text",
            results=f"{_results_base}/baseline_ideal_rag_text/baseline_ideal_rag_text_results.jsonl",
            retrieval_files=None,
            is_vision=False,
            no_context=False,
        ),
        dict(
            name="image_rag",
            results=f"{_results_base}/image_rag/image_rag_results.jsonl",
            retrieval_files=[
                f"{_results_base}/image_rag/retrieval_finreport.json",
                f"{_results_base}/image_rag/retrieval_techreport.json",
                f"{_results_base}/image_rag/retrieval_techslides.json",
            ],
            is_vision=True,
            no_context=False,
        ),
        dict(
            name="ocr_rag",
            results=f"{_results_base}/ocr_rag/ocr_rag_results.jsonl",
            retrieval_files=[
                f"{_results_base}/ocr_rag/retrieval_finreport.json",
                f"{_results_base}/ocr_rag/retrieval_techreport.json",
                f"{_results_base}/ocr_rag/retrieval_techslides.json",
            ],
            is_vision=False,
            no_context=False,
        ),
    ]

    # ── Run ──────────────────────────────────────────────────────────────────────

    batch_ids = {}
    for exp in EXPERIMENTS:
        base_batch_file = os.path.join(BATCHES_DIR, f"batch_{exp['name']}.jsonl")
        print(f"\n[{exp['name']}] Creating batch files...")
        
        chunked_files = create_batch_file(
            model_id=MODEL_ID,
            results_file=exp['results'],
            output_batch_file=base_batch_file,
            retrieval_files=exp.get('retrieval_files'),
            is_vision=exp['is_vision'],
            no_context=exp.get('no_context', False),
        )

        batch_ids[exp['name']] = []
        
        for i, chunk_file in enumerate(chunked_files):
            print(f"[{exp['name']} - Chunk {i}] Uploading {os.path.basename(chunk_file)}...")
            file_response = client.files.upload(file=chunk_file, purpose="batch-api", check=False)

            print(f"[{exp['name']} - Chunk {i}] Submitting batch job...")
            batch_response = client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
            )

    print("\n=== All batch jobs submitted ===")
        
    print("\nCheck status: client.batches.retrieve('<job_id>')")