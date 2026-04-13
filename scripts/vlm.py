import os
import torch
import time
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from huggingface_hub import snapshot_download

# sglang can require native deps (triton). Try importing Engine but allow
# a graceful fallback so the script can run on platforms without triton
# (e.g. macOS without CUDA). If Engine fails to import/initialize, we'll
# fall back to a Transformers-based inference path below.
try:
    from sglang import Engine
    _HAS_SGLANG = True
except Exception as e:
    Engine = None
    _SGLANG_IMPORT_ERROR = e
    _HAS_SGLANG = False

def prepare_model(model_name_or_path, local_dir):
    """Checks for local files; if missing, downloads from Hugging Face."""
    if os.path.exists(local_dir) and any(os.scandir(local_dir)):
        print(f"--- Local model found at: {local_dir} ---")
    else:
        print(f"--- Model not found locally. Downloading to: {local_dir} ---")
        snapshot_download(
            repo_id=model_name_or_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            # revision="main" # Optional: specify a branch or commit hash
        )
    return local_dir

class VLM:
    def __init__(self, model_id, checkpoint_path):
        self.checkpoint_path = prepare_model(model_id, checkpoint_path)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)

        # Try to initialize sglang Engine backend; if unavailable (commonly
        # due to missing `triton` on macOS), fall back to a Transformers
        # based model implementation.
        self.backend = None
        self.model = None
        self.transformer_model = None

        if _HAS_SGLANG and Engine is not None:
            try:
                self.model = Engine(
                    model_path=self.checkpoint_path,
                    enable_multimodal=True,
                    mem_fraction_static=0.8,
                    tp_size=torch.cuda.device_count()
                )
                self.backend = "sglang"
            except ModuleNotFoundError as e:
                print("Warning: sglang Engine failed to initialize (missing native dependency). Falling back to Transformers backend.")
                print("  Import error:", e)
                self.backend = "transformers"
        else:
            print("Info: sglang Engine not available. Falling back to Transformers backend.")
            if not _HAS_SGLANG:
                print("  sglang import error:", getattr(__import__('builtins'), '_SGLANG_IMPORT_ERROR', None) or _SGLANG_IMPORT_ERROR)
            self.backend = "transformers"

        if self.backend == "transformers":
            # Attempt to load a Transformers-based multimodal model. Qwen's
            # dedicated class may be available; otherwise fall back to
            # AutoModelForCausalLM (note: image support may be limited).
            try:
                from transformers import Qwen3VLForConditionalGeneration

                self.transformer_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.checkpoint_path, dtype="auto", device_map="auto"
                )
            except Exception:
                from transformers import AutoModelForCausalLM

                try:
                    self.transformer_model = AutoModelForCausalLM.from_pretrained(
                        self.checkpoint_path, device_map="auto"
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to initialize a Transformers fallback model.\n"
                        "If you want to use the sglang Engine, install the native dependency 'triton' on a CUDA-enabled Linux environment, or run on a machine with a supported GPU.\n"
                        f"Original error: {e}"
                    )
            self.transformer_model.eval()

    def generate(self, messages, sampling_params={"max_new_tokens": 1024}):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if self.backend == "sglang":
            image_inputs, _ = process_vision_info(messages, image_patch_size=self.processor.image_processor.patch_size)
            response = self.model.generate(prompt=text, image_data=image_inputs, sampling_params=sampling_params)
            return response

        # Transformers fallback: prepare tokenized inputs and run .generate()
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.transformer_model.device)
        generated_ids = self.transformer_model.generate(**inputs, max_new_tokens=sampling_params.get("max_new_tokens", 1024))
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"text": output_text[0] if isinstance(output_text, list) else output_text}
    

if __name__ == "__main__":
    # VLM settings
    model_id = "Qwen/Qwen3-VL-2B-Instruct" # HF Repo ID
    checkpoint_path = "./Qwen3-VL-2B-Instruct" # local destination folder

    # Initialize VLM
    vlm = VLM(model_id, checkpoint_path)

    # Example input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data/raw/REAL-MM-RAG_FinReport/test/pages/1_1.png"},
                {"type": "text", "text": "Was IBM's revenue strong in 2006?"}
            ],
        }
    ]
    
    # Generate response
    start = time.time()
    response = vlm.generate(messages)
    print(f"Response costs: {time.time() - start:.2f}s")
    print(f"Generated text: {response['text']}")    

# if __name__ == "__main__":
#     # Settings
#     model_id = "Qwen/Qwen3-VL-8B-Instruct" # The HF Repo ID
#     checkpoint_path = "./Qwen3-VL-8B-Instruct" # Your local destination folder

#     # Step 1: Ensure model exists locally
#     prepare_model(model_id, checkpoint_path)

#     # Step 2: Initialize Processor
#     processor = AutoProcessor.from_pretrained(checkpoint_path)

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/receipt.png",
#                 },
#                 {"type": "text", "text": "Read all the text in the image."},
#             ],
#         }
#     ]

#     text = processor.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     image_inputs, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)

#     # Step 3: Initialize SGLang Engine
#     llm = Engine(
#         model_path=checkpoint_path,
#         enable_multimodal=True,
#         mem_fraction_static=0.8,
#         tp_size=torch.cuda.device_count(),
#         attention_backend="fa3"
#     )

#     # Step 4: Inference
#     start = time.time()
#     sampling_params = {"max_new_tokens": 1024}
#     response = llm.generate(prompt=text, image_data=image_inputs, sampling_params=sampling_params)
    
#     print(f"Response costs: {time.time() - start:.2f}s")
#     print(f"Generated text: {response['text']}")

# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# # default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
# )

# # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# # model = Qwen3VLForConditionalGeneration.from_pretrained(
# #     "Qwen/Qwen3-VL-4B-Instruct",
# #     dtype=torch.bfloat16,
# #     attn_implementation="flash_attention_2",
# #     device_map="auto",
# # )

# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt"
# )
# inputs = inputs.to(model.device)

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)
