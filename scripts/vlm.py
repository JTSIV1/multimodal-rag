import os
import torch
import time
from PIL import Image
from sglang import Engine
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from huggingface_hub import snapshot_download

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
        self.model = Engine(
            model_path=self.checkpoint_path,
            enable_multimodal=True,
            mem_fraction_static=0.8,
            tp_size=torch.cuda.device_count()
        )

    def generate(self, messages, sampling_params={"max_new_tokens": 1024}):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages, image_patch_size=self.processor.image_processor.patch_size)
        response = self.model.generate(prompt=text, image_data=image_inputs, sampling_params=sampling_params)
        return response    
    

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
                {
                    "type": "image",
                    "image": "data/raw/REAL-MM-RAG-FinReport/1_1.png",
                },
                {"type": "text", "text": "Was IBM's revenue strong in 2006?"},
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
