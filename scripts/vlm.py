import os
import torch
import time
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    from sglang import Engine
    _HAS_SGLANG = True
except Exception as e:
    Engine = None
    _SGLANG_IMPORT_ERROR = e
    _HAS_SGLANG = False

def prepare_model(model_name_or_path, local_dir):
    if os.path.exists(local_dir) and any(os.scandir(local_dir)):
        print(f"--- local model found at: {local_dir} ---")
    else:
        print(f"--- model not found locally. downloading to: {local_dir} ---")
        snapshot_download(
            repo_id=model_name_or_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            # revision="main"
        )
    return local_dir

class QwenVLDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        batch = {}
        first = features[0]
        
        if "input_ids" in first:
            input_ids = [f["input_ids"] for f in features]
            batch["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
            
        if "attention_mask" in first:
            attention_masks = [f["attention_mask"] for f in features]
            batch["attention_mask"] = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            
        if "labels" in first:
            labels = [f["labels"] for f in features]
            batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
                
        for k in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if k in first:
                tensors = [f[k] for f in features if k in f and f[k] is not None]
                if tensors:
                    batch[k] = torch.cat(tensors, dim=0)
                    
        return batch

class QwenVLDataset(Dataset):
    def __init__(self, df, processor, cache_dir="./tensor_cache"):
        self.data = df.reset_index()
        self.processor = processor
        self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        final_file = f"{self.cache_dir}/item_{len(self.data)-1}.pt"
        
        if not os.path.exists(final_file):
            print(f"\n--- preprocessing {len(self.data)} images to disk ---")
            
            for idx in tqdm(range(len(self.data)), desc="saving Tensors"):
                save_path = f"{self.cache_dir}/item_{idx}.pt"
                
                if os.path.exists(save_path):
                    continue
                    
                item = self.data.iloc[idx]
                
                messages = [
                    {
                        "role": "system", 
                        "content": [{"type": "text", "text": "You are a question answering assistant for corporate applications. Respond in one sentence using all available information."}]
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": f"../{item['intended_img']}", "max_pixels": 501760},
                            {"type": "text", "text": item['question']}
                        ]
                    },
                    {
                        "role": "assistant", 
                        "content": [{"type": "text", "text": item['answer']}]
                    }
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    max_length=1536,
                    truncation=True,
                    return_tensors="pt"
                )
                
                for k in ["input_ids", "attention_mask"]:
                    if k in inputs:
                        inputs[k] = inputs[k].squeeze(0)
                        
                inputs["labels"] = inputs["input_ids"].clone()
                inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100
                
                torch.save(inputs, save_path)
        else:
            print(f"\n--- found existing cached tensors in {self.cache_dir} ---")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.load(f"{self.cache_dir}/item_{idx}.pt", weights_only=False)

class VLM:
    def __init__(self, model_id, checkpoint_path, force_transformers=False):
        self.checkpoint_path = prepare_model(model_id, checkpoint_path)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)
        self.backend = None
        self.model = None
        self.transformer_model = None

        if _HAS_SGLANG and Engine is not None and not force_transformers:
            try:
                self.model = Engine(
                    model_path=self.checkpoint_path,
                    enable_multimodal=True,
                    mem_fraction_static=0.8,
                    tp_size=torch.cuda.device_count()
                )
                self.backend = "sglang"
            except ModuleNotFoundError as e:
                print(e)
                self.backend = "transformers"
        else:
            print("using transformers backend")
            self.backend = "transformers"

        if self.backend == "transformers":
            from transformers import Qwen3VLForConditionalGeneration
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.transformer_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.checkpoint_path, 
                device_map="auto",
                quantization_config=bnb_config
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
    
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.transformer_model.device)

        generated_ids = self.transformer_model.generate(
            **inputs, 
            max_new_tokens=sampling_params.get("max_new_tokens", 1024)
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"text": output_text[0] if isinstance(output_text, list) else output_text}
    
    def finetune(self, train_df, output_dir="./finetuned_qwen_adapter"):

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.transformer_model.gradient_checkpointing_enable()
        self.transformer_model = prepare_model_for_kbit_training(self.transformer_model)
        self.transformer_model = get_peft_model(self.transformer_model, lora_config)
        self.transformer_model.print_trainable_parameters()

        train_dataset = QwenVLDataset(train_df, self.processor)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,     
            gradient_accumulation_steps=2, 
            optim="paged_adamw_8bit",
            learning_rate=2e-4,
            num_train_epochs=3, 
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            remove_unused_columns=False,
            report_to="none",     
            dataloader_num_workers=2,
            dataloader_prefetch_factor=2,
            dataloader_pin_memory=True,
        )

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        trainer = Trainer(
            model=self.transformer_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=QwenVLDataCollator(pad_token_id=pad_token_id),
        )
        
        print("training...")
        self.transformer_model.train()
        trainer.train()
        
        print(f"saving fine-tuned adapter to {output_dir}")
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        
        self.transformer_model.eval()

    def load_adapter(self, adapter_path):

        print(f"loading LoRA adapter from {adapter_path}...")
        if hasattr(self.transformer_model, "load_adapter"):
            self.transformer_model.load_adapter(adapter_path, adapter_name="eval_adapter")
            self.transformer_model.set_adapter("eval_adapter")
        else:
            from peft import PeftModel
            self.transformer_model = PeftModel.from_pretrained(self.transformer_model, adapter_path)

        for name, param in self.transformer_model.named_parameters():
            if "lora_" in name and param.device.type == "cpu":
                param.data = param.data.to("cuda:0")

        self.transformer_model.eval()
    

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-VL-2B-Instruct" # HF Repo ID
    checkpoint_path = "./Qwen3-VL-2B-Instruct" # local destination folder

    vlm = VLM(model_id, checkpoint_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data/raw/REAL-MM-RAG_FinReport/test/pages/1_1.png"},
                {"type": "text", "text": "Was IBM's revenue strong in 2006?"}
            ],
        }
    ]
    
    response = vlm.generate(messages)
    print(f"Generated text: {response['text']}")    
