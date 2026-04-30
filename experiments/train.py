import os
import sys

# Environment variables must be set before heavy imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.abspath('/home/ec2-user/multimodal-rag'))

import torch
import nest_asyncio
import pandas as pd
import json
import glob
from tqdm import tqdm  # Terminal-friendly tqdm
from scripts.vlm import VLM
from scripts.evaluation import evaluate_dataframe

def main():
    torch.cuda.empty_cache()
    nest_asyncio.apply()

    # ==========================================
    # Configurations
    # ==========================================
    SEED = 19
    QUESTION_PATHS = {
        "FinancialReport" : "../data/raw/REAL-MM-RAG_FinReport/test/qas.jsonl",
        "TechReport" : "../data/raw/REAL-MM-RAG_TechReport/test/qas.jsonl",
        "TechSlides" : "../data/raw/REAL-MM-RAG_TechSlides/test/qas.jsonl"
    }
    RESULTS_PATH = "./results/baseline_no_rag"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    checkpoint_path = "../Qwen3-VL-8B-Instruct"
    adapter_dir = "../finetuned_qwen_adapter"

    # ==========================================
    # Load Data
    # ==========================================
    print("Loading datasets...")
    dataframes = {}

    for key, path in QUESTION_PATHS.items():
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append({
                    "example_index": obj.get("example_index"),
                    "question": obj.get("question"),
                    "answer": obj.get("answer"),
                    "intended_img": obj.get("image_path")
                })
        dataframes[key] = pd.DataFrame(records).set_index("example_index")

    # Split into test and train
    splits = {}
    for key, df in dataframes.items():
        train = df.sample(frac=0.8, random_state=SEED)
        test = df.drop(train.index)
        splits[key] = {"train": train, "test": test}

    train_frames = [split['train'] for split in splits.values()]
    combined_train_df = pd.concat(train_frames)
    print(f"Total training examples: {len(combined_train_df)}")

    # ==========================================
    # Fine-Tuning
    # ==========================================
    # Force transformers backend for training
    vlm = VLM(model_id, checkpoint_path, force_transformers=True)

    # Run fine-tuning (this updates vlm.transformer_model in-place with the trained adapter)
    vlm.finetune(combined_train_df, output_dir=adapter_dir)

    # ==========================================
    # Inference Loop
    # ==========================================
    checkpoints = sorted(glob.glob(f"{adapter_dir}/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))

    if not checkpoints:
        print(f"No checkpoints found in {adapter_dir}. Did fine-tuning complete?")
        
    os.makedirs(RESULTS_PATH, exist_ok=True)

    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n{'='*50}\nEvaluating {checkpoint_name}\n{'='*50}")
        
        # Load this specific epoch's weights
        vlm.load_adapter(checkpoint_path)
        
        output_filepath = f"{RESULTS_PATH}/results_{checkpoint_name}.jsonl"
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            for key, split in splits.items():
                test_df = split['test']
                print(f"  -> Running Inference for {key}")
                
                for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": "You are a question answering assistant for corporate applications. Respond in one sentence using all available information."}]},
                        {"role": "user", "content": [
                            {"type": "image", "image": f"../{row['intended_img']}", "max_pixels": 1003520},
                            {"type": "text", "text": row['question']}
                        ]}
                    ]
                    
                    try:
                        response = vlm.generate(messages)
                    except Exception as e:
                        print(f"Error on {idx}: {e}")
                        continue
                    
                    result_obj = {
                        "checkpoint": checkpoint_name,
                        "dataset": key,
                        "example_index": idx,
                        "question": row['question'],
                        "generated_answer": response['text'],
                        "ground_truth": row['answer'],
                        "intended_img": row['intended_img']
                    }
                    
                    f.write(json.dumps(result_obj) + "\n")
                    f.flush() 

        print(f"✅ Saved results for {checkpoint_name} to {output_filepath}")

    # ==========================================
    # Evaluation Comparison
    # ==========================================
    checkpoint_results_files = glob.glob(f"{RESULTS_PATH}/results_checkpoint-*.jsonl")
    checkpoint_results_files = sorted(checkpoint_results_files, key=lambda x: int(x.split("checkpoint-")[-1].split(".")[0]))

    all_evaluations = {}

    for input_filepath in checkpoint_results_files:
        checkpoint_name = os.path.basename(input_filepath).replace("results_", "").replace(".jsonl", "")
        report_filepath = f"{RESULTS_PATH}/{checkpoint_name}_evaluation_report.txt"
        detailed_jsonl_filepath = f"{RESULTS_PATH}/{checkpoint_name}_evaluated_details.jsonl"
        
        print(f"\nLoading data for {checkpoint_name}...")
        records = []
        with open(input_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    
        df_results = pd.DataFrame(records)
        
        print(f"Evaluating {checkpoint_name}...")
        # evaluate_dataframe processes the df, adds metric columns, and writes the txt report
        evaluated_df = evaluate_dataframe(df_results, report_filepath)
        
        # Save the detailed row-by-row scores just in case you want to inspect specific failures later
        evaluated_df.to_json(detailed_jsonl_filepath, orient="records", lines=True)
        
        # Extract the aggregate scores based on evaluate.py's column names
        if 'bert_f1' in evaluated_df.columns:
            mean_bert = evaluated_df['bert_f1'].mean()
            mean_rougeL = evaluated_df['rougeL'].mean()
            
            all_evaluations[checkpoint_name] = {
                'BERT_F1': mean_bert,
                'ROUGE-L': mean_rougeL
            }
            print(f"✅ {checkpoint_name} -> BERT_F1: {mean_bert:.4f} | ROUGE-L: {mean_rougeL:.4f}")
        else:
            print(f"⚠️ Warning: Expected metric columns not found in {checkpoint_name}.")

    print("\n" + "="*50)
    print("🏆 FINAL CHECKPOINT COMPARISON")
    print("="*50)
    print(f"{'Checkpoint':<20} | {'BERT_F1':<10} | {'ROUGE-L':<10}")
    print("-" * 46)
    for ckpt, scores in all_evaluations.items():
        print(f"{ckpt:<20} | {scores['BERT_F1']:.4f}     | {scores['ROUGE-L']:.4f}")

if __name__ == "__main__":
    main()