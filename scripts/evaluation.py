import pandas as pd
import numpy as np
import warnings
from scipy import stats
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as b_score

# Ensure nltk tokenizer is available (uncomment if needed)
# nltk.download('punkt')

warnings.filterwarnings('ignore')

def calculate_bleu(reference, hypothesis):
    """Calculates smoothed sentence-level BLEU score."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        return 0.0
    
    # Simple whitespace tokenization for BLEU
    ref_tokens = str(reference).split()
    hyp_tokens = str(hypothesis).split()
    
    # Use smoothing method 1 to avoid 0 scores for short sentences without exact n-gram matches
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)

def calculate_rouge(reference, hypothesis):
    """Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(str(reference), str(hypothesis))
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bertscore(references, hypotheses, lang='en', model_type=None):
    """Calculates BERTScore F1 for a list of strings."""
    references = [str(r) if pd.notna(r) else "" for r in references]
    hypotheses = [str(h) if pd.notna(h) else "" for h in hypotheses]
    
    # BERTScore processes batches significantly faster than single items
    P, R, F1 = b_score(hypotheses, references, lang=lang, model_type=model_type, verbose=False)
    return F1.tolist()

def evaluate_dataframe(df, output_filepath):
    """
    Evaluates a dataframe with 'dataset', 'ground_truth', and 'generated_answer' columns.
    Writes the aggregated metrics and significance tests to a text file.
    """
    required_cols = ['dataset', 'ground_truth', 'generated_answer']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Calculating BLEU scores...")
    df['bleu'] = df.apply(lambda row: calculate_bleu(row['ground_truth'], row['generated_answer']), axis=1)
    
    print("Calculating ROUGE scores...")
    rouge_results = df.apply(lambda row: calculate_rouge(row['ground_truth'], row['generated_answer']), axis=1)
    df['rouge1'] = [res['rouge1'] for res in rouge_results]
    df['rouge2'] = [res['rouge2'] for res in rouge_results]
    df['rougeL'] = [res['rougeL'] for res in rouge_results]
    
    print("Calculating BERT scores (this may take a moment)...")
    df['bert_f1'] = calculate_bertscore(df['ground_truth'].tolist(), df['generated_answer'].tolist())

    metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_f1']
    datasets = df['dataset'].unique()
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================\n")
        f.write("      MODEL EVALUATION REPORT\n")
        f.write("=========================================\n\n")
        
        # OVERALL SCORES
        f.write("--- OVERALL METRICS ---\n")
        f.write(f"Total Samples: {len(df)}\n")
        for m in metrics:
            mean_val = df[m].mean()
            std_val = df[m].std()
            # 95% Confidence Interval for the mean
            ci = stats.t.interval(0.95, len(df)-1, loc=mean_val, scale=stats.sem(df[m].dropna()))
            f.write(f"{m.upper():<10}: {mean_val:.4f} ± {std_val:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])\n")
        f.write("\n")
        
        # PER DATASET SCORES
        f.write("--- METRICS BY DATASET ---\n")
        grouped = df.groupby('dataset')
        for name, group in grouped:
            f.write(f"Dataset: {name} (N={len(group)})\n")
            for m in metrics:
                mean_val = group[m].mean()
                std_val = group[m].std()
                f.write(f"  {m.upper():<10}: {mean_val:.4f} ± {std_val:.4f}\n")
            f.write("\n")
            
        # SIGNIFICANCE TESTING (Are datasets significantly different in performance?)
        f.write("--- SIGNIFICANCE TESTS (Performance Across Datasets) ---\n")
        f.write("Using Kruskal-Wallis H-test (non-parametric ANOVA) to test if scores differ by dataset.\n")
        
        if len(datasets) > 1:
            for m in metrics:
                dataset_scores = [group[m].dropna().values for name, group in grouped]
                # Only run test if we have at least 2 datasets with variations
                if all(len(d) > 0 for d in dataset_scores):
                    stat, p_val = stats.kruskal(*dataset_scores)
                    sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    f.write(f"{m.upper()}: H-stat={stat:.4f}, p-value={p_val:.4e} [{sig_level}]\n")
            f.write("(* p<0.05, ** p<0.01, *** p<0.001, ns = not significant)\n")
        else:
            f.write("Only one dataset found. Skipping across-dataset significance tests.\n")
            
    print(f"Report successfully saved to {output_filepath}")
    return df

# Example Usage
if __name__ == '__main__':
    data = {
        'dataset': ['Finance', 'Finance', 'Tech', 'Tech'],
        'ground_truth': ['The revenue increased by 20%.', 'IBM acquired Red Hat.', 'The CPU runs at 3GHz.', 'RAM is 16GB.'],
        'generated_answer': ['Revenue went up 20 percent.', 'IBM bought Red Hat.', 'It is a 3GHz CPU.', 'Memory is 16GB.']
    }
    df = pd.DataFrame(data)
    evaluated_df = evaluate_dataframe(df, 'evaluation_report.txt')