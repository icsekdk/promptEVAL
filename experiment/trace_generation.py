import csv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse_prompt import send_request
from scripts.save_results import write_summary_results, write_detailed_results,format_generated_traces_for_csv
from prompts.trace_generation_prompt import generate_prompt
from scripts.trace_satisfaction import check_trace
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import re
from typing import Tuple

# def extract_satisfying_falsifying_traces(response: str) -> Tuple[str, str]:
#     satisfying_trace = ""
#     falsifying_trace = ""
    
#     # Find SATISFYING section - extract everything after "SATISFYING:" until "FALSIFYING:"
#     satisfying_match = re.search(r'SATISFYING:\s*(.+?)(?=\s*FALSIFYING:|$)', response, re.DOTALL)
#     if satisfying_match:
#         satisfying_trace = satisfying_match.group(1).strip()
    
#     # Find FALSIFYING section - extract everything after "FALSIFYING:" until end or next section
#     falsifying_match = re.search(r'FALSIFYING:\s*(.+?)(?=\s*SATISFYING:|$)', response, re.DOTALL)
#     if falsifying_match:
#         falsifying_trace = falsifying_match.group(1).strip()
    
#     return satisfying_trace, falsifying_trace

# def extract_positive_negative_trace(response: str) -> Tuple[str, str]:

#     return extract_satisfying_falsifying_traces(response)
def extract_satisfying_falsifying_traces(response: str) -> Tuple[str, str]:
    satisfying_trace = ""
    falsifying_trace = ""
    
    # Updated regex patterns to match actual response format
    # Find Positive Trace section
    positive_match = re.search(r'Positive Trace:\s*(.+?)(?=\s*Negative Trace:|$)', response, re.DOTALL)
    if positive_match:
        satisfying_trace = positive_match.group(1).strip()
    
    # Find Negative Trace section
    negative_match = re.search(r'Negative Trace:\s*(.+?)(?=\s*Positive Trace:|$)', response, re.DOTALL)
    if negative_match:
        falsifying_trace = negative_match.group(1).strip()
    
    return satisfying_trace, falsifying_trace

def extract_positive_negative_trace(response: str) -> Tuple[str, str]:
    return extract_satisfying_falsifying_traces(response)

def process_stored_responses(raw_responses_csv, output_dir):
    """Process stored responses from CSV"""
    
    # Initialize results dictionaries
    trace_generation_summary_results = {}
    generated_traces_results = []
    
    # Read the raw responses
    raw_df = pd.read_csv(raw_responses_csv)
    
    for _, row in raw_df.iterrows():
        model = row["model"]
        approach = row["approach"]
        formula = row["formula"]
        initial_response = row["initial_response"]
        
        key = f"{model}_{approach}"
        if key not in trace_generation_summary_results:
            trace_generation_summary_results[key] = []
        
        # Process based on approach
        if approach == "zero_shot_self_refine":
            refined_response = row["refined_response"]
            positive_trace, negative_trace = extract_positive_negative_trace(refined_response)
            llm_response = format_llm_response(initial_response, refined_response)
        else:
            positive_trace, negative_trace = extract_positive_negative_trace(initial_response)
            llm_response = format_llm_response(initial_response)
        
        # Store processed results
        generated_traces_results.append({
            "Model": model,
            "Approach": approach,
            "LTL Formula": formula,
            "LLM Response": llm_response,
            "Positive Trace": positive_trace,
            "Negative Trace": negative_trace
        })
        
        trace_generation_summary_results[key].append({
            "LTL Formula": formula,
            "Positive Trace": positive_trace,
            "Negative Trace": negative_trace
        })
    
    # Save processed results
    write_detailed_results(output_dir / 'trace_generation_detailed.csv', generated_traces_results)
    write_summary_results(output_dir / 'trace_generation_summary.csv', 
                         trace_generation_summary_results, 
                         ['Positive Trace', 'Negative Trace'])

# Helper function to save individual raw responses (for debugging or inspection)
def save_raw_responses(model, approach, formula, response_data, output_dir):
    """Save individual raw responses to files for reference"""
    raw_dir = output_dir / 'raw_responses' / model / approach
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from the formula
    safe_formula = "".join(c if c.isalnum() else "_" for c in formula)
    if len(safe_formula) > 50:
        safe_formula = safe_formula[:50]
    
    with open(raw_dir / f"{safe_formula}.txt", 'w', encoding='utf-8') as f:
        f.write(f"Formula: {formula}\n\n")
        f.write(f"Initial Prompt:\n{response_data['initial_prompt']}\n\n")
        f.write(f"Initial Response:\n{response_data['initial_response']}\n\n")
        
        if approach == "zero_shot_self_refine":
            f.write(f"Refined Prompt:\n{response_data['refined_prompt']}\n\n")
            f.write(f"Refined Response:\n{response_data['refined_response']}\n\n")


def query_and_store_raw_responses(dataset, models, approaches, output_csv):
    """Query LLMs and store raw responses to CSV"""
    
    # Create CSV file with headers
    # headers = [
    #     "formula", "model", "approach", 
    #     "initial_prompt", "initial_response",
    #     "refined_prompt", "refined_response"
    # ]
    headers = [
        "formula", "model", "approach", 
      "initial_response", "refined_response"
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for model in models:
            for approach in approaches:
                print(f"Querying {model} with {approach} approach...")
                
                for _, row in dataset.iterrows():
                    formula = row["Ground Truth"]
                    
                    response_data = {
                        "formula": formula,
                        "model": model,
                        "approach": approach,
                        # "refined_prompt": "",
                        "refined_response": ""
                    }

                    if approach == "zero_shot_self_refine":
                        # Step 1: generate initial prompt & response
                        base_prompt = generate_prompt(formula, "zero_shot")
                        initial_response = send_request(base_prompt, model)

                        # Step 2: generate refined prompt based on that response
                        refined_prompt = generate_prompt(formula, "zero_shot_self_refine", initial_response)
                        refined_response = send_request(refined_prompt, model)

                        # Save both responses
                        # response_data["initial_prompt"] = base_prompt
                        response_data["initial_response"] = initial_response
                        # response_data["refined_prompt"] = refined_prompt
                        response_data["refined_response"] = refined_response
                    
                    else:
                        # Regular zero-shot or few-shot
                        initial_prompt = generate_prompt(formula, approach)
                        initial_response = send_request(initial_prompt, model)

                        # response_data["initial_prompt"] = initial_prompt
                        response_data["initial_response"] = initial_response

                    writer.writerow(response_data)


def format_llm_response(initial_response: str, refined_response: str = None) -> str:
    """Format LLM response combining initial and refined responses if available."""
    if refined_response:
        return f"Initial Response:\n{initial_response}\n\nRefined Response:\n{refined_response}"
    return initial_response

def analyze_results(results_file: str, metrics_output_file: str) -> None:
    """Analyze trace satisfiability results and save metrics to CSV."""
    df = pd.read_csv(results_file)
    
    # Initialize metrics storage
    all_metrics = []
    
    # Calculate metrics for each model and approach combination
    for model in df['Model'].unique():
        for approach in df['Approach'].unique():
            model_approach_df = df[(df['Model'] == model) & (df['Approach'] == approach)]
            
            if len(model_approach_df) == 0:
                continue
                
            # Convert results to binary format
            y_true = []
            y_pred = []
            
            # Positive traces should be SATISFIED
            positive_true = [1] * len(model_approach_df)
            positive_pred = [1 if result == 'SATISFIED' else 0 
                           for result in model_approach_df['Positive Result']]
            
            # Negative traces should be FALSIFIED
            negative_true = [0] * len(model_approach_df)
            negative_pred = [0 if result == 'FALSIFIED' else 1 
                           for result in model_approach_df['Negative Result']]
            
            y_true.extend(positive_true + negative_true)
            y_pred.extend(positive_pred + negative_pred)
            
            metrics = {
                'Model': model,
                'Approach': approach,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'F1': f1_score(y_true, y_pred),
                'Total_Traces': len(model_approach_df) * 2,  # Both positive and negative
                'Satisfied_Positive': sum(positive_pred),
                'Falsified_Negative': sum([1 - p for p in negative_pred]),
                'Errors': (
                    model_approach_df['Positive Result'].str.contains('ERROR').sum() +
                    model_approach_df['Negative Result'].str.contains('ERROR').sum()
                ),
                'Satisfied_Rate': sum(positive_pred) / len(model_approach_df),
                'Falsified_Rate': sum([1 - p for p in negative_pred]) / len(model_approach_df),
                'Error_Rate': (
                    (model_approach_df['Positive Result'].str.contains('ERROR').sum() +
                     model_approach_df['Negative Result'].str.contains('ERROR').sum()) /
                    (len(model_approach_df) * 2)
                )
            }
            
            all_metrics.append(metrics)
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(all_metrics)
    
    # Round numeric columns to 4 decimal places
    numeric_columns = metrics_df.select_dtypes(include=['float64']).columns
    metrics_df[numeric_columns] = metrics_df[numeric_columns].round(4)
    
    # Sort by F1 score (or any other metric you prefer)
    metrics_df = metrics_df.sort_values('F1', ascending=False)
    
    # Save to CSV
    metrics_df.to_csv(metrics_output_file, index=False, float_format="%.2f")
    
    # Print summary
    print("\nMetrics Summary:")
    print(metrics_df)
    
    # Calculate and print overall metrics
    print("\nOverall Metrics:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        mean_value = metrics_df[metric].mean()
        std_value = metrics_df[metric].std()
        print(f"Average {metric}: {mean_value:.4f} (±{std_value:.4f})")
def calculate_and_visualize_metrics(results_file: str, output_dir: Path):
    """Calculate metrics and create visualizations."""
    # Read results
    df = pd.read_csv(results_file)
    
    # Calculate metrics for each model-approach combination
    metrics_data = []
    
    for model in df['Model'].unique():
        for approach in df['Approach'].unique():
            model_approach_data = df[(df['Model'] == model) & (df['Approach'] == approach)]
            
            # Count total traces
            total_traces = len(model_approach_data) * 2  # Both positive and negative
            
            # Calculate various metrics
            positive_satisfaction = (model_approach_data['Positive Result'] == 'SATISFIED').sum()
            negative_falsification = (model_approach_data['Negative Result'] == 'FALSIFIED').sum()
            
            positive_errors = model_approach_data['Positive Result'].str.contains('ERROR').sum()
            negative_errors = model_approach_data['Negative Result'].str.contains('ERROR').sum()
            
            # Calculate accuracy metrics
            true_positives = positive_satisfaction
            true_negatives = negative_falsification
            false_positives = len(model_approach_data) - negative_falsification
            false_negatives = len(model_approach_data) - positive_satisfaction
            
            total_correct = true_positives + true_negatives
            
            metrics_data.append({
                'Model': model,
                'Approach': approach,
                'Total_Traces': total_traces,
                'Accuracy': (total_correct / total_traces) * 100,
                'Precision': (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0,
                'Recall': (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0,
                'F1_Score': (2 * true_positives / (2 * true_positives + false_positives + false_negatives)) * 100 if (2 * true_positives + false_positives + false_negatives) > 0 else 0,
                'Positive_Satisfaction_Rate': (positive_satisfaction / len(model_approach_data)) * 100,
                'Negative_Falsification_Rate': (negative_falsification / len(model_approach_data)) * 100,
                'Error_Rate': ((positive_errors + negative_errors) / total_traces) * 100,
                'Positive_Satisfaction_Count': positive_satisfaction,
                'Negative_Falsification_Count': negative_falsification,
                'Positive_Errors': positive_errors,
                'Negative_Errors': negative_errors
            })
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save metrics to CSV
    metrics_csv = output_dir / 'detailed_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False, float_format="%.2f")
    
    # Create visualizations
    create_visualizations(metrics_df, output_dir)

def create_visualizations(metrics_df: pd.DataFrame, output_dir: Path):
    """Create various visualizations of the metrics."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Main Metrics Comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    plot_data = pd.melt(metrics_df, 
                        id_vars=['Model', 'Approach'],
                        value_vars=metrics_to_plot,
                        var_name='Metric',
                        value_name='Score')
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=plot_data, x='Model', y='Score', hue='Metric')
    plt.title('Performance Metrics by Model')
    plt.xticks(rotation=45)
    plt.ylabel('Score (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Approach Comparison
    plt.figure(figsize=(15, 8))
    for metric in metrics_to_plot:
        plt.subplot(2, 2, metrics_to_plot.index(metric) + 1)
        sns.barplot(data=metrics_df, x='Approach', y=metric, hue='Model')
        plt.title(f'{metric} by Approach')
        plt.xticks(rotation=45)
        plt.ylabel('Score (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'approach_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error Analysis
    error_data = pd.melt(metrics_df,
                        id_vars=['Model', 'Approach'],
                        value_vars=['Positive_Errors', 'Negative_Errors'],
                        var_name='Error_Type',
                        value_name='Count')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=error_data, x='Model', y='Count', hue='Error_Type')
    plt.title('Error Analysis by Model')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Errors')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Success Rate Analysis
    success_data = pd.melt(metrics_df,
                          id_vars=['Model', 'Approach'],
                          value_vars=['Positive_Satisfaction_Rate', 'Negative_Falsification_Rate'],
                          var_name='Success_Type',
                          value_name='Rate')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=success_data, x='Model', y='Rate', hue='Success_Type')
    plt.title('Success Rates by Model')
    plt.xticks(rotation=45)
    plt.ylabel('Success Rate (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap of all metrics
    plt.figure(figsize=(15, 10))
    metrics_for_heatmap = metrics_df.drop(['Model', 'Approach'], axis=1)
    sns.heatmap(metrics_for_heatmap.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

# def main():
#     # Create output directory
#     output_dir = Path('output/ltl/trace_generation')
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     nl2ltl_dataset = load_data('data/nl2ltl_finalized.csv')

#     learning_approaches = ["zero_shot", "zero_shot_self_refine", "few_shot"]
#     models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-sonnet", "gemini"]
    
#     # Initialize results dictionaries
#     trace_generation_summary_results = {}
#     generated_traces_results = []
    
#     for model in models:
#         for approach in learning_approaches:
#             print(f"Running {model} with {approach} approach...")
#             key = f"{model}_{approach}"
#             if key not in trace_generation_summary_results:
#                 trace_generation_summary_results[key] = []
            
#             for _, row in nl2ltl_dataset.iterrows():
#                 formula = row["Ground Truth"]
                
#                 # Generate and save initial response
#                 initial_prompt = generate_prompt(formula, approach)
#                 initial_response = send_request(initial_prompt, model)
                
#                 # Save raw response immediately
#                 response_data = {
#                     "formula": formula,
#                     "model": model,
#                     "approach": approach,
#                     "initial_prompt": initial_prompt,
#                     "initial_response": initial_response,
#                 }
                
#                 if approach == "zero_shot_self_refine":
#                     # Generate and save refined response
#                     refined_prompt = generate_prompt(formula, approach, initial_response)
#                     refined_response = send_request(refined_prompt, model)
#                     response_data["refined_prompt"] = refined_prompt
#                     response_data["refined_response"] = refined_response
                    
#                     # Extract traces from refined response
#                     positive_trace, negative_trace = extract_positive_negative_trace(refined_response)
#                     # Format combined response
#                     llm_response = format_llm_response(initial_response, refined_response)
#                 else:
#                     # Extract traces from initial response
#                     positive_trace, negative_trace = extract_positive_negative_trace(initial_response)
#                     llm_response = format_llm_response(initial_response)
                
#                 # Save raw response data
#                 save_raw_responses(model, approach, formula, response_data, output_dir)
                
#                 # Store processed results
#                 generated_traces_results.append({
#                     "Model": model,
#                     "Approach": approach,
#                     "LTL Formula": formula,
#                     "LLM Response": llm_response,
#                     "Positive Trace": positive_trace,
#                     "Negative Trace": negative_trace
#                 })
                
#                 trace_generation_summary_results[key].append({
#                     "LTL Formula": formula,
#                     "Positive Trace": positive_trace,
#                     "Negative Trace": negative_trace
#                 })
    
#     # Save processed results
#     write_detailed_results(output_dir / 'trace_generation_detailed.csv', generated_traces_results)
#     write_summary_results(output_dir / 'trace_generation_summary.csv', 
#                          trace_generation_summary_results, 
#                          ['Positive Trace', 'Negative Trace'])
    
#     # Get base directory for ltl_utils
#     base_dir = Path("/Users/priscilladanso/Documents/StonyBrook/TowardsDissertation/LLM4NL2LTL_project")
#     ltl_utils_dir = base_dir / "LTL" / "corrected_version" / "ltlutils"
    
#     # Check trace satisfiability with correct parameters
#     check_trace(
#         input_csv=output_dir / 'trace_generation_detailed.csv',
#         output_csv=output_dir / 'llm_trace_satisfaction_results.csv',
#         ltl_utils_dir=ltl_utils_dir
#     )
    
#     # Define paths
#     output_csv = output_dir / "llm_trace_satisfaction_results.csv"
#     metrics_csv = output_dir / "metrics_results.csv"
    
#     # Analyze results and save metrics to CSV
#     analyze_results(output_csv, metrics_csv)
    
#     calculate_and_visualize_metrics(output_csv, output_dir)


from pathlib import Path
import pandas as pd
import csv
import os

def main():
    # Create output directory
    output_dir = Path('/Users/priscilladanso/Library/Mobile Documents/com~apple~CloudDocs/Documents/StonyBrook/TowardsDissertation/PYTHON4LTL/output/trace_generation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nl2ltl_dataset = pd.read_csv('dataset/nl2ltl_finalized.csv')

    learning_approaches = ["zero_shot", "zero_shot_self_refine", "few_shot"]
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-sonnet", "gemini"]
    
    # File to store raw responses
    raw_responses_csv = output_dir / 'raw_llm_responses.csv'
    
    # Phase 1: Query LLMs and store raw responses
    if not os.path.exists(raw_responses_csv):
        print("Phase 1: Querying LLMs and storing raw responses...")
        query_and_store_raw_responses(nl2ltl_dataset, models, learning_approaches, raw_responses_csv)
    else:
        print(f"Raw responses file {raw_responses_csv} already exists. Skipping query phase.")
    
    # Phase 2: Process stored responses
    print("Phase 2: Processing stored responses...")
    process_stored_responses(raw_responses_csv, output_dir)
    project_root = Path(__file__).resolve().parents[2] 
    print(f"Project root directory: {project_root}")
    # Get base directory for ltl_utils
    # base_dir = Path("/Users/priscilladanso/Documents/StonyBrook/TowardsDissertation/LLM4NL2LTL_project")
    ltl_utils_dir = "/Users/priscilladanso/Library/Mobile Documents/com~apple~CloudDocs/Documents/StonyBrook/TowardsDissertation/LLM4NL2LTL_project/llm_LTL/LTL/corrected_version/ltlutils"
    # project_root = Path(__file__).resolve().parents[2]
    # ltl_utils_dir = project_root / "LLM4NL2LTL_project" / "llm_LTL" / "LTL" / "corrected_version" / "ltlutils"

    # Check trace satisfiability with correct parameters
    check_trace(
        input_csv=output_dir / 'trace_generation_detailed.csv',
        output_csv=output_dir / 'llm_trace_satisfaction_results.csv',
        ltl_utils_dir=ltl_utils_dir
    )
    
    # Define paths for analysis
    output_csv = output_dir / "llm_trace_satisfaction_results.csv"
    metrics_csv = output_dir / "metrics_results.csv"
    
    # Analyze results and save metrics to CSV
    analyze_results(output_csv, metrics_csv)
    
    calculate_and_visualize_metrics(output_csv, output_dir)

if __name__ == "__main__":
    main()