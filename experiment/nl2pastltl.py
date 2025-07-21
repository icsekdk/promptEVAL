import pandas as pd
import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now your imports should work
from parse_prompt import send_request
from prompts.ltl_past_prompt_template import generate_prompt
from collections import defaultdict
import csv
from pathlib import Path
import re
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_detailed_results(filename, results):
    if not results:
        print(f"No results to write to {filename}")
        return
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
def format_nl2ltl_results_for_csv(results, output_file):
    # Create a dictionary to store the data
    data = defaultdict(lambda: defaultdict(str))
    
    # Column headers
    headers = ['Natural Language', 'Ground Truth']
    model_approaches = set()

    # Process the results
    for result in results:
        nl = result['Natural Language']
        gt = result['Ground Truth']
        model_approach = f"{result['Model']}_{result['Approach']}"
        extract_ltl = result['LLM Response']

        data[nl]['Ground Truth'] = gt
        data[nl][model_approach] = extract_ltl
        model_approaches.add(model_approach)

    # Sort the model_approaches and add them to headers
    headers.extend(sorted(model_approaches))

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for nl, row_data in data.items():
            row = {'Natural Language': nl}
            row.update(row_data)
            writer.writerow(row)

def save_raw_responses_to_csv(filename, data):
    """Save raw responses to a CSV file."""
    df = pd.DataFrame(data)
    # Fix: Check if file exists before writing header
    file_exists = os.path.exists(filename)
    df.to_csv(filename, index=False, mode='a', header=not file_exists)

def read_saved_responses_from_csv(filename):
    """Read previously saved raw responses from a CSV file."""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        print(f"{filename} does not exist. Run the initial response generation first.")
        return pd.DataFrame()

def generate_raw_responses(nl2ltl_dataset, models, learning_approaches, raw_responses_file):
    """Generate and save raw responses from different models and approaches."""
    if os.path.exists(raw_responses_file):
        print(f"{raw_responses_file} already exists. Skipping generation.")
        return
    
    raw_responses = []
    total_combinations = len(models) * len(learning_approaches) * len(nl2ltl_dataset)
    current_count = 0
    
    for model in models:
        for approach in learning_approaches:
            print(f"Generating responses for {model} with {approach} approach...")
            for _, row in nl2ltl_dataset.iterrows():
                current_count += 1
                print(f"Processing {current_count}/{total_combinations}")
                
                response = process_single_response(model, approach, row)
                raw_responses.append(response)
                
                # Save periodically to avoid losing data
                if len(raw_responses) % 50 == 0:
                    save_raw_responses_to_csv(raw_responses_file, raw_responses[-50:])
    
    # Save any remaining responses
    if raw_responses:
        save_raw_responses_to_csv(raw_responses_file, raw_responses)
    print(f"Raw responses saved to {raw_responses_file}")

def process_single_response(model, approach, row):
    """Process a single response for given model, approach and input row."""
    natural_language = row['Natural Language']
    ground_truth = row['Ground Truth']
    atomic_propositions = row['Atomic Proposition']
    
    # Generate initial prompt
    try:
        initial_prompt = generate_prompt(natural_language, atomic_propositions, approach)
        initial_response = send_request(initial_prompt, model)
    except Exception as e:
        print(f"Error for initial response - {natural_language}: {e}")
        initial_response = "Error"
    
    # For approaches other than self-refine, return immediately
    if approach != "zero_shot_self_refine":
        return {
            'Model': model,
            'Approach': approach,
            'Natural Language': natural_language,
            'Atomic Proposition': atomic_propositions,
            'Ground Truth': ground_truth,
            'Initial Response': initial_response,
            'Refined Response': None
        }
    
    # For self-refine approach, generate refined response
    try:
        refined_prompt = generate_prompt(natural_language, atomic_propositions, approach, initial_response)
        refined_response = send_request(refined_prompt, model)
    except Exception as e:
        print(f"Error during refinement for {natural_language}: {e}")
        refined_response = "Error"
    
    return {
        'Model': model,
        'Approach': approach,
        'Natural Language': natural_language,
        'Atomic Proposition': atomic_propositions,
        'Ground Truth': ground_truth,
        'Initial Response': initial_response,
        'Refined Response': refined_response
    }


def extract_ltl_formula(response_text):
    """
    Extract LTL formula from model response, handling various code block formats.
    """
    # Handle NaN, None, or non-string values
    if response_text is None or (isinstance(response_text, float) and pd.isna(response_text)):
        return "No LTL formula extracted"
    
    # Convert to string if it's not already
    if not isinstance(response_text, str):
        response_text = str(response_text)
    
    if not response_text or response_text.strip() == "Error":
        return "No LTL formula extracted"
    
    # Clean the response
    response = response_text.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "LTL Formula:",
        "LTL:",
        "Formula:",
        "The LTL formula is:",
        "Result:",
        "Output:"
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response.replace(prefix, "").strip()
    
    # Remove code block markers and language identifiers
    # Handle various formats: ```, ```plaintext, ```ltl, etc.
    code_block_patterns = [
        r'^```[\w]*\n?',  # Opening code blocks (```plaintext, ```ltl, etc.)
        r'\n?```$',       # Closing code blocks
        r'^```[\w]*$',    # Standalone code block markers
        r'```$',          # Closing code blocks at end
        r'^```',          # Opening code blocks at start
        r'```'            # Any remaining code block markers
    ]
    
    for pattern in code_block_patterns:
        response = re.sub(pattern, '', response, flags=re.MULTILINE)
    
    # Remove any remaining backticks
    response = response.replace('`', '')
    
    # Clean up whitespace and newlines
    response = response.strip()
    
    # Handle multiline responses - take the first non-empty line that looks like LTL
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    if not lines:
        return "No LTL formula extracted"
    
    # If there's only one line, return it
    if len(lines) == 1:
        return lines[0]
    
    # If multiple lines, try to find the one that looks most like an LTL formula
    # LTL formulas typically contain operators like H, Y, O, S, G, F, U, &, |, !, ->, <->
    ltl_operators = ['H', 'Y', 'O', 'S', 'G', 'F', 'U', '&', '|', '!', '->', '<->', 'true', 'false']
    
    for line in lines:
        # Check if line contains LTL operators
        if any(op in line for op in ltl_operators):
            # Additional check: avoid lines that seem like explanations
            if not any(word in line.lower() for word in ['the', 'this', 'formula', 'means', 'represents', 'where']):
                return line
    
    # If no line clearly looks like LTL, return the first non-empty line
    return lines[0]

def process_responses(raw_responses_df, detailed_results_file, summary_results_file):
    """Process raw responses and generate detailed and summary results."""
    if raw_responses_df.empty:
        return
    
    nl2ltl_detailed_results = []
    nl2ltl_summary_results = {}
    
    for _, row in raw_responses_df.iterrows():
        detailed_result, summary_result = process_single_result(row)
        nl2ltl_detailed_results.append(detailed_result)
        
        key = f"{row['Model']}_{row['Approach']}"
        if key not in nl2ltl_summary_results:
            nl2ltl_summary_results[key] = []
        nl2ltl_summary_results[key].append(summary_result)
    
    write_detailed_results(detailed_results_file, nl2ltl_detailed_results)
    format_nl2ltl_results_for_csv(nl2ltl_detailed_results, summary_results_file)
    print(f"Detailed results saved to {detailed_results_file}")
    print(f"Summary results saved to {summary_results_file}")

def process_single_result(row):
    """Process a single result row and return detailed and summary results."""
    
    # Helper function to safely get string value from potentially NaN column
    def safe_get_string(value):
        if pd.isna(value):
            return ""
        return str(value)
    
    # Handle the response based on approach
    initial_response = safe_get_string(row['Initial Response'])
    refined_response = safe_get_string(row['Refined Response']) if 'Refined Response' in row else ""
    
    if row['Approach'] == "zero_shot_self_refine" and refined_response:
        generated_response = f"Initial: {initial_response}\nRefined: {refined_response}"
        extract_ltl = extract_ltl_formula(refined_response)
    else:
        generated_response = initial_response
        extract_ltl = extract_ltl_formula(initial_response)
    
    # Ensure we have a valid LTL extraction
    if not extract_ltl or extract_ltl.strip() == "":
        extract_ltl = "No LTL formula extracted"
    
    detailed_result = {
        'Model': row['Model'],
        'Approach': row['Approach'],
        'Natural Language': row['Natural Language'],
        'Atomic Proposition': row['Atomic Proposition'],
        "Ground Truth": row['Ground Truth'],
        'Generated Response': generated_response,
        "LLM Response": extract_ltl
    }
    
    summary_result = {
        'Natural Language': row['Natural Language'],
        'Atomic Proposition': row['Atomic Proposition'],
        "Ground Truth": row['Ground Truth'],
        'Extracted LTL': extract_ltl
    }
    
    return detailed_result, summary_result
# def main():
#     """Main function to coordinate the LTL processing pipeline."""
#     pd.options.display.float_format = "{:,.2f}".format
    
#     # Configuration
#     nl2ltl_dataset = pd.read_csv('dataset/past_ltl.csv')
#     learning_approaches = ["zero_shot", "zero_shot_self_refine", "few_shot"]
#     models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-sonnet", "gemini"]
    
#     file_paths = {
#         "detailed_results_file": "output/nl2pastltl/nl2ltl_detailed_results.csv",
#         "summary_results_file": "output/nl2pastltl/nl2ltl_summary_results.csv",
#         "output_dir": "output/nl2pastltl",
#         "raw_responses_file": 'output/nl2pastltl/nl2ltl_raw_responses.csv',
#     }
    
#     # Create output directory
#     Path(file_paths["output_dir"]).mkdir(parents=True, exist_ok=True)
    
#     # Phase 1: Generate raw responses
#     print("Phase 1: Generating raw responses...")
#     # generate_raw_responses(nl2ltl_dataset, models, learning_approaches, file_paths["raw_responses_file"])
    
#     # Phase 2: Process responses
#     print("Phase 2: Processing responses...")
#     raw_responses = read_saved_responses_from_csv(file_paths["raw_responses_file"])
#     process_responses(raw_responses, file_paths["detailed_results_file"], file_paths["summary_results_file"])
    
#     print("Processing complete!")

# if __name__ == "__main__":
#     main()


def main():
    # start_time = time.time()
    # processing...

    """Main function to coordinate the LTL processing pipeline."""
    # args = parse_args()
    
    pd.options.display.float_format = "{:,.2f}".format
    nl2ltl_dataset = pd.read_csv('dataset/past_ltl.csv')
    # Load dataset
    # nl2ltl_dataset = pd.read_csv(args.dataset)

    # Set experiment output directory
    output_dir = Path("/Users/priscilladanso/Library/Mobile Documents/com~apple~CloudDocs/Documents/StonyBrook/TowardsDissertation/PYTHON4LTL/output/littletrickylogic_nl2pastltl")
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    file_paths = {
        "raw_responses_file": output_dir / "nl2ltl_raw_responses.csv",
        "detailed_results_file": output_dir / "nl2ltl_detailed_results.csv",
        "summary_results_file": output_dir / "nl2ltl_summary_results.csv",
    }

    # learning_approaches = ["zero_shot", "zero_shot_self_refine", 
    learning_approaches = ["few_shot"]
    # models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-flash"]
    models = ["claude-3.5-sonnet"]
    # Phase 1: Generate raw responses (only if not already done)
    print("Phase 1: Checking for existing raw responses...")
    if os.path.exists(file_paths["raw_responses_file"]):
        print(f"Raw responses file {file_paths['raw_responses_file']} already exists.")
        print("Skipping raw response generation.")
    else:
        print("Generating raw responses...")
        generate_raw_responses(nl2ltl_dataset, models, learning_approaches, file_paths["raw_responses_file"])


    # Phase 2: Process raw responses into detailed/summary (only if not already done)
    print("\nPhase 2: Checking for existing processed results...")
    if (os.path.exists(file_paths["detailed_results_file"]) and 
        os.path.exists(file_paths["summary_results_file"])):
        print("Processed results already exist. Skipping processing.")
        print("If you want to reprocess, delete the existing result files.")
    else:
        print("Processing raw responses into detailed and summary files...")
        raw_responses = read_saved_responses_from_csv(file_paths["raw_responses_file"])
        if not raw_responses.empty:
            process_responses(raw_responses, file_paths["detailed_results_file"], file_paths["summary_results_file"])
        else:
            print("No raw responses found to process.")

    print("\nProcessing complete!")
    # estimate_total_cost()
    # duration = time.time() - start_time
    # print(f"Processed in {duration:.2f}s")
if __name__ == "__main__":
    main()
