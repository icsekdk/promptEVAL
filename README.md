# A Multi-dimensional Evaluation of LLMs in Translating Natural Language Requirements to Linear Temporal Logic (LTL) Formulas

This repository contains the code and data for our ICSE 2026 paper:  
**"A Multi-dimensional Evaluation of LLMs in Translating Natural Language Requirements to Linear Temporal Logic (LTL) Formulas"**.

We evaluate multiple Large Language Models (LLMs) across three prompting strategies and three translation tasks: NL2PL, NL2FutureLTL, and NL2PastLTL. We introduce a rigorous benchmark to analyze syntactic and semantic correctness of generated formulas.


## Explanation of main Repository Structure
* LTL - This directory contains the OCaml used for model checking.
* dashboard - Dashboard hosted on Streamlit acting as a playground to show the comparative analysis of our experimental results
* data - Further split into input_data (contains dataset used to run each experiment) and output_data (storage location of all experiments)
- experiment - Each benchmark code is found in this directory
- handle_keys.py - API configuration set
- parse_prompt.py - Client configuration to handle communuication between the prompt and the API
- prompts - Categorized according the three prompting strategies; Detailed, Minimal, Python with carefully-crafted prompt aligned with the benchmark 
- requirements.txt - Dependencies
- scripts - All required scripts needed for execution can be found here.

## Installation

```bash
git clone https://github.com/yourusername/NL2LTL-Evaluation.git
cd NL2LTL-Evaluation

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## Run Specific Experiment Type
```bash 
python experiment/nl2ltl.py --dataset data/input_data/nl2ltl_finalized.csv --experiment_type minimal --experiment_name nl2futureltl_littletrickylogic
```
OR 
```bash
python experiment/nl2ltl.py --dataset data/input_data/nl2ltl_finalized.csv --experiment_type detailed --experiment_name nl2futureltl_textbook
```
OR 
```bash
python experiment/nl2ltl.py --dataset data/input_data/nl2ltl_finalized_ast.csv --experiment_type python --experiment_name nl2futureltl_littletrickylogic
```


## Supported Learning approaches by experiment type
LEARNING APPROACHES - Same for all 

* Zero-Shot
* Zero-Shot Self-Refine
*  Few Shot

MODEL SETS

For Minimal and Detailed
  * GPT-3.5-Turbo
  * GPT-4o-Mini 
  * GPT-4o
  * Claude-3.5-Sonnet
  * Gemini-1.5-Pro

For Python
  * Claude-3.5-Sonnet
  * Gemini-1.5-Pro
  * Gemini-1.5-Flash
  * Gemini-2.5-Flash 
 

## Evaluation Metrics
* Syntactic accuracy: Exact string match and normalized equivalence
* Semantic consistency: Trace-based evaluation (using NuSMV and structured traces)
* F1, Jaccard, Levenshtein: For structural closeness


