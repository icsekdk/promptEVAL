#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path
import sys

# Get paths relative to current script
script_dir = Path(__file__).parent
project_root = script_dir.parent  # Assumes script is in experiments/
print(f"Project root directory: {project_root}")
experiment_name = sys.argv[1] if len(sys.argv) > 1 else "nl2futureltl"

experiment_dir = project_root / "output" / experiment_name

INPUT_CSV = experiment_dir / "nl2ltl_summary_results.csv"
OUTPUT_CSV = experiment_dir / "semantic_equiv_entail_results.csv"

# Configuration - SET THESE PATHS
# Convert string path to Path object
LTL_DIR = Path("/Users/priscilladanso/Documents/GitHub/LTL")

# NUSMV_PATH = "/usr/local/NuSMV-2.6.0/bin/nusmv"
NUSMV_PATH = "/usr/local/bin/nusmv"

def run_ocaml(formula1: str, formula2: str = "", cmd_type: str = "equiv") -> str:
    """Run the OCaml binary with the given input"""
    input_str = f"{cmd_type}\n{formula1}"
    if formula2:
        input_str += f"\n{formula2}\n"
    else:
        input_str += "\n"
    
    try:
        result = subprocess.run(
            ["dune", "exec", "--root", str(LTL_DIR), "./corrected_version/ltlutils/bin/main.exe"],
            input=input_str.encode(),
            capture_output=True,
            check=True,
            cwd=str(LTL_DIR)
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Error processing: {input_str}")
        print(f"Error: {e.stderr.decode()}")
        return "ERROR"

def setup_environment():
    """Configure the environment and compile the project"""
    # Update NuSMV path in compile script
    compile_script = LTL_DIR / "corrected_version" / "ltlutils" / "compile-project"
    
    # Check if the compile script exists
    if not compile_script.exists():
        print(f"Warning: Compile script not found at {compile_script}")
        return
    
    try:
        with open(compile_script, 'r+') as f:
            content = f.read()
            f.seek(0)
            f.write(content.replace("NUSMV_HOME=/path/to/NuSMV", f"NUSMV_HOME={NUSMV_PATH}"))
            f.truncate()
        compile_script.chmod(0o755)
        
        # Compile the project
        subprocess.run(["./compile-project"], 
                      cwd=LTL_DIR / "corrected_version" / "ltlutils",
                      check=True)
    except Exception as e:
        print(f"Error setting up environment: {e}")
        print("Continuing without compilation...")

def process_csv():
    """Process the input CSV and generate results"""
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        return
    
    with open(INPUT_CSV) as f_in, open(OUTPUT_CSV, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Read header and add result columns
        header = next(reader)
        new_header = header.copy()
        
        # Add equivalence columns
        for i in range(1, len(header)):
            for j in range(i+1, len(header)):
                new_header.append(f"Eq_{header[i]}_vs_{header[j]}")
        
        # Add entailment columns
        for i in range(1, len(header)):
            for j in range(1, len(header)):
                if i != j:
                    new_header.append(f"En_{header[i]}_to_{header[j]}")
        
        # Add syntax check columns
        for i in range(1, len(header)):
            new_header.append(f"Syntax_{header[i]}")
        
        writer.writerow(new_header)
        
        # Process each row
        for row_num, row in enumerate(reader, 1):
            print(f"Processing row {row_num}...")
            new_row = row.copy()
            
            # Equivalence checks
            for i in range(1, len(header)):
                for j in range(i+1, len(header)):
                    res = run_ocaml(row[i], row[j], "equiv")
                    new_row.append(res)
            
            # Entailment checks
            for i in range(1, len(header)):
                for j in range(1, len(header)):
                    if i != j:
                        res = run_ocaml(row[i], row[j], "check_entailment")
                        new_row.append(res)
            
            # Syntax checks
            for i in range(1, len(header)):
                res = run_ocaml(row[i], "", "print_formula")
                new_row.append(res)
            
            writer.writerow(new_row)

def main():
    print("Setting up environment...")
    setup_environment()
    
    print(f"Processing {INPUT_CSV}...")
    process_csv()
    
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()