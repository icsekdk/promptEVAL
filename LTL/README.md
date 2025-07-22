# LTL Utils - Linear Temporal Logic Utility Tool

This OCaml project provides a comprehensive toolkit for working with Linear Temporal Logic (LTL) formulas, including syntax checking, semantic analysis, equivalence checking, and trace generation/validation.

## Dependencies

### Required Software
* **OCaml Package Manager (opam)**: Follow installation instructions at [https://opam.ocaml.org/doc/Install.html]
* **NuSMV Model Checker**: Download from [https://nusmv.fbk.eu/downloads.html]

### OCaml Dependencies
```bash
opam init
opam install dune
opam install merlin
opam user-setup install
opam install menhir
opam install ocaml-lsp-server
```

## Setup and Compilation

### NuSMV Configuration
1. Edit the BASH script `compile-project` in `LTL/corrected_version/ltlutils/`
2. Replace the NuSMV path with your system's full path to the NuSMV `bin` folder
3. Make the script executable:
   ```bash
   chmod a+x compile-project
   ```

### Build Process
Navigate to `LTL/corrected_version/ltlutils/` and run:
```bash
./compile-project
dune build
```

## Execution

### Basic Usage
```bash
dune exec ./bin/main.exe < input_file
```

Replace `input_file` with your command file (e.g., `inp_file`, `testingLTL`, etc.)

### Input Format Rules
- **Each command and its arguments must be on separate lines**
- **No empty lines between arguments**
- **Only the first command in a file will be executed**
- **Additional commands will be ignored**

## Supported Commands

### 1. Formula Syntax and Printing
#### `print_formula`
Converts and prints LTL formulas in NuSMV format.

**Format:**
```
print_formula
<formula>
```

**Examples:**
```
print_formula
x1 -> X(X(X(x1)))

print_formula
G(X(x1) <-> (x2 <-> X(!x2)))

print_formula
x & y -> z
```

#### `generate_random_formula`
Generates a well-formed random LTL formula.

**Format:**
```
generate_random_formula
```

### 2. Semantic Equivalence Checking
#### `equiv`
Checks logical equivalence between two LTL formulas.

**Format:**
```
equiv
<formula_1>
<formula_2>
```

**Examples:**
```
equiv
!F(!a)
G(a)

equiv
F(a)
!G(!a)

equiv
F(a)
!G(a)
```

### 3. Temporal Operator Classification
#### `check_future`
Verifies if a formula contains only propositional and future temporal operators (no past operators).

**Format:**
```
check_future
<formula>
```

**Example:**
```
check_future
G(a -> F(b))
```

#### `check_past`
Verifies if a formula contains only propositional and past temporal operators (no future operators).

**Format:**
```
check_past
<formula>
```

**Example:**
```
check_past
H(a -> O(b))
```

### 4. Entailment Checking
#### `check_entailment`
Checks if the first formula logically entails the second formula.

**Format:**
```
check_entailment
<formula_1>
<formula_2>
```

**Examples:**
```
check_entailment
G(a)
!F(a)

check_entailment
!F(a)
G(a)
```

### 5. Trace Generation
#### `positive_trace_gen`
Generates a trace that **satisfies** the given formula. Output stored in `nusmv_executable_path/trace.out`.

**Format:**
```
positive_trace_gen
<formula>
```

**Example:**
```
positive_trace_gen
F(a)
```

#### `negative_trace_gen`
Generates a trace that **falsifies** the given formula. Output stored in `nusmv_executable_path/trace.out`.

**Format:**
```
negative_trace_gen
<formula>
```

**Example:**
```
negative_trace_gen
G(X(x1) -> (x2 & !X(x2)))
```

### 6. Trace Satisfaction Checking
#### `check_trace_satisfaction`
Verifies whether a given trace satisfies an LTL formula.

**Format:**
```
check_trace_satisfaction
<formula>
<trace>
```

**Trace Format:** `[var1=value1, var2=value2, ...]; [var1=value1, var2=value2, ...]; ...`

**Examples:**
```
check_trace_satisfaction
G(p -> (a S b))
[a=false, b=true, p=false]; [a=true, b=false, p=false]; [a=true, b=false, p=true]

check_trace_satisfaction
a S b
[a=false, b=true]

check_trace_satisfaction
G(a -> Y(b))
[a=false,b=true]; [a=false,b=false]; [a=false,b=false]; [a=false,b=false]; [a=true,b=false]

check_trace_satisfaction
G(a -> b)
[a=true,b=true]; [a=false,b=false]; [a=false,b=false]; [a=true,b=false]; [a=false,b=true]

check_trace_satisfaction
X(Y a)
[a=true]; [a=false]

check_trace_satisfaction
a U b
[a=false,b=true]

check_trace_satisfaction
G(a) & G(c)
[a=true,c=true]; [a=true,c=true]
```

## LTL Operators Reference

### Temporal Operators
- **X** (Next): Formula holds in the next state
- **F** (Finally/Eventually): Formula will hold at some future state
- **G** (Globally/Always): Formula holds in all future states
- **U** (Until): First formula holds until second formula becomes true
- **S** (Since): Past version of Until
- **Y** (Yesterday): Formula held in the previous state
- **O** (Once): Formula held at some past state
- **H** (Historically): Formula held in all past states

### Propositional Operators
- **!** (Not): Negation
- **&** (And): Conjunction
- **|** (Or): Disjunction
- **->** (Implies): Implication
- **<->** (If and only if): Biconditional

## Creating Custom Input Files

Create a new file (e.g., `my_commands.txt`) with your desired commands:

```
equiv
F(a)
!G(!a)
```

Then execute:
```bash
dune exec ./bin/main.exe < my_commands.txt
```

## Output Files

- **Trace files**: Generated traces are stored in `nusmv_executable_path/trace.out`
- **Console output**: Results and analysis are printed to standard output

## Notes

- The tool processes only the first command in each input file
- Ensure NuSMV is properly installed and the path is correctly configured
- All formulas should follow standard LTL syntax
- Trace format uses semicolon-separated states with comma-separated variable assignments
