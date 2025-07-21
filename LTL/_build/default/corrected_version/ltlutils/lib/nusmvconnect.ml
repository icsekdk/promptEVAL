open Ast
open Utilfuncs
(* open Sys  *)

(* Fix the following executable path according to your system to point to the NuSMV binary ...  *)

let rec get_var_decls (var_list : string list) : string =
  match var_list with
  | [] -> "\n"
  | x :: xs -> String.cat (String.cat x " : boolean;\n") (get_var_decls xs)

let rec get_variable_assignments (var_list : string list) : string =
  match var_list with
  | [] -> "\n"
  | x :: xs ->
      let initx =
        String.cat (String.cat "init(" x)
          (Ast.convert_string_list_to_one_string [ ") := {TRUE, FALSE};\n" ])
      in
      let nextx =
        String.cat (String.cat "next(" x)
          (Ast.convert_string_list_to_one_string [ ") := {TRUE, FALSE};\n" ])
      in
      String.cat initx
        (Ast.convert_string_list_to_one_string
           [ nextx; get_variable_assignments xs ])

let generate_equivalence_query both_sides f1 f2 =
  let preamble = "MODULE main\nVAR\n" in
  let variables =
    StringSet.elements
      (StringSet.union
         (get_prop_vars_from_formula f1)
         (get_prop_vars_from_formula f2))
  in
  let variable_decl_list = get_var_decls variables in
  let assign_list = get_variable_assignments variables in
  let ltlspec = "LTLSPEC " in
  let formula1 = formulaToString f1 in
  let formula2 = formulaToString f2 in
  if both_sides then 
  String.cat preamble
    (Ast.convert_string_list_to_one_string
       [
         variable_decl_list;
         "ASSIGN\n";
         assign_list;
         ltlspec;
         (* "G("; *)
         "("; 
         formula1;
         " <-> ";
         formula2;
         ")\n";
       ])
else
  String.cat preamble
    (Ast.convert_string_list_to_one_string
       [
         variable_decl_list;
         "ASSIGN\n";
         assign_list;
         ltlspec;
         (* "G("; *)
         "(";
         formula1;
         " -> ";
         formula2;
         ")\n";
       ])

let generate_negative_trace_query f =
  let preamble = "MODULE main\nVAR\n" in
  let variables = StringSet.elements (get_prop_vars_from_formula f) in
  let variable_decl_list = get_var_decls variables in
  let assign_list = get_variable_assignments variables in
  let ltlspec = "LTLSPEC " in
  let formula1 = formulaToString f in
  Ast.convert_string_list_to_one_string
    [
      preamble;
      variable_decl_list;
      "ASSIGN\n";
      assign_list;
      ltlspec;
      formula1;
      "\n";
    ]

let generate_positive_trace_query f =
  let preamble = "MODULE main\nVAR\n" in
  let variables = StringSet.elements (get_prop_vars_from_formula f) in
  let variable_decl_list = get_var_decls variables in
  let assign_list = get_variable_assignments variables in
  let ltlspec = "LTLSPEC " in
  let formula1 = formulaToString f in
  Ast.convert_string_list_to_one_string
    [
      preamble;
      variable_decl_list;
      "ASSIGN\n";
      assign_list;
      ltlspec;
      "!(";
      formula1;
      ")\n";
    ]

(* ./NuSMV  -dcx test2.smv | grep "^--" | grep -oE '[^ ]+$' *)
