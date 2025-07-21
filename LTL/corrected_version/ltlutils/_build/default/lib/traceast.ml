open Utilfuncs 
open Ast 

type propvar = string 

type trace_state = (propvar * bool) list 

type trace = trace_state list 

let get_variables_from_state s =
  let var_list = List.map (fun x -> fst x) s 
in StringSet.of_list var_list 


let rec check_consistency (variable_set : StringSet.t) (t: trace) : bool = 
  match t with 
  | [] -> true 
  | x :: xs -> let vset = get_variables_from_state x in 
  if StringSet.equal vset variable_set then check_consistency variable_set xs 
  else false 


let is_trace_consistent (t : trace) : bool = 
  match t with 
  | [] -> true 
  | x :: xs -> let var_set = get_variables_from_state x in    
               check_consistency var_set xs  

let rec get_truth_value_from_state (vname : string) (cstate : trace_state) : bool option =
  match cstate with 
  | [] -> None 
  | (x, y) :: xs -> if x = vname then Some y else get_truth_value_from_state vname xs 

let get_variables_from_consistent_trace (t : trace)  = 
  match t with 
  | [] -> StringSet.of_list [] 
  | x :: _ -> get_variables_from_state x    

let print_state (cstate : trace_state) = 
  Printf.eprintf "[";
  List.iter (fun x -> let (vname, v) = x in if v = true then Printf.eprintf "%s |=> true," vname else Printf.eprintf "%s |=> false," vname) cstate ;
  Printf.eprintf "]"  

let print_trace (t : trace) = 
  (* Printf.printf "[" ;  *)
  List.iter (fun x -> Printf.eprintf "["; print_state x ; Printf.eprintf "];") t ; 
  Printf.eprintf "...\n"


  let isVariableTrueInPositionI (var : string) (t : trace) (i : int) : bool option = 
    let state_i_opt = List.nth_opt t i in 
    match state_i_opt with 
    | None -> None 
    | Some(state_i) -> get_truth_value_from_state var state_i  
  
  let combine_results (op : binlogicalop) (r1 : bool option) (r2 : bool option) : bool option = 
    match op with 
    | AND -> (match (r1, r2 ) with 
      | None, _ -> None 
      | _, None -> None 
      | Some(b1), Some(b2) -> let x = (b1 && b2) in Some(x)
      )
    | OR -> (match (r1, r2 ) with 
      | None, _ -> None 
      | _, None -> None 
      | Some(b1), Some(b2) -> let x = (b1 || b2) in Some(x)
      )
      | EQUIV -> (match (r1, r2 ) with 
        | None, _ -> None 
        | _, None -> None 
        | Some(b1), Some(b2) -> if b1 = b2 then Some(true) else Some(false)
      )
      | IMPLY -> (match (r1, r2 ) with 
        | None, _ -> None 
        | _, None -> None 
        | Some(b1), Some(b2) -> let x = ((not b1) || b2)  in Some(x)
      )
  
  
  let rec evaluate_formula_in_trace_position (f: ltlformula) (t:trace) (i : int) (max_trace_len : int ) : bool option = 
    if i >= max_trace_len then None 
    else 
    match f with 
    | Identifier(s) -> isVariableTrueInPositionI s t i 
    | LTrue -> Some(true)
    | LFalse -> Some(false) 
    | Not(fx) -> 
      (let r1 = (evaluate_formula_in_trace_position fx t i max_trace_len) in 
        match r1 with
        | None -> None 
        | Some true -> Some(false)
        | Some false -> Some (true)
      )  
    | BinPOP(op, f1, f2) -> 
      let r1 = evaluate_formula_in_trace_position f1 t i max_trace_len in 
      let r2 = evaluate_formula_in_trace_position f2 t i max_trace_len in 
      combine_results op r1 r2
    | UnTOP(op, f1) -> 
      (
        match op with 
        | YESTERDAY -> if i > 0 then evaluate_formula_in_trace_position f1 t (i-1) max_trace_len else Some(false) 
        | TOMORROW -> if i>= (max_trace_len-1) then None else evaluate_formula_in_trace_position f1 t (i+1) max_trace_len 
        | ONCE -> (let r1 = evaluate_formula_in_trace_position f1 t i max_trace_len in 
                    match r1 with 
                    | Some(true) -> Some(true)
                    | _ -> (if i = 0 then r1 
                            else evaluate_formula_in_trace_position f t (i-1) max_trace_len)
                  ) 
        | HISTORICALLY -> 
          (let r1 = evaluate_formula_in_trace_position f1 t i max_trace_len in 
          match r1 with 
          | Some(false) -> Some(false)
          | None -> None 
          | _ -> if i = 0 then r1 
                 else let r2 = evaluate_formula_in_trace_position f t (i-1) max_trace_len in 
                      combine_results AND r1 r2
          ) 
        | GLOBALLY -> 
          (let r1 = evaluate_formula_in_trace_position f1 t i max_trace_len in 
          match r1 with 
          | Some(false) -> Some(false)
          | None -> None 
          | _ -> if i = max_trace_len - 1 then r1 
                 else let r2 = evaluate_formula_in_trace_position f t (i+1) max_trace_len in 
                 combine_results AND r1 r2)
        | EVENTUALLY -> (let r1 = evaluate_formula_in_trace_position f1 t i max_trace_len in
                          match r1 with 
                          | Some(true) -> Some(true) 
                          | _ -> if i = max_trace_len - 1 then r1 
                                else evaluate_formula_in_trace_position f t (i+1) max_trace_len
                        )
      )
    | BinTOP(op, f1, f2) -> (
      match op with 
      | SINCE -> handle_since f1 f2 t i max_trace_len 
      | UNTIL -> handle_until f1 f2 t i max_trace_len
    )
and 

handle_since f1 f2 t i max_trace_len = 
      let r1 = check_formula_truth_in_bound_backward 0 i f2 t max_trace_len in 
      match r1 with 
      | Some(x) -> if x = i then Some(true) else let v = formula_true_in_bound f1 t (x+1) i (x+1) max_trace_len in Some(v)
      | _ -> Some(false)  

and 

handle_until f1 f2 t i max_trace_len = 
      let r1 = check_formula_truth_in_bound_forward i (max_trace_len-1) f2 t max_trace_len in 
      match r1 with 
      | Some(x) -> if x = i then Some(true) else let v = formula_true_in_bound f1 t i (x-1) i max_trace_len in Some(v)
      | _ -> Some(false)

and 

formula_true_in_bound f t left right i max_trace_len = 
if left > right then true else 
if i >= max_trace_len || i > right || i < left then false 
else 
  let r1 = evaluate_formula_in_trace_position f t i max_trace_len in 
  match r1 with 
  | Some(true) -> if i = right then true else formula_true_in_bound f t left right (i+1) max_trace_len 
    | _ -> false  

and 

check_formula_truth_in_bound_backward start cur f t max_trace_len =
if cur >= max_trace_len then None 
else  
  let r1 = evaluate_formula_in_trace_position f t cur max_trace_len in
  match r1 with 
  | Some(true) -> Some(cur) 
  | _ -> if cur = start then None else check_formula_truth_in_bound_backward start (cur-1) f t max_trace_len  
  
and 

check_formula_truth_in_bound_forward cur end_limit f t max_trace_len = 
  if cur >= max_trace_len then None 
  else 
    let r1 = evaluate_formula_in_trace_position f t cur max_trace_len in 
    match r1 with 
    | Some(true) -> Some(cur) 
    | _ -> if cur = end_limit then None else check_formula_truth_in_bound_forward (cur + 1) end_limit f t max_trace_len
    
  let evaluate_formula_in_trace (f : ltlformula) (t : trace) : bool option = 
    let len = List.length t in 
    (evaluate_formula_in_trace_position f t 0 len) 