open Ast
(* open Traceast *)

module StringSet = Set.Make (struct
  type t = string

  let compare = Stdlib.compare
end)

let rec collect_props_from_formula (f : ltlformula) : string list =
  match f with
  | Identifier s -> [ s ]
  | LTrue -> []
  | LFalse -> []
  | BinPOP (_, f1, f2) ->
      collect_props_from_formula f1 @ collect_props_from_formula f2
  | BinTOP (_, f1, f2) ->
      collect_props_from_formula f1 @ collect_props_from_formula f2
  | Not f1 -> collect_props_from_formula f1
  | UnTOP (_, f1) -> collect_props_from_formula f1

let get_prop_vars_from_formula (f : ltlformula) : StringSet.t =
  StringSet.of_list (collect_props_from_formula f)

let rec is_temporal_formula (f : ltlformula) : bool =
  match f with
  | Identifier _ -> false
  | LTrue -> false
  | LFalse -> false
  | BinPOP (_, f1, f2) -> is_temporal_formula f1 || is_temporal_formula f2
  | BinTOP (_, _, _) -> true
  | Not f1 -> is_temporal_formula f1
  | UnTOP (_, _) -> true

let is_future_bin_op (op : bintemporalop) : bool = op = UNTIL
let is_past_bin_op (op : bintemporalop) : bool = op = SINCE

let is_past_un_op (op : unarytemporalop) : bool =
  op = YESTERDAY || op = ONCE || op = HISTORICALLY

let is_future_un_op (op : unarytemporalop) : bool =
  op = TOMORROW || op = EVENTUALLY || op = GLOBALLY

let rec is_past_temporal_formula (f : ltlformula) : bool =
  match f with
  | Identifier _ -> true
  | LTrue -> true
  | LFalse -> true
  | BinPOP (_, f1, f2) ->
      is_past_temporal_formula f1 && is_past_temporal_formula f2
  | BinTOP (op, f1, f2) ->
      is_past_bin_op op
      && is_past_temporal_formula f1
      && is_past_temporal_formula f2
  | Not f1 -> is_past_temporal_formula f1
  | UnTOP (op, f1) -> is_past_un_op op && is_past_temporal_formula f1

let rec is_future_temporal_formula (f : ltlformula) : bool =
  match f with
  | Identifier _ -> true
  | LTrue -> true
  | LFalse -> true
  | BinPOP (_, f1, f2) ->
      is_future_temporal_formula f1 && is_future_temporal_formula f2
  | BinTOP (op, f1, f2) ->
      is_future_bin_op op
      && is_future_temporal_formula f1
      && is_future_temporal_formula f2
  | Not f1 -> is_future_temporal_formula f1
  | UnTOP (op, f1) -> is_future_un_op op && is_future_temporal_formula f1



let rand_chr () = (Char.chr (97 + (let () = Random.self_init () in Random.int 26)))
let rec rand_voy () = let got = (rand_chr ()) in match got with | 'a' | 'e' | 'i' | 'o' | 'u' | 'y' ->  got | _ -> rand_voy ()  

let rec rand_con () = let got = (rand_chr ()) in match got with | 'a' | 'e' | 'i' | 'o' | 'u' | 'y' ->  rand_con () | _ -> got  

let rec rand_convoy acc syll_number () = match syll_number with | 0 -> acc; | _ -> rand_convoy (acc ^ (Char.escaped (rand_con ())) ^ (Char.escaped(rand_voy()))) (syll_number - 1) ()

let rand_word () = (rand_convoy "" (3 + (let () = Random.self_init () in Random.int 3)) ())


let rec gen_ltl level = 
if level = 0 then 
  let () = Random.self_init () in 
  let choice = Random.int 4 in 
match choice with 
| 0  | 1 -> let vname = rand_word() in Identifier(vname) 
| 2 -> LTrue
| _ -> LFalse 
else 
  let () = Random.self_init () in 
  let choice = Random.int 13 in 
  match choice with 
  | 0 -> let f1 = gen_ltl (level - 1) in let f2 = gen_ltl (level-1) in BinTOP(SINCE, f1, f2)
  | 1 -> let f1 = gen_ltl (level - 1) in let f2 = gen_ltl (level-1) in BinTOP(UNTIL, f1, f2)
  | 2 -> let f1 = gen_ltl (level - 1) in let f2 = gen_ltl (level-1) in BinPOP(AND, f1, f2)
  | 3 -> let f1 = gen_ltl (level - 1) in let f2 = gen_ltl (level-1) in BinPOP(OR, f1, f2)
  | 4 -> let f1 = gen_ltl (level - 1) in let f2 = gen_ltl (level-1) in BinPOP(IMPLY, f1, f2)
  | 5 -> let f1 = gen_ltl (level - 1) in let f2 = gen_ltl (level-1) in BinPOP(EQUIV, f1, f2) 
  | 6 -> let f1 = gen_ltl (level - 1) in Not(f1) 
  | 7 -> let f1 = gen_ltl (level - 1) in UnTOP(YESTERDAY, f1) 
  | 8 -> let f1 = gen_ltl (level - 1) in UnTOP(ONCE, f1) 
  | 9 -> let f1 = gen_ltl (level - 1) in UnTOP(HISTORICALLY, f1) 
  | 10 -> let f1 = gen_ltl (level - 1) in UnTOP(GLOBALLY, f1) 
  | 11 -> let f1 = gen_ltl (level - 1) in UnTOP(EVENTUALLY, f1) 
  | _ -> let f1 = gen_ltl (level - 1) in UnTOP(TOMORROW, f1) 




let gen_ltl_formula _ = 
 let () = Random.self_init () in 
  let max_level = (Random.int 7) in 
  let () = Printf.eprintf "Found level %d\n" max_level in 
  let f = gen_ltl max_level in 
Printf.printf "%s\n" (Ast.formulaToString f) 
