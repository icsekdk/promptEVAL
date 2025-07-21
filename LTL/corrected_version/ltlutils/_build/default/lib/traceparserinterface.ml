(* open Lexing *)
(* open Tracelexer *)
(* open Traceparser *)

(* let colnum pos = pos.pos_cnum - pos.pos_bol - 1

let pos_string pos =
  let l = string_of_int pos.pos_lnum and c = string_of_int (colnum pos + 1) in
  "line " ^ l ^ ", column " ^ c

let parse' f s =
  let lexbuf = Lexing.from_string s in
  try f Lexer.read lexbuf
  with Parser.Error ->
    raise (Failure ("Parse error at " ^ pos_string lexbuf.lex_curr_p))

let parse_formula s = parse' Parser.formula s *)

let parse: string -> Traceast.trace 
= fun s ->
  let lexbuf = Lexing.from_string s in 
  Traceparser.singletrace Tracelexer.read lexbuf
