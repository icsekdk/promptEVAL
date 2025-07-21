
(* The type of tokens. *)

type token = 
  | TRUE
  | SEMICOLON
  | RBRAC
  | PROPEQUAL
  | LBRAC
  | ID of (string)
  | FALSE
  | EOF
  | ELLIPSIS
  | COMMA

(* This exception is raised by the monolithic API functions. *)

exception Error

(* The monolithic API. *)

val singletrace: (Lexing.lexbuf -> token) -> Lexing.lexbuf -> (Traceast.trace)
