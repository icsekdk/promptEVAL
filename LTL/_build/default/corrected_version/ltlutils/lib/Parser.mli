
(* The type of tokens. *)

type token = 
  | YESTERDAY
  | UNTIL
  | TRUE
  | TOMORROW
  | SINCE
  | RPAREN
  | OR
  | ONCE
  | NOT
  | LPAREN
  | IMPLY
  | ID of (string)
  | HISTORICALLY
  | GLOBALLY
  | FALSE
  | EVENTUALLY
  | EQUIV
  | EOF
  | AND

(* This exception is raised by the monolithic API functions. *)

exception Error

(* The monolithic API. *)

val formula: (Lexing.lexbuf -> token) -> Lexing.lexbuf -> (Ast.ltlformula)
