{
open Lexing 
open Parser

let advance_line lexbuf =
  let pos = lexbuf.lex_curr_p in
  let pos' = { pos with
    pos_bol = lexbuf.lex_curr_pos;
    pos_lnum = pos.pos_lnum + 1
  } in
  lexbuf.lex_curr_p <- pos'
}

let white = [' ' '\t']+
let digit = ['0'-'9']
let int = '-'? digit+
let letter = ['a'-'z' 'A'-'Z']
let id = letter ( letter | digit | '_' )*
let mtrue = "TRUE" | "True" | "true" 
let mfalse = "FALSE" | "False" | "false" 
        
rule read = 
        parse 
        | white { read lexbuf } 
        | mtrue { TRUE } 
        | mfalse { FALSE }
        | "!" { NOT } 
        | "~" { NOT } 
        | "YESTERDAY" { YESTERDAY }
        | "(-)" { YESTERDAY} 
        | "Y" {YESTERDAY}
        | "ONCE"  { ONCE }
        (* | "<->" { ONCE } *)
        | "O" { ONCE }
        | "GLOBALLY" { GLOBALLY }
        | "[]" { GLOBALLY }
        | "G" { GLOBALLY }
        | "X" {TOMORROW} 
        | "HISTORICALLY" { HISTORICALLY }
        | "H" { HISTORICALLY }
        | "[-]" { HISTORICALLY }
        | "FUTURE" { EVENTUALLY } 
        | "<>" { EVENTUALLY }
        | "F" { EVENTUALLY }
        | "(" { LPAREN } 
        | ")" { RPAREN } 
        | "SINCE" { SINCE }
        | "S" { SINCE }
        | "UNTIL" { UNTIL }
        | "U" { UNTIL }
        | "&&" { AND } 
        | "&" {AND} 
        | "/\\" { AND } 
        | "\\/" { OR }  
        | "||" { OR }
        | "|" {OR} 
        | "=>" { IMPLY }
        | "==>" {IMPLY}
        | "-->" {IMPLY}
        | "->" { IMPLY }
        | "<==>" { EQUIV }
        | "<-->" { EQUIV }
        | "<->" { EQUIV }
        | id { ID (Lexing.lexeme lexbuf) }
        | eof { EOF }
        | _ { raise (Failure ("Character not allowed in source text: '" ^ Lexing.lexeme lexbuf ^ "'")) } 
        

