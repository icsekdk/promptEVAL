{
(* open Lexing  *)
open Traceparser
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
        | "[" { LBRAC } 
        | "]" { RBRAC } 
        | "," { COMMA }
        | ";" { SEMICOLON} 
        | "..." {ELLIPSIS}
        | "=" {PROPEQUAL}
        | id { ID (Lexing.lexeme lexbuf) }
        | eof { EOF }
        | _ { raise (Failure ("Character not allowed in source text: '" ^ Lexing.lexeme lexbuf ^ "'")) } 
        

