%{
open Ast
%}

%token <string> ID 
%token TRUE 
%token FALSE 
%token NOT
%token SINCE 
%token UNTIL 
%token AND 
%token OR 
%token LPAREN
%token RPAREN
%token EQUIV
%token IMPLY 
%token YESTERDAY
%token ONCE 
%token HISTORICALLY 
%token GLOBALLY
%token EVENTUALLY
%token TOMORROW 
%token EOF 

%right  IMPLY EQUIV
%left OR 
%left AND 
%right UNTIL SINCE 
%right HISTORICALLY ONCE GLOBALLY EVENTUALLY  
%right YESTERDAY TOMORROW 
%nonassoc NOT 

  

%start <Ast.ltlformula> formula 

%%

formula: 
        | f = ltlf; EOF { f } 
        ;
ltlf: 
        | TRUE { LTrue }
        | LPAREN; f = ltlf ; RPAREN { f } 
        | FALSE { LFalse } 
        | x = ID { Identifier x } 
        | NOT; f = ltlf { Not f } 
        | YESTERDAY; f = ltlf { UnTOP(YESTERDAY, f) } 
        | TOMORROW; f = ltlf { UnTOP(TOMORROW, f) } 
        | ONCE; f = ltlf { UnTOP(ONCE, f) } 
        | EVENTUALLY; f = ltlf { UnTOP(EVENTUALLY, f) } 
        | HISTORICALLY; f = ltlf { UnTOP(HISTORICALLY, f) } 
        | GLOBALLY; f = ltlf { UnTOP(GLOBALLY, f) } 
        | f1 = ltlf; SINCE; f2 = ltlf { BinTOP( SINCE, f1, f2) } 
        | f1 = ltlf; UNTIL; f2 = ltlf { BinTOP(UNTIL, f1, f2) } 
        | f1 = ltlf; AND; f2 = ltlf { BinPOP(AND, f1, f2) } 
        | f1 = ltlf; OR; f2 = ltlf { BinPOP(OR, f1, f2) } 
        | f1 = ltlf; IMPLY; f2 = ltlf { BinPOP(IMPLY, f1, f2) } 
        | f1 = ltlf; EQUIV; f2 = ltlf { BinPOP(EQUIV, f1, f2) } 
        ; 
