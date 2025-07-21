%{
%}

%token <string> ID 
%token TRUE 
%token FALSE 
%token LBRAC
%token RBRAC
%token COMMA
%token SEMICOLON 
%token ELLIPSIS
%token PROPEQUAL
%token EOF 


  

%start <Traceast.trace> singletrace 

%%

singletrace: 
        | ELLIPSIS; EOF {[]}
        | t1 = state; EOF {[t1]}
        | t1 = state; SEMICOLON; t = singletrace; EOF {t1 :: t}
        ; 
state:
        | LBRAC; t = assignment_list ; RBRAC {t} 
        ; 
assignment_list: 
        | t = assignment;  {t} 
        | t = assignment; COMMA ; trest = assignment_list {List.append t  trest} 
        ; 
assignment: 
        | x = ID; PROPEQUAL; v = bval {[(x, v)]}
        ; 
bval: 
        | TRUE {true} 
        | FALSE {false}
        ;
