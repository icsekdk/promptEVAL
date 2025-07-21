type bintemporalop = SINCE | UNTIL
type binlogicalop = AND | OR | EQUIV | IMPLY

type unarytemporalop =
  | YESTERDAY
  | ONCE
  | HISTORICALLY
  | GLOBALLY
  | EVENTUALLY
  | TOMORROW

type ltlformula =
  | Identifier of string
  | LTrue
  | LFalse
  | BinTOP of bintemporalop * ltlformula * ltlformula
  | UnTOP of unarytemporalop * ltlformula
  | Not of ltlformula
  | BinPOP of binlogicalop * ltlformula * ltlformula

let bintopToString f = match f with SINCE -> " S " | UNTIL -> " U "

let binlogToString f =
  match f with AND -> " & " | OR -> " | " | EQUIV -> " <-> " | IMPLY -> " -> "

let binlogToSignature f =
    match f with AND -> "&" | OR -> "|" | EQUIV -> " <=> " | IMPLY -> "=>"
  

let untopToString f =
  match f with
  | YESTERDAY -> "Y"
  | ONCE -> "O"
  | HISTORICALLY -> "H"
  | GLOBALLY -> "G"
  | EVENTUALLY -> "F"
  | TOMORROW -> "X"

let convert_string_list_to_one_string s = List.fold_left String.cat "" s


let rec formulaToSignatureFormat f = 
  match f with
  | Identifier var -> var
  | LTrue -> "TRUE"
  | LFalse -> "FALSE"
  | BinTOP (btop, f1, f2) ->
      convert_string_list_to_one_string
        [
          bintopToString btop;
          "(";
          formulaToSignatureFormat f1;
          ", ";
          formulaToSignatureFormat f2;
          ")";
        ]
  | UnTOP (untop, f1) ->
      convert_string_list_to_one_string
        [ untopToString untop; " ("; formulaToSignatureFormat f1; ")" ]
  | Not f1 ->
      convert_string_list_to_one_string [ "!"; "("; formulaToSignatureFormat f1; ")" ]
  | BinPOP (bpop, f1, f2) ->
      convert_string_list_to_one_string
        [
          binlogToSignature bpop;
          "(";
          formulaToSignatureFormat f1;
          ", ";
          formulaToSignatureFormat f2;
          ")";
        ]


let rec formulaToString f =
  match f with
  | Identifier var -> var
  | LTrue -> "TRUE"
  | LFalse -> "FALSE"
  | BinTOP (btop, f1, f2) ->
      convert_string_list_to_one_string
        [
          "(";
          formulaToString f1;
          " ";
          bintopToString btop;
          " ";
          formulaToString f2;
          ")";
        ]
  | UnTOP (untop, f1) ->
      convert_string_list_to_one_string
        [ untopToString untop; " ("; formulaToString f1; ")" ]
  | Not f1 ->
      convert_string_list_to_one_string [ "!"; "("; formulaToString f1; ")" ]
  | BinPOP (bpop, f1, f2) ->
      convert_string_list_to_one_string
        [
          "(";
          formulaToString f1;
          " ";
          binlogToString bpop;
          " ";
          formulaToString f2;
          ")";
        ]
