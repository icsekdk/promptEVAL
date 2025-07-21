
module MenhirBasics = struct
  
  exception Error
  
  let _eRR =
    fun _s ->
      raise Error
  
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
    | ID of (
# 5 "lib/Parser.mly"
       (string)
# 26 "lib/Parser.ml"
  )
    | HISTORICALLY
    | GLOBALLY
    | FALSE
    | EVENTUALLY
    | EQUIV
    | EOF
    | AND
  
end

include MenhirBasics

# 1 "lib/Parser.mly"
  
open Ast

# 44 "lib/Parser.ml"

type ('s, 'r) _menhir_state = 
  | MenhirState00 : ('s, _menhir_box_formula) _menhir_state
    (** State 00.
        Stack shape : .
        Start symbol: formula. *)

  | MenhirState01 : (('s, _menhir_box_formula) _menhir_cell1_YESTERDAY, _menhir_box_formula) _menhir_state
    (** State 01.
        Stack shape : YESTERDAY.
        Start symbol: formula. *)

  | MenhirState03 : (('s, _menhir_box_formula) _menhir_cell1_TOMORROW, _menhir_box_formula) _menhir_state
    (** State 03.
        Stack shape : TOMORROW.
        Start symbol: formula. *)

  | MenhirState04 : (('s, _menhir_box_formula) _menhir_cell1_ONCE, _menhir_box_formula) _menhir_state
    (** State 04.
        Stack shape : ONCE.
        Start symbol: formula. *)

  | MenhirState05 : (('s, _menhir_box_formula) _menhir_cell1_NOT, _menhir_box_formula) _menhir_state
    (** State 05.
        Stack shape : NOT.
        Start symbol: formula. *)

  | MenhirState06 : (('s, _menhir_box_formula) _menhir_cell1_LPAREN, _menhir_box_formula) _menhir_state
    (** State 06.
        Stack shape : LPAREN.
        Start symbol: formula. *)

  | MenhirState08 : (('s, _menhir_box_formula) _menhir_cell1_HISTORICALLY, _menhir_box_formula) _menhir_state
    (** State 08.
        Stack shape : HISTORICALLY.
        Start symbol: formula. *)

  | MenhirState09 : (('s, _menhir_box_formula) _menhir_cell1_GLOBALLY, _menhir_box_formula) _menhir_state
    (** State 09.
        Stack shape : GLOBALLY.
        Start symbol: formula. *)

  | MenhirState11 : (('s, _menhir_box_formula) _menhir_cell1_EVENTUALLY, _menhir_box_formula) _menhir_state
    (** State 11.
        Stack shape : EVENTUALLY.
        Start symbol: formula. *)

  | MenhirState16 : (('s, _menhir_box_formula) _menhir_cell1_ltlf, _menhir_box_formula) _menhir_state
    (** State 16.
        Stack shape : ltlf.
        Start symbol: formula. *)

  | MenhirState18 : (('s, _menhir_box_formula) _menhir_cell1_ltlf, _menhir_box_formula) _menhir_state
    (** State 18.
        Stack shape : ltlf.
        Start symbol: formula. *)

  | MenhirState21 : (('s, _menhir_box_formula) _menhir_cell1_ltlf, _menhir_box_formula) _menhir_state
    (** State 21.
        Stack shape : ltlf.
        Start symbol: formula. *)

  | MenhirState23 : (('s, _menhir_box_formula) _menhir_cell1_ltlf, _menhir_box_formula) _menhir_state
    (** State 23.
        Stack shape : ltlf.
        Start symbol: formula. *)

  | MenhirState25 : (('s, _menhir_box_formula) _menhir_cell1_ltlf, _menhir_box_formula) _menhir_state
    (** State 25.
        Stack shape : ltlf.
        Start symbol: formula. *)

  | MenhirState27 : (('s, _menhir_box_formula) _menhir_cell1_ltlf, _menhir_box_formula) _menhir_state
    (** State 27.
        Stack shape : ltlf.
        Start symbol: formula. *)


and ('s, 'r) _menhir_cell1_ltlf = 
  | MenhirCell1_ltlf of 's * ('s, 'r) _menhir_state * (Ast.ltlformula)

and ('s, 'r) _menhir_cell1_EVENTUALLY = 
  | MenhirCell1_EVENTUALLY of 's * ('s, 'r) _menhir_state

and ('s, 'r) _menhir_cell1_GLOBALLY = 
  | MenhirCell1_GLOBALLY of 's * ('s, 'r) _menhir_state

and ('s, 'r) _menhir_cell1_HISTORICALLY = 
  | MenhirCell1_HISTORICALLY of 's * ('s, 'r) _menhir_state

and ('s, 'r) _menhir_cell1_LPAREN = 
  | MenhirCell1_LPAREN of 's * ('s, 'r) _menhir_state

and ('s, 'r) _menhir_cell1_NOT = 
  | MenhirCell1_NOT of 's * ('s, 'r) _menhir_state

and ('s, 'r) _menhir_cell1_ONCE = 
  | MenhirCell1_ONCE of 's * ('s, 'r) _menhir_state

and ('s, 'r) _menhir_cell1_TOMORROW = 
  | MenhirCell1_TOMORROW of 's * ('s, 'r) _menhir_state

and ('s, 'r) _menhir_cell1_YESTERDAY = 
  | MenhirCell1_YESTERDAY of 's * ('s, 'r) _menhir_state

and _menhir_box_formula = 
  | MenhirBox_formula of (Ast.ltlformula) [@@unboxed]

let _menhir_action_01 =
  fun f ->
    (
# 40 "lib/Parser.mly"
                        ( f )
# 158 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_02 =
  fun () ->
    (
# 43 "lib/Parser.mly"
               ( LTrue )
# 166 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_03 =
  fun f ->
    (
# 44 "lib/Parser.mly"
                                    ( f )
# 174 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_04 =
  fun () ->
    (
# 45 "lib/Parser.mly"
                ( LFalse )
# 182 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_05 =
  fun x ->
    (
# 46 "lib/Parser.mly"
                 ( Identifier x )
# 190 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_06 =
  fun f ->
    (
# 47 "lib/Parser.mly"
                        ( Not f )
# 198 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_07 =
  fun f ->
    (
# 48 "lib/Parser.mly"
                              ( UnTOP(YESTERDAY, f) )
# 206 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_08 =
  fun f ->
    (
# 49 "lib/Parser.mly"
                             ( UnTOP(TOMORROW, f) )
# 214 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_09 =
  fun f ->
    (
# 50 "lib/Parser.mly"
                         ( UnTOP(ONCE, f) )
# 222 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_10 =
  fun f ->
    (
# 51 "lib/Parser.mly"
                               ( UnTOP(EVENTUALLY, f) )
# 230 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_11 =
  fun f ->
    (
# 52 "lib/Parser.mly"
                                 ( UnTOP(HISTORICALLY, f) )
# 238 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_12 =
  fun f ->
    (
# 53 "lib/Parser.mly"
                             ( UnTOP(GLOBALLY, f) )
# 246 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_13 =
  fun f1 f2 ->
    (
# 54 "lib/Parser.mly"
                                      ( BinTOP( SINCE, f1, f2) )
# 254 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_14 =
  fun f1 f2 ->
    (
# 55 "lib/Parser.mly"
                                      ( BinTOP(UNTIL, f1, f2) )
# 262 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_15 =
  fun f1 f2 ->
    (
# 56 "lib/Parser.mly"
                                    ( BinPOP(AND, f1, f2) )
# 270 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_16 =
  fun f1 f2 ->
    (
# 57 "lib/Parser.mly"
                                   ( BinPOP(OR, f1, f2) )
# 278 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_17 =
  fun f1 f2 ->
    (
# 58 "lib/Parser.mly"
                                      ( BinPOP(IMPLY, f1, f2) )
# 286 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_action_18 =
  fun f1 f2 ->
    (
# 59 "lib/Parser.mly"
                                      ( BinPOP(EQUIV, f1, f2) )
# 294 "lib/Parser.ml"
     : (Ast.ltlformula))

let _menhir_print_token : token -> string =
  fun _tok ->
    match _tok with
    | AND ->
        "AND"
    | EOF ->
        "EOF"
    | EQUIV ->
        "EQUIV"
    | EVENTUALLY ->
        "EVENTUALLY"
    | FALSE ->
        "FALSE"
    | GLOBALLY ->
        "GLOBALLY"
    | HISTORICALLY ->
        "HISTORICALLY"
    | ID _ ->
        "ID"
    | IMPLY ->
        "IMPLY"
    | LPAREN ->
        "LPAREN"
    | NOT ->
        "NOT"
    | ONCE ->
        "ONCE"
    | OR ->
        "OR"
    | RPAREN ->
        "RPAREN"
    | SINCE ->
        "SINCE"
    | TOMORROW ->
        "TOMORROW"
    | TRUE ->
        "TRUE"
    | UNTIL ->
        "UNTIL"
    | YESTERDAY ->
        "YESTERDAY"

let _menhir_fail : unit -> 'a =
  fun () ->
    Printf.eprintf "Internal failure -- please contact the parser generator's developers.\n%!";
    assert false

include struct
  
  [@@@ocaml.warning "-4-37"]
  
  let rec _menhir_run_01 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_YESTERDAY (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState01 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_02 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _tok = _menhir_lexer _menhir_lexbuf in
      let _v = _menhir_action_02 () in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_goto_ltlf : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match _menhir_s with
      | MenhirState00 ->
          _menhir_run_33 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState01 ->
          _menhir_run_32 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | MenhirState03 ->
          _menhir_run_31 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | MenhirState04 ->
          _menhir_run_30 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | MenhirState05 ->
          _menhir_run_29 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | MenhirState27 ->
          _menhir_run_28 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState25 ->
          _menhir_run_26 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState23 ->
          _menhir_run_24 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState21 ->
          _menhir_run_22 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState18 ->
          _menhir_run_19 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState16 ->
          _menhir_run_17 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState06 ->
          _menhir_run_15 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState08 ->
          _menhir_run_14 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | MenhirState09 ->
          _menhir_run_13 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | MenhirState11 ->
          _menhir_run_12 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
  
  and _menhir_run_33 : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | OR ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_21 _menhir_stack _menhir_lexbuf _menhir_lexer
      | IMPLY ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_25 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EQUIV ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_27 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EOF ->
          let f = _v in
          let _v = _menhir_action_01 f in
          MenhirBox_formula _v
      | AND ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_23 _menhir_stack _menhir_lexbuf _menhir_lexer
      | _ ->
          _eRR ()
  
  and _menhir_run_16 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _menhir_s = MenhirState16 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_03 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_TOMORROW (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState03 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_04 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_ONCE (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState04 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_05 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_NOT (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState05 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_06 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_LPAREN (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState06 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_07 : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s ->
      let _tok = _menhir_lexer _menhir_lexbuf in
      let x = _v in
      let _v = _menhir_action_05 x in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_08 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_HISTORICALLY (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState08 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_09 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_GLOBALLY (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState09 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_10 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _tok = _menhir_lexer _menhir_lexbuf in
      let _v = _menhir_action_04 () in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_11 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_formula) _menhir_state -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_EVENTUALLY (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState11 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_18 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _menhir_s = MenhirState18 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_21 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _menhir_s = MenhirState21 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_25 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _menhir_s = MenhirState25 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_27 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _menhir_s = MenhirState27 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_23 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _menhir_s = MenhirState23 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_32 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_YESTERDAY -> _ -> _ -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_YESTERDAY (_menhir_stack, _menhir_s) = _menhir_stack in
      let f = _v in
      let _v = _menhir_action_07 f in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_31 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_TOMORROW -> _ -> _ -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_TOMORROW (_menhir_stack, _menhir_s) = _menhir_stack in
      let f = _v in
      let _v = _menhir_action_08 f in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_30 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_ONCE -> _ -> _ -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_ONCE (_menhir_stack, _menhir_s) = _menhir_stack in
      let f = _v in
      let _v = _menhir_action_09 f in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_29 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_NOT -> _ -> _ -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_NOT (_menhir_stack, _menhir_s) = _menhir_stack in
      let f = _v in
      let _v = _menhir_action_06 f in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_28 : type  ttv_stack. ((ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf as 'stack) -> _ -> _ -> _ -> ('stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | OR ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_21 _menhir_stack _menhir_lexbuf _menhir_lexer
      | IMPLY ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_25 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EQUIV ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_27 _menhir_stack _menhir_lexbuf _menhir_lexer
      | AND ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_23 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EOF | RPAREN ->
          let MenhirCell1_ltlf (_menhir_stack, _menhir_s, f1) = _menhir_stack in
          let f2 = _v in
          let _v = _menhir_action_18 f1 f2 in
          _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_26 : type  ttv_stack. ((ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf as 'stack) -> _ -> _ -> _ -> ('stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | OR ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_21 _menhir_stack _menhir_lexbuf _menhir_lexer
      | IMPLY ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_25 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EQUIV ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_27 _menhir_stack _menhir_lexbuf _menhir_lexer
      | AND ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_23 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EOF | RPAREN ->
          let MenhirCell1_ltlf (_menhir_stack, _menhir_s, f1) = _menhir_stack in
          let f2 = _v in
          let _v = _menhir_action_17 f1 f2 in
          _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_24 : type  ttv_stack. ((ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf as 'stack) -> _ -> _ -> _ -> ('stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | AND | EOF | EQUIV | IMPLY | OR | RPAREN ->
          let MenhirCell1_ltlf (_menhir_stack, _menhir_s, f1) = _menhir_stack in
          let f2 = _v in
          let _v = _menhir_action_15 f1 f2 in
          _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_22 : type  ttv_stack. ((ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf as 'stack) -> _ -> _ -> _ -> ('stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | AND ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_23 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EOF | EQUIV | IMPLY | OR | RPAREN ->
          let MenhirCell1_ltlf (_menhir_stack, _menhir_s, f1) = _menhir_stack in
          let f2 = _v in
          let _v = _menhir_action_16 f1 f2 in
          _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_19 : type  ttv_stack. ((ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf as 'stack) -> _ -> _ -> _ -> ('stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | AND | EOF | EQUIV | IMPLY | OR | RPAREN ->
          let MenhirCell1_ltlf (_menhir_stack, _menhir_s, f1) = _menhir_stack in
          let f2 = _v in
          let _v = _menhir_action_13 f1 f2 in
          _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_17 : type  ttv_stack. ((ttv_stack, _menhir_box_formula) _menhir_cell1_ltlf as 'stack) -> _ -> _ -> _ -> ('stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | AND | EOF | EQUIV | IMPLY | OR | RPAREN ->
          let MenhirCell1_ltlf (_menhir_stack, _menhir_s, f1) = _menhir_stack in
          let f2 = _v in
          let _v = _menhir_action_14 f1 f2 in
          _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_15 : type  ttv_stack. ((ttv_stack, _menhir_box_formula) _menhir_cell1_LPAREN as 'stack) -> _ -> _ -> _ -> ('stack, _menhir_box_formula) _menhir_state -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | UNTIL ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer
      | SINCE ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer
      | RPAREN ->
          let _tok = _menhir_lexer _menhir_lexbuf in
          let MenhirCell1_LPAREN (_menhir_stack, _menhir_s) = _menhir_stack in
          let f = _v in
          let _v = _menhir_action_03 f in
          _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | OR ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_21 _menhir_stack _menhir_lexbuf _menhir_lexer
      | IMPLY ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_25 _menhir_stack _menhir_lexbuf _menhir_lexer
      | EQUIV ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_27 _menhir_stack _menhir_lexbuf _menhir_lexer
      | AND ->
          let _menhir_stack = MenhirCell1_ltlf (_menhir_stack, _menhir_s, _v) in
          _menhir_run_23 _menhir_stack _menhir_lexbuf _menhir_lexer
      | _ ->
          _eRR ()
  
  and _menhir_run_14 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_HISTORICALLY -> _ -> _ -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_HISTORICALLY (_menhir_stack, _menhir_s) = _menhir_stack in
      let f = _v in
      let _v = _menhir_action_11 f in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_13 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_GLOBALLY -> _ -> _ -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_GLOBALLY (_menhir_stack, _menhir_s) = _menhir_stack in
      let f = _v in
      let _v = _menhir_action_12 f in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_run_12 : type  ttv_stack. (ttv_stack, _menhir_box_formula) _menhir_cell1_EVENTUALLY -> _ -> _ -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_EVENTUALLY (_menhir_stack, _menhir_s) = _menhir_stack in
      let f = _v in
      let _v = _menhir_action_10 f in
      _menhir_goto_ltlf _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  let _menhir_run_00 : type  ttv_stack. ttv_stack -> _ -> _ -> _menhir_box_formula =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _menhir_s = MenhirState00 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | YESTERDAY ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TRUE ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | TOMORROW ->
          _menhir_run_03 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ONCE ->
          _menhir_run_04 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | NOT ->
          _menhir_run_05 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | LPAREN ->
          _menhir_run_06 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | ID _v ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | HISTORICALLY ->
          _menhir_run_08 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | GLOBALLY ->
          _menhir_run_09 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | FALSE ->
          _menhir_run_10 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | EVENTUALLY ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
      | _ ->
          _eRR ()
  
end

let formula =
  fun _menhir_lexer _menhir_lexbuf ->
    let _menhir_stack = () in
    let MenhirBox_formula v = _menhir_run_00 _menhir_stack _menhir_lexbuf _menhir_lexer in
    v
