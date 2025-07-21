
module MenhirBasics = struct
  
  exception Error
  
  let _eRR =
    fun _s ->
      raise Error
  
  type token = 
    | TRUE
    | SEMICOLON
    | RBRAC
    | PROPEQUAL
    | LBRAC
    | ID of (
# 4 "corrected_version/ltlutils/lib/traceparser.mly"
       (string)
# 20 "corrected_version/ltlutils/lib/traceparser.ml"
  )
    | FALSE
    | EOF
    | ELLIPSIS
    | COMMA
  
end

include MenhirBasics

# 1 "corrected_version/ltlutils/lib/traceparser.mly"
  

# 34 "corrected_version/ltlutils/lib/traceparser.ml"

type ('s, 'r) _menhir_state = 
  | MenhirState00 : ('s, _menhir_box_singletrace) _menhir_state
    (** State 00.
        Stack shape : .
        Start symbol: singletrace. *)

  | MenhirState01 : (('s, _menhir_box_singletrace) _menhir_cell1_LBRAC, _menhir_box_singletrace) _menhir_state
    (** State 01.
        Stack shape : LBRAC.
        Start symbol: singletrace. *)

  | MenhirState10 : (('s, _menhir_box_singletrace) _menhir_cell1_assignment, _menhir_box_singletrace) _menhir_state
    (** State 10.
        Stack shape : assignment.
        Start symbol: singletrace. *)

  | MenhirState15 : (('s, _menhir_box_singletrace) _menhir_cell1_state, _menhir_box_singletrace) _menhir_state
    (** State 15.
        Stack shape : state.
        Start symbol: singletrace. *)

  | MenhirState19 : (('s, _menhir_box_singletrace) _menhir_cell1_state, _menhir_box_singletrace) _menhir_state
    (** State 19.
        Stack shape : state.
        Start symbol: singletrace. *)


and ('s, 'r) _menhir_cell1_assignment = 
  | MenhirCell1_assignment of 's * ('s, 'r) _menhir_state * (Traceast.trace_state)

and ('s, 'r) _menhir_cell1_state = 
  | MenhirCell1_state of 's * ('s, 'r) _menhir_state * (Traceast.trace_state)

and ('s, 'r) _menhir_cell1_ID = 
  | MenhirCell1_ID of 's * ('s, 'r) _menhir_state * (
# 4 "corrected_version/ltlutils/lib/traceparser.mly"
       (string)
# 73 "corrected_version/ltlutils/lib/traceparser.ml"
)

and ('s, 'r) _menhir_cell1_LBRAC = 
  | MenhirCell1_LBRAC of 's * ('s, 'r) _menhir_state

and _menhir_box_singletrace = 
  | MenhirBox_singletrace of (Traceast.trace) [@@unboxed]

let _menhir_action_01 =
  fun v x ->
    (
# 35 "corrected_version/ltlutils/lib/traceparser.mly"
                                      ([(x, v)])
# 87 "corrected_version/ltlutils/lib/traceparser.ml"
     : (Traceast.trace_state))

let _menhir_action_02 =
  fun t ->
    (
# 31 "corrected_version/ltlutils/lib/traceparser.mly"
                           (t)
# 95 "corrected_version/ltlutils/lib/traceparser.ml"
     : (Traceast.trace_state))

let _menhir_action_03 =
  fun t trest ->
    (
# 32 "corrected_version/ltlutils/lib/traceparser.mly"
                                                          (List.append t  trest)
# 103 "corrected_version/ltlutils/lib/traceparser.ml"
     : (Traceast.trace_state))

let _menhir_action_04 =
  fun () ->
    (
# 38 "corrected_version/ltlutils/lib/traceparser.mly"
               (true)
# 111 "corrected_version/ltlutils/lib/traceparser.ml"
     : (bool))

let _menhir_action_05 =
  fun () ->
    (
# 39 "corrected_version/ltlutils/lib/traceparser.mly"
                (false)
# 119 "corrected_version/ltlutils/lib/traceparser.ml"
     : (bool))

let _menhir_action_06 =
  fun () ->
    (
# 23 "corrected_version/ltlutils/lib/traceparser.mly"
                        ([])
# 127 "corrected_version/ltlutils/lib/traceparser.ml"
     : (Traceast.trace))

let _menhir_action_07 =
  fun t1 ->
    (
# 24 "corrected_version/ltlutils/lib/traceparser.mly"
                          ([t1])
# 135 "corrected_version/ltlutils/lib/traceparser.ml"
     : (Traceast.trace))

let _menhir_action_08 =
  fun t t1 ->
    (
# 25 "corrected_version/ltlutils/lib/traceparser.mly"
                                                      (t1 :: t)
# 143 "corrected_version/ltlutils/lib/traceparser.ml"
     : (Traceast.trace))

let _menhir_action_09 =
  fun t ->
    (
# 28 "corrected_version/ltlutils/lib/traceparser.mly"
                                             (t)
# 151 "corrected_version/ltlutils/lib/traceparser.ml"
     : (Traceast.trace_state))

let _menhir_print_token : token -> string =
  fun _tok ->
    match _tok with
    | COMMA ->
        "COMMA"
    | ELLIPSIS ->
        "ELLIPSIS"
    | EOF ->
        "EOF"
    | FALSE ->
        "FALSE"
    | ID _ ->
        "ID"
    | LBRAC ->
        "LBRAC"
    | PROPEQUAL ->
        "PROPEQUAL"
    | RBRAC ->
        "RBRAC"
    | SEMICOLON ->
        "SEMICOLON"
    | TRUE ->
        "TRUE"

let _menhir_fail : unit -> 'a =
  fun () ->
    Printf.eprintf "Internal failure -- please contact the parser generator's developers.\n%!";
    assert false

include struct
  
  [@@@ocaml.warning "-4-37"]
  
  let _menhir_run_26 : type  ttv_stack. ttv_stack -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _v ->
      MenhirBox_singletrace _v
  
  let rec _menhir_goto_singletrace : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match _menhir_s with
      | MenhirState00 ->
          _menhir_run_26 _menhir_stack _v
      | MenhirState15 ->
          _menhir_run_23 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | MenhirState19 ->
          _menhir_run_20 _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
      | _ ->
          _menhir_fail ()
  
  and _menhir_run_23 : type  ttv_stack. (ttv_stack, _menhir_box_singletrace) _menhir_cell1_state -> _ -> _ -> _ -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      match (_tok : MenhirBasics.token) with
      | EOF ->
          let MenhirCell1_state (_menhir_stack, _menhir_s, t1) = _menhir_stack in
          let t = _v in
          let _v = _menhir_action_08 t t1 in
          _menhir_goto_singletrace _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_20 : type  ttv_stack. (ttv_stack, _menhir_box_singletrace) _menhir_cell1_state -> _ -> _ -> _ -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      match (_tok : MenhirBasics.token) with
      | EOF ->
          let _tok = _menhir_lexer _menhir_lexbuf in
          let MenhirCell1_state (_menhir_stack, _menhir_s, t1) = _menhir_stack in
          let t = _v in
          let _v = _menhir_action_08 t t1 in
          _menhir_goto_singletrace _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  let _menhir_run_16 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | EOF ->
          let _tok = _menhir_lexer _menhir_lexbuf in
          let _v = _menhir_action_06 () in
          _menhir_goto_singletrace _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  let rec _menhir_run_01 : type  ttv_stack. ttv_stack -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s ->
      let _menhir_stack = MenhirCell1_LBRAC (_menhir_stack, _menhir_s) in
      let _menhir_s = MenhirState01 in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | ID _v ->
          _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_run_02 : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s ->
      let _menhir_stack = MenhirCell1_ID (_menhir_stack, _menhir_s, _v) in
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | PROPEQUAL ->
          let _tok = _menhir_lexer _menhir_lexbuf in
          (match (_tok : MenhirBasics.token) with
          | TRUE ->
              let _tok = _menhir_lexer _menhir_lexbuf in
              let _v = _menhir_action_04 () in
              _menhir_goto_bval _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
          | FALSE ->
              let _tok = _menhir_lexer _menhir_lexbuf in
              let _v = _menhir_action_05 () in
              _menhir_goto_bval _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok
          | _ ->
              _eRR ())
      | _ ->
          _eRR ()
  
  and _menhir_goto_bval : type  ttv_stack. (ttv_stack, _menhir_box_singletrace) _menhir_cell1_ID -> _ -> _ -> _ -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _tok ->
      let MenhirCell1_ID (_menhir_stack, _menhir_s, x) = _menhir_stack in
      let v = _v in
      let _v = _menhir_action_01 v x in
      match (_tok : MenhirBasics.token) with
      | COMMA ->
          let _menhir_stack = MenhirCell1_assignment (_menhir_stack, _menhir_s, _v) in
          let _menhir_s = MenhirState10 in
          let _tok = _menhir_lexer _menhir_lexbuf in
          (match (_tok : MenhirBasics.token) with
          | ID _v ->
              _menhir_run_02 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
          | _ ->
              _eRR ())
      | RBRAC ->
          let t = _v in
          let _v = _menhir_action_02 t in
          _menhir_goto_assignment_list _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
      | _ ->
          _eRR ()
  
  and _menhir_goto_assignment_list : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s ->
      match _menhir_s with
      | MenhirState10 ->
          _menhir_run_11 _menhir_stack _menhir_lexbuf _menhir_lexer _v
      | MenhirState01 ->
          _menhir_run_07 _menhir_stack _menhir_lexbuf _menhir_lexer _v
      | _ ->
          _menhir_fail ()
  
  and _menhir_run_11 : type  ttv_stack. (ttv_stack, _menhir_box_singletrace) _menhir_cell1_assignment -> _ -> _ -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v ->
      let MenhirCell1_assignment (_menhir_stack, _menhir_s, t) = _menhir_stack in
      let trest = _v in
      let _v = _menhir_action_03 t trest in
      _menhir_goto_assignment_list _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s
  
  and _menhir_run_07 : type  ttv_stack. (ttv_stack, _menhir_box_singletrace) _menhir_cell1_LBRAC -> _ -> _ -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v ->
      let _tok = _menhir_lexer _menhir_lexbuf in
      let MenhirCell1_LBRAC (_menhir_stack, _menhir_s) = _menhir_stack in
      let t = _v in
      let _v = _menhir_action_09 t in
      _menhir_goto_state _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
  
  and _menhir_goto_state : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match _menhir_s with
      | MenhirState19 ->
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState15 ->
          _menhir_run_18 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | MenhirState00 ->
          _menhir_run_14 _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _menhir_fail ()
  
  and _menhir_run_18 : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | SEMICOLON ->
          let _menhir_stack = MenhirCell1_state (_menhir_stack, _menhir_s, _v) in
          let _menhir_s = MenhirState19 in
          let _tok = _menhir_lexer _menhir_lexbuf in
          (match (_tok : MenhirBasics.token) with
          | LBRAC ->
              _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
          | ELLIPSIS ->
              _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
          | _ ->
              _eRR ())
      | EOF ->
          let _tok = _menhir_lexer _menhir_lexbuf in
          let t1 = _v in
          let _v = _menhir_action_07 t1 in
          _menhir_goto_singletrace _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  and _menhir_run_14 : type  ttv_stack. ttv_stack -> _ -> _ -> _ -> (ttv_stack, _menhir_box_singletrace) _menhir_state -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok ->
      match (_tok : MenhirBasics.token) with
      | SEMICOLON ->
          let _menhir_stack = MenhirCell1_state (_menhir_stack, _menhir_s, _v) in
          let _menhir_s = MenhirState15 in
          let _tok = _menhir_lexer _menhir_lexbuf in
          (match (_tok : MenhirBasics.token) with
          | LBRAC ->
              _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
          | ELLIPSIS ->
              _menhir_run_16 _menhir_stack _menhir_lexbuf _menhir_lexer _menhir_s
          | _ ->
              _eRR ())
      | EOF ->
          let t1 = _v in
          let _v = _menhir_action_07 t1 in
          _menhir_goto_singletrace _menhir_stack _menhir_lexbuf _menhir_lexer _v _menhir_s _tok
      | _ ->
          _eRR ()
  
  let _menhir_run_00 : type  ttv_stack. ttv_stack -> _ -> _ -> _menhir_box_singletrace =
    fun _menhir_stack _menhir_lexbuf _menhir_lexer ->
      let _tok = _menhir_lexer _menhir_lexbuf in
      match (_tok : MenhirBasics.token) with
      | LBRAC ->
          _menhir_run_01 _menhir_stack _menhir_lexbuf _menhir_lexer MenhirState00
      | ELLIPSIS ->
          let _tok = _menhir_lexer _menhir_lexbuf in
          (match (_tok : MenhirBasics.token) with
          | EOF ->
              let _v = _menhir_action_06 () in
              _menhir_run_26 _menhir_stack _v
          | _ ->
              _eRR ())
      | _ ->
          _eRR ()
  
end

let singletrace =
  fun _menhir_lexer _menhir_lexbuf ->
    let _menhir_stack = () in
    let MenhirBox_singletrace v = _menhir_run_00 _menhir_stack _menhir_lexbuf _menhir_lexer in
    v
