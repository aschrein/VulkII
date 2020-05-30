#include "rendering.hpp"

#define ALLOC_VAL() (Value *)alloc_value()
#define CALL_EVAL(x)                                                                               \
  eval_unwrap(x);                                                                                  \
  CHECK_ERROR()
#define ASSERT_SMB(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::SYMBOL);
#define ASSERT_I32(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::I32);
#define ASSERT_F32(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::F32);
#define ASSERT_ANY(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::ANY);

#define EVAL_SMB(res, id)                                                                          \
  Value *res = eval_unwrap(l->get(id));                                                            \
  ASSERT_SMB(res)
#define EVAL_I32(res, id)                                                                          \
  Value *res = eval_unwrap(l->get(id));                                                            \
  ASSERT_I32(res)
#define EVAL_F32(res, id)                                                                          \
  Value *res = eval_unwrap(l->get(id));                                                            \
  ASSERT_F32(res)
#define EVAL_ANY(res, id)                                                                          \
  Value *res = eval_unwrap(l->get(id));                                                            \
  ASSERT_ANY(res)


struct Default_Evaluator final : public IEvaluator {
  void               init() {}
  Default_Evaluator *create() {
    Default_Evaluator *out = new Default_Evaluator;
    out->init();
    return out;
  }
  void  release() override { delete this; }
  Match eval(List *l) override {
    if (l == NULL) return NULL;
    TMP_STORAGE_SCOPE;
    if (l->child != NULL) {
      ASSERT_EVAL(!l->nonempty());
      return global_eval(l->child);
    } else if (l->nonempty()) {
      i32  imm32;
      f32  immf32;
      bool is_imm32  = !l->quoted && parse_decimal_int(l->symbol.ptr, l->symbol.len, &imm32);
      bool is_immf32 = !l->quoted && parse_float(l->symbol.ptr, l->symbol.len, &immf32);
      if (is_imm32) {
        Value *new_val = ALLOC_VAL();
        new_val->i     = imm32;
        new_val->type  = (i32)Value::Value_t::I32;
        return new_val;
      } else if (is_immf32) {
        Value *new_val = ALLOC_VAL();
        new_val->f     = immf32;
        new_val->type  = (i32)Value::Value_t::F32;
        return new_val;
      } else if (l->cmp_symbol("for-range")) {
        List *name_l = l->next;
        ASSERT_EVAL(name_l->nonempty());
        string_ref name = name_l->symbol;
        Value *    lb   = CALL_EVAL(l->get(2));
        ASSERT_EVAL(lb != NULL && lb->type == (i32)Value::Value_t::I32);
        Value *ub = CALL_EVAL(l->get(3));
        ASSERT_EVAL(ub != NULL && ub->type == (i32)Value::Value_t::I32);
        Value *new_val = ALLOC_VAL();
        new_val->i     = 0;
        new_val->type  = (i32)Value::Value_t::I32;
        for (i32 i = lb->i; i < ub->i; i++) {
          state->symbol_table.enter_scope();
          new_val->i = i;
          state->symbol_table.add_symbol(name, new_val);
          defer(state->symbol_table.exit_scope());
          eval_args(l->get(4));
        }
        return NULL;
      } else if (l->cmp_symbol("for-items")) {
        Value *name = CALL_EVAL(l->next);
        ASSERT_EVAL(name->type == (i32)Value::Value_t::SYMBOL);
        SmallArray<Value *, 8> items;
        items.init();
        defer(items.release());
        eval_args_and_collect(l->next->next->child, items);
        ito(items.size) {
          state->symbol_table.enter_scope();
          state->symbol_table.add_symbol(name->str, items[i]);
          defer(state->symbol_table.exit_scope());
          eval_args(l->get(3));
        }
        return NULL;
      } else if (l->cmp_symbol("if")) {
        EVAL_I32(cond, 1);
        state->symbol_table.enter_scope();
        defer(state->symbol_table.exit_scope());
        if (cond->i != 0) {
          Value *val = CALL_EVAL(l->get(2));
          return val;
        } else {
          Value *val = CALL_EVAL(l->get(3));
          return val;
        }
      } else if (l->cmp_symbol("add-mode")) {
        EVAL_SMB(name, 1);
        state->symbol_table.enter_scope();
        defer(state->symbol_table.exit_scope());
        IEvaluator *mode = IEvaluator::create_mode(name->str);
        ASSERT_EVAL(mode != NULL);
        IEvaluator *old_head = IEvaluator::get_head();
        IEvaluator::set_head(mode);
        eval_args(l->get(2));
        IEvaluator::set_head(old_head);
        mode->release();
        return NULL;
      } else if (l->cmp_symbol("lambda")) {
        Value *new_val = ALLOC_VAL();
        new_val->list  = l->next;
        new_val->type  = (i32)Value::Value_t::LAMBDA;
        return new_val;
      } else if (l->cmp_symbol("scope")) {
        state->symbol_table.enter_scope();
        defer(state->symbol_table.exit_scope());
        return eval_args(l->next);
      } else if (l->cmp_symbol("add")) {
        SmallArray<Value *, 2> args;
        args.init();
        defer(args.release());
        eval_args_and_collect(l->next, args);
        ASSERT_EVAL(args.size == 2);
        Value *op1 = args[0];
        ASSERT_EVAL(op1 != NULL);
        Value *op2 = args[1];
        ASSERT_EVAL(op2 != NULL);
        ASSERT_EVAL(op1->type == op2->type);
        if (op1->type == (i32)Value::Value_t::I32) {
          Value *new_val = ALLOC_VAL();
          new_val->i     = op1->i + op2->i;
          new_val->type  = (i32)Value::Value_t::I32;
          return new_val;
        } else if (op1->type == (i32)Value::Value_t::F32) {
          Value *new_val = ALLOC_VAL();
          new_val->f     = op1->f + op2->f;
          new_val->type  = (i32)Value::Value_t::F32;
          return new_val;
        } else {
          ASSERT_EVAL(false && "add: unsopported operand types");
        }
        return NULL;
      } else if (l->cmp_symbol("sub")) {
        SmallArray<Value *, 2> args;
        args.init();
        defer(args.release());
        eval_args_and_collect(l->next, args);
        ASSERT_EVAL(args.size == 2);
        Value *op1 = args[0];
        ASSERT_EVAL(op1 != NULL);
        Value *op2 = args[1];
        ASSERT_EVAL(op2 != NULL);
        ASSERT_EVAL(op1->type == op2->type);
        if (op1->type == (i32)Value::Value_t::I32) {
          Value *new_val = ALLOC_VAL();
          new_val->i     = op1->i - op2->i;
          new_val->type  = (i32)Value::Value_t::I32;
          return new_val;
        } else if (op1->type == (i32)Value::Value_t::F32) {
          Value *new_val = ALLOC_VAL();
          new_val->f     = op1->f - op2->f;
          new_val->type  = (i32)Value::Value_t::F32;
          return new_val;
        } else {
          ASSERT_EVAL(false && "sub: unsopported operand types");
        }
        return NULL;
      } else if (l->cmp_symbol("mul")) {
        SmallArray<Value *, 2> args;
        args.init();
        defer(args.release());
        eval_args_and_collect(l->next, args);
        ASSERT_EVAL(args.size == 2);
        Value *op1 = args[0];
        ASSERT_EVAL(op1 != NULL);
        Value *op2 = args[1];
        ASSERT_EVAL(op2 != NULL);
        ASSERT_EVAL(op1->type == op2->type);
        if (op1->type == (i32)Value::Value_t::I32) {
          Value *new_val = ALLOC_VAL();
          new_val->i     = op1->i * op2->i;
          new_val->type  = (i32)Value::Value_t::I32;
          return new_val;
        } else if (op1->type == (i32)Value::Value_t::F32) {
          Value *new_val = ALLOC_VAL();
          new_val->f     = op1->f * op2->f;
          new_val->type  = (i32)Value::Value_t::F32;
          return new_val;
        } else {
          ASSERT_EVAL(false && "mul: unsopported operand types");
        }
        return NULL;
      } else if (l->cmp_symbol("cmp")) {
        List *                 mode = l->next;
        SmallArray<Value *, 2> args;
        args.init();
        defer(args.release());
        eval_args_and_collect(mode->next, args);
        ASSERT_EVAL(args.size == 2);
        Value *op1 = args[0];
        ASSERT_EVAL(op1 != NULL);
        Value *op2 = args[1];
        ASSERT_EVAL(op2 != NULL);
        ASSERT_EVAL(op1->type == op2->type);
        if (mode->cmp_symbol("lt")) {
          if (op1->type == (i32)Value::Value_t::I32) {
            Value *new_val = ALLOC_VAL();
            new_val->i     = op1->i < op2->i ? 1 : 0;
            new_val->type  = (i32)Value::Value_t::I32;
            return new_val;
          } else if (op1->type == (i32)Value::Value_t::F32) {
            Value *new_val = ALLOC_VAL();
            new_val->i     = op1->f < op2->f ? 1 : 0;
            new_val->type  = (i32)Value::Value_t::I32;
            return new_val;
          } else {
            ASSERT_EVAL(false && "cmp: unsopported operand types");
          }
        } else if (mode->cmp_symbol("eq")) {
          if (op1->type == (i32)Value::Value_t::I32) {
            Value *new_val = ALLOC_VAL();
            new_val->i     = op1->i == op2->i ? 1 : 0;
            new_val->type  = (i32)Value::Value_t::I32;
            return new_val;
          } else if (op1->type == (i32)Value::Value_t::F32) {
            Value *new_val = ALLOC_VAL();
            new_val->i     = op1->f == op2->f ? 1 : 0;
            new_val->type  = (i32)Value::Value_t::I32;
            return new_val;
          } else {
            ASSERT_EVAL(false && "cmp: unsopported operand types");
          }
        } else {
          ASSERT_EVAL(false && "cmp: unsopported op");
        }
        return NULL;
      } else if (l->cmp_symbol("let")) {
        List *name = l->next;
        ASSERT_EVAL(name->nonempty());
        Value *val = CALL_EVAL(l->get(2));
        state->symbol_table.add_symbol(name->symbol, val);
        return val;
      } else if (l->cmp_symbol("get-scope")) {
        Value *new_val = ALLOC_VAL();
        new_val->any   = state->symbol_table.get_scope();
        new_val->type  = (i32)Value::Value_t::SCOPE;
        return new_val;
      } else if (l->cmp_symbol("set-scope")) {
        Value *val = CALL_EVAL(l->get(1));
        ASSERT_EVAL(val != NULL && val->type == (i32)Value::Value_t::SCOPE);
        void *old_scope = state->symbol_table.get_scope();
        state->symbol_table.set_scope(val->any);
        state->symbol_table.enter_scope();
        defer({
          state->symbol_table.exit_scope();
          state->symbol_table.set_scope(old_scope);
        });
        { // Preserve list
          List *cur = l->get(2)->child;
          while (cur != NULL) {
            if (cur->nonempty()) {
              state->symbol_table.add_symbol(
                  cur->symbol, state->symbol_table.lookup_value(cur->symbol, old_scope));
            }
            cur = cur->next;
          }
        }
        return eval_args(l->get(3));
      } else if (l->cmp_symbol("get-mode")) {
        Value *new_val = ALLOC_VAL();
        new_val->any   = get_head();
        new_val->type  = (i32)Value::Value_t::MODE;
        return new_val;
      } else if (l->cmp_symbol("set-mode")) {
        Value *val = CALL_EVAL(l->get(1));
        ASSERT_EVAL(val != NULL && val->type == (i32)Value::Value_t::MODE);
        IEvaluator *old_mode = get_head();
        set_head((IEvaluator *)val->any);
        defer(set_head(old_mode););
        return eval_args(l->get(2));
      } else if (l->cmp_symbol("quote")) {
        Value *new_val = ALLOC_VAL();
        new_val->list  = l->next;
        new_val->type  = (i32)Value::Value_t::BINDING;
        return new_val;
      } else if (l->cmp_symbol("deref")) {
        return state->symbol_table.lookup_value(l->next->symbol);
      } else if (l->cmp_symbol("unbind")) {
        ASSERT_EVAL(l->next->nonempty());
        Value *sym = state->symbol_table.lookup_value(l->next->symbol);
        ASSERT_EVAL(sym != NULL && sym->type == (i32)Value::Value_t::BINDING);
        return global_eval(sym->list);
      } else if (l->cmp_symbol("unquote")) {
        ASSERT_EVAL(l->next->nonempty());
        Value *sym = state->symbol_table.lookup_value(l->next->symbol);
        ASSERT_EVAL(sym != NULL && sym->type == (i32)Value::Value_t::BINDING);
        ASSERT_EVAL(sym->list->nonempty());
        Value *new_val = ALLOC_VAL();
        new_val->str   = sym->list->symbol;
        new_val->type  = (i32)Value::Value_t::SYMBOL;
        return new_val;
      } else if (l->cmp_symbol("nil")) {
        return NULL;
      } else if (l->cmp_symbol("print")) {
        EVAL_SMB(str, 1);
        fprintf(stdout, "%.*s\n", STRF(str->str));
        return NULL;
      } else if (l->cmp_symbol("format")) {
        // state->symbol_table.dump();
        SmallArray<Value *, 4> args;
        args.init();
        defer({ args.release(); });
        eval_args_and_collect(l->next, args);
        Value *fmt    = args[0];
        u32    cur_id = 1;
        {
          char *      tmp_buf = (char *)tl_alloc_tmp(0x100);
          u32         cursor  = 0;
          char const *c       = fmt->str.ptr;
          char const *end     = fmt->str.ptr + fmt->str.len;
          while (c != end) {
            if (c[0] == '%') {
              if (c + 1 == end) {
                ASSERT_EVAL(false && "[format] Format string ends with %%");
              }
              if (cur_id == args.size) {
                ASSERT_EVAL(false && "[format] Not enough arguments");
              } else {
                i32    num_chars = 0;
                Value *val       = args[cur_id];
                if (c[1] == 'i') {
                  ASSERT_EVAL(val != NULL && val->type == (i32)Value::Value_t::I32);
                  num_chars = sprintf(tmp_buf + cursor, "%i", val->i);
                } else if (c[1] == 'f') {
                  ASSERT_EVAL(val != NULL && val->type == (i32)Value::Value_t::F32);
                  num_chars = sprintf(tmp_buf + cursor, "%f", val->f);
                } else if (c[1] == 's') {
                  ASSERT_EVAL(val != NULL && val->type == (i32)Value::Value_t::SYMBOL);
                  num_chars = sprintf(tmp_buf + cursor, "%.*s", (i32)val->str.len, val->str.ptr);
                } else {
                  ASSERT_EVAL(false && "[format]  Unknown format");
                }
                if (num_chars < 0) {
                  ASSERT_EVAL(false && "[format] Blimey!");
                }
                if (num_chars > 0x100) {
                  ASSERT_EVAL(false && "[format] Format buffer overflow!");
                }
                cursor += num_chars;
              }
              cur_id += 1;
              c += 1;
            } else {
              tmp_buf[cursor++] = c[0];
            }
            c += 1;
          }
          tmp_buf[cursor] = '\0';
          Value *new_val  = ALLOC_VAL();
          new_val->str    = move_cstr(stref_s(tmp_buf));
          new_val->type   = (i32)Value::Value_t::SYMBOL;
          return new_val;
        }
      } else {
        ASSERT_EVAL(l->nonempty());
        Value *sym = state->symbol_table.lookup_value(l->symbol);
        if (sym != NULL) {
          if (sym->type == (i32)Value::Value_t::LAMBDA) {
            ASSERT_EVAL(sym->list->child != NULL);
            List *lambda   = sym->list; // Try to evaluate
            List *arg_name = lambda->child;
            List *arg_val  = l->next;
            state->symbol_table.enter_scope();
            defer(state->symbol_table.exit_scope());
            bool saw_vararg = false;
            while (arg_name != NULL && arg_name->nonempty()) { // Bind arguments
              ASSERT_EVAL(!saw_vararg && "vararg must be the last argument");
              ASSERT_EVAL(arg_val != NULL);
              ASSERT_EVAL(arg_name->nonempty());
              if (arg_name->cmp_symbol("...")) {
                Value *new_val = ALLOC_VAL();
                new_val->list  = arg_val;
                new_val->type  = (i32)Value::Value_t::BINDING;
                state->symbol_table.add_symbol(arg_name->symbol, new_val);
                saw_vararg = true;
              } else {
                Value *val = CALL_EVAL(arg_val);
                state->symbol_table.add_symbol(arg_name->symbol, val);
              }
              arg_name = arg_name->next;
              arg_val  = arg_val->next;
            }
            return eval_args(lambda->next);
          } else if (sym->type == (i32)Value::Value_t::BINDING) {
            //            Value *val = CALL_EVAL(sym->list);
            Value *val = sym;
            return val;
          }
          return sym;
        }
        Value *new_val = ALLOC_VAL();
        new_val->str   = l->symbol;
        new_val->type  = (i32)Value::Value_t::SYMBOL;
        return new_val;
      }
    }
    TRAP;
  }
};

IEvaluator *g_head = NULL;

IEvaluator *IEvaluator::get_head() {
  if (g_head == NULL) {
    Default_Evaluator *head = new Default_Evaluator();
    head->init();
    g_head = head;
  }
  return g_head;
}

void IEvaluator::set_head(IEvaluator *newhead) { g_head = newhead; }

Hash_Table<string_ref, Evaluator_Creator_t> &get_factory_table() {
  static Hash_Table<string_ref, Evaluator_Creator_t> table;
  static int                                         _init = [&] {
    table.init();
    return 0;
  }();
  (void)_init;
  return table;
}

void IEvaluator::add_mode(string_ref name, Evaluator_Creator_t creat) {
  get_factory_table().insert(name, creat);
}

IEvaluator *IEvaluator::create_mode(string_ref name) {
  if (get_factory_table().contains(name)) {
    IEvaluator *out = (*get_factory_table().get(name))();
    out->prev       = get_head();
    out->state      = get_head()->state;
    return out;
  }
  return NULL;
}

void parse_and_eval(string_ref text) {
  Evaluator_State state;
  state.init();
  defer(state.release());

  struct List_Allocator {
    Evaluator_State *state;
    List *           alloc() {
      List *out = state->list_storage.alloc_zero(1);
      return out;
    }
  } list_allocator;
  list_allocator.state = &state;
  List *root           = List::parse(text, list_allocator);
  if (root == NULL) {
    state.push_error("Couldn't parse");
    return;
  }
  root->dump_list_graph();

  IEvaluator::get_head()->state = &state;
  IEvaluator::get_head()->eval(root);
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  ASSERT_ALWAYS(argc == 2);
  parse_and_eval(stref_s(read_file_tmp(argv[1])));
  return 0;
}
