#ifndef SCRIPT_HPP
#define SCRIPT_HPP

#include "utils.hpp"
#include <cmath>

static inline bool parse_decimal_int(char const *str, i32 len, int32_t *result) {
  i32 final = 0;
  i32 pow   = 1;
  i32 sign  = 1;
  i32 i     = 0;
  // parsing in reverse order
  for (; i < len; ++i) {
    switch (str[len - 1 - i]) {
    case '0': break;
    case '1': final += 1 * pow; break;
    case '2': final += 2 * pow; break;
    case '3': final += 3 * pow; break;
    case '4': final += 4 * pow; break;
    case '5': final += 5 * pow; break;
    case '6': final += 6 * pow; break;
    case '7': final += 7 * pow; break;
    case '8': final += 8 * pow; break;
    case '9': final += 9 * pow; break;
    // it's ok to have '-'/'+' as the first char in a string
    case '-': {
      if (i == len - 1)
        sign = -1;
      else
        return false;
      break;
    }
    case '+': {
      if (i == len - 1)
        sign = 1;
      else
        return false;
      break;
    }
    default: return false;
    }
    pow *= 10;
  }
  *result = sign * final;
  return true;
}

template <typename T> static inline bool parse_float(char const *str, i32 len, T *result) {
  T   final = (T)0.0;
  i32 i     = 0;
  T   sign  = (T)1.0;
  if (str[0] == '-') {
    sign = (T)-1.0;
    i    = 1;
  }
  for (; i < len; ++i) {
    if (str[i] == '.') break;
    switch (str[i]) {
    case '\0': goto finish;
    case '0': final = final * (T)10.0; break;
    case '1': final = final * (T)10.0 + (T)1.0; break;
    case '2': final = final * (T)10.0 + (T)2.0; break;
    case '3': final = final * (T)10.0 + (T)3.0; break;
    case '4': final = final * (T)10.0 + (T)4.0; break;
    case '5': final = final * (T)10.0 + (T)5.0; break;
    case '6': final = final * (T)10.0 + (T)6.0; break;
    case '7': final = final * (T)10.0 + (T)7.0; break;
    case '8': final = final * (T)10.0 + (T)8.0; break;
    case '9': final = final * (T)10.0 + (T)9.0; break;
    default: return false;
    }
  }
  if (str[i] == '.') {
    i++;
    T pow = (T)1.0e-1;
    for (; i < len; ++i) {
      switch (str[i]) {
      case '\0': goto finish;
      case '0': break;
      case '1': final += (T)1.0 * pow; break;
      case '2': final += (T)2.0 * pow; break;
      case '3': final += (T)3.0 * pow; break;
      case '4': final += (T)4.0 * pow; break;
      case '5': final += (T)5.0 * pow; break;
      case '6': final += (T)6.0 * pow; break;
      case '7': final += (T)7.0 * pow; break;
      case '8': final += (T)8.0 * pow; break;
      case '9': final += (T)9.0 * pow; break;
      case 'f': goto finish;
      case 'F': goto finish;
      case 'e': goto parse_exponent;
      case 'E': goto parse_exponent;
      default: return false;
      }
      pow *= (T)1.0e-1;
    }

  parse_exponent:
    if (i < len) {
      if (str[i] != 'e' && str[i] != 'E') return false;
      i++;
      i32 exp = -1;
      i32 l   = 0;
      for (; l < (i32)len - i; l++)
        if (str[i + l] == 'f' || str[i + l] == 'F') break;
      if (!parse_decimal_int(str + i, l, &exp)) return false;
      final = final * std::pow((T)10.0, (T)exp);
    }
  }
finish:
  *result = sign * final;
  return true;
}

static double parse_float(char const *str) {
  double out;
  ASSERT_ALWAYS(parse_float(str, (i32)strlen(str), &out));
  return out;
}

static int __test_float_parsing = [] {
  double EPS = 1.0e-6;
  ASSERT_ALWAYS(parse_float("0.0") == 0.0);
  ASSERT_ALWAYS(parse_float("0000000000.0000000000") == 0.0);
  ASSERT_ALWAYS(parse_float("0.0000000000") == 0.0);
  ASSERT_ALWAYS(parse_float("-0.0000000000") == 0.0);
  ASSERT_ALWAYS(int(parse_float("1.12") * 100.0 * (1.0 + EPS)) == 112);
  ASSERT_ALWAYS(int(parse_float("125.125") * 1000.0 * (1.0 + EPS)) == 125125);
  ASSERT_ALWAYS(int(parse_float("-1.12e-1") * 1000.0 * (1.0 + EPS)) == -112);
  ASSERT_ALWAYS(int(parse_float("-5") * (1.0 + EPS)) == -5);
  ASSERT_ALWAYS(int(parse_float("-2") * (1.0 + EPS)) == -2);
  ASSERT_ALWAYS(int(parse_float("1.12e+1") * 10.0 * (1.0 + EPS)) == 112);
  ASSERT_ALWAYS(int(parse_float("1.12e+2") * 1.0 * (1.0 + EPS)) == 112);
  ASSERT_ALWAYS(int(parse_float("1.12e+2f") * 1.0 * (1.0 + EPS)) == 112);
  ASSERT_ALWAYS(int(parse_float("1.12e+2F") * 1.0 * (1.0 + EPS)) == 112);

  return 0;
}();

struct List {
  string_ref symbol = {};
  List *     child  = NULL;
  List *     next   = NULL;
  u32        id     = 0;
  bool       quoted = false;

  void toString(String_Builder &sb) {
    if (quoted) sb.putf("\"\"\"");
    sb.putstr(symbol);
    if (quoted) sb.putf("\"\"\"");
    if (child) {
      sb.putf(" (");
      child->toString(sb);
      sb.putf(")");
    }
    if (next) {
      next->toString(sb);
    }
  }
  string_ref get_symbol() {
    ASSERT_ALWAYS(nonempty());
    return symbol;
  }
  string_ref get_umbrella_string() {
    string_ref out = symbol;
    if (child != NULL) {
      string_ref th = child->get_umbrella_string();
      if (out.ptr == NULL) out.ptr = th.ptr;
      out.len += (size_t)(th.ptr - out.ptr) - out.len + th.len;
    }
    if (next != NULL) {
      string_ref th = next->get_umbrella_string();
      if (out.ptr == NULL) out.ptr = th.ptr;
      out.len += (size_t)(th.ptr - out.ptr) - out.len + th.len;
    }
    return out;
  }
  i32 parse_int() {
    i32 res;
    ASSERT_ALWAYS(parse_decimal_int(symbol.ptr, (i32)symbol.len, &res));
    return res;
  }
  f32 parse_float() {
    f32 res;
    ASSERT_ALWAYS(::parse_float(symbol.ptr, (i32)symbol.len, &res));
    return res;
  }
  bool nonempty() { return symbol.ptr != 0 && symbol.len != 0; }
  bool cmp_symbol(char const *str) {
    if (symbol.ptr == NULL) return false;
    return symbol == stref_s(str);
  }
  bool has_child(char const *name) { return child != NULL && child->cmp_symbol(name); }
  template <typename T> void match_children(char const *name, T on_match) {
    if (child != NULL) {
      if (child->cmp_symbol(name)) {
        on_match(child);
      }
      child->match_children(name, on_match);
    }
    if (next != NULL) {
      next->match_children(name, on_match);
    }
  }
  List *get(u32 i) {
    List *cur = this;
    while (i != 0) {
      if (cur == NULL) return NULL;
      cur = cur->next;
      i -= 1;
    }
    return cur;
  }

  int ATTR_USED dump(u32 indent = 0) const {
    ito(indent) fprintf(stdout, " ");
    if (symbol.ptr != NULL) {
      fprintf(stdout, "%.*s\n", (i32)symbol.len, symbol.ptr);
    } else {
      fprintf(stdout, "$\n");
    }
    if (child != NULL) {
      child->dump(indent + 2);
    }
    if (next != NULL) {
      next->dump(indent);
    }
    fflush(stdout);
    return 0;
  }
  void dump_list_graph() {
    List *root     = this;
    FILE *dotgraph = fopen("list.dot", "wb");
    fprintf(dotgraph, "digraph {\n");
    fprintf(dotgraph, "node [shape=record];\n");
    tl_alloc_tmp_enter();
    defer(tl_alloc_tmp_exit());
    List **stack        = (List **)tl_alloc_tmp(sizeof(List *) * (1 << 10));
    u32    stack_cursor = 0;
    List * cur          = root;
    u32    null_id      = 0xffffffffu;
    while (cur != NULL || stack_cursor != 0) {
      if (cur == NULL) {
        cur = stack[--stack_cursor];
      }
      ASSERT_ALWAYS(cur != NULL);
      if (cur->symbol.ptr != NULL) {
        ASSERT_ALWAYS(cur->symbol.len != 0);
        fprintf(dotgraph, "%i [label = \"%.*s\", shape = record];\n", cur->id, (int)cur->symbol.len,
                cur->symbol.ptr);
      } else {
        fprintf(dotgraph, "%i [label = \"$\", shape = record, color=red];\n", cur->id);
      }
      if (cur->next == NULL) {
        fprintf(dotgraph, "%i [label = \"nil\", shape = record, color=blue];\n", null_id);
        fprintf(dotgraph, "%i -> %i [label = \"next\"];\n", cur->id, null_id);
        null_id++;
      } else
        fprintf(dotgraph, "%i -> %i [label = \"next\"];\n", cur->id, cur->next->id);

      if (cur->child != NULL) {
        if (cur->next != NULL) stack[stack_cursor++] = cur->next;
        fprintf(dotgraph, "%i -> %i [label = \"child\"];\n", cur->id, cur->child->id);
        cur = cur->child;
      } else {
        cur = cur->next;
      }
    }
    fprintf(dotgraph, "}\n");
    fflush(dotgraph);
    fclose(dotgraph);
  }
  template <typename T>
  static List *parse(string_ref text, T allocator, char const **end_of_list = NULL) {
    List *root = allocator.alloc();
    List *cur  = root;
    TMP_STORAGE_SCOPE;
    List **stack        = (List **)tl_alloc_tmp(sizeof(List *) * (1 << 8));
    u32    stack_cursor = 0;
    enum class State : char {
      UNDEFINED = 0,
      SAW_QUOTE,
      SAW_LPAREN,
      SAW_RPAREN,
      SAW_PRINTABLE,
      SAW_SEPARATOR,
      SAW_SEMICOLON,
    };
    u32   i  = 0;
    u32   id = 1;
    State state_table[0x100];
    memset(state_table, 0, sizeof(state_table));
    for (u8 j = 0x20; j <= 0x7f; j++) state_table[j] = State::SAW_PRINTABLE;
    state_table[(u32)'(']  = State::SAW_LPAREN;
    state_table[(u32)')']  = State::SAW_RPAREN;
    state_table[(u32)'"']  = State::SAW_QUOTE;
    state_table[(u32)' ']  = State::SAW_SEPARATOR;
    state_table[(u32)'\n'] = State::SAW_SEPARATOR;
    state_table[(u32)'\t'] = State::SAW_SEPARATOR;
    state_table[(u32)'\r'] = State::SAW_SEPARATOR;
    state_table[(u32)';']  = State::SAW_SEMICOLON;

    auto next_item = [&]() {
      List *next = allocator.alloc();
      next->id   = id++;
      if (cur != NULL) cur->next = next;
      cur = next;
    };

    auto push_item = [&]() {
      List *new_head = allocator.alloc();
      new_head->id   = id++;
      if (cur != NULL) {
        stack[stack_cursor++] = cur;
        cur->child            = new_head;
      }
      cur = new_head;
    };

    auto pop_item = [&]() -> bool {
      if (stack_cursor == 0) {
        return false;
      }
      cur = stack[--stack_cursor];
      return true;
    };

    auto append_char = [&]() {
      if (cur->symbol.ptr == NULL) { // first character for that item
        cur->symbol.ptr = text.ptr + i;
      }
      cur->symbol.len++;
    };

    auto set_quoted = [&]() { cur->quoted = true; };

    auto cur_non_empty = [&]() { return cur != NULL && cur->symbol.len != 0; };
    auto cur_has_child = [&]() { return cur != NULL && cur->child != NULL; };

    i                = 0;
    State prev_state = State::UNDEFINED;
    while (i < text.len) {
      char  c     = text.ptr[i];
      State state = state_table[(u8)c];
      switch (state) {
      case State::UNDEFINED: {
        goto error_parsing;
      }
      case State::SAW_SEMICOLON: {
        i += 1;
        while (text.ptr[i] != '\n') {
          i += 1;
        }
        break;
      }
      case State::SAW_QUOTE: {
        if (cur_non_empty() || cur_has_child()) next_item();
        set_quoted();
        if (text.ptr[i + 1] == '"' && text.ptr[i + 2] == '"') {
          i += 3;
          while (text.ptr[i + 0] != '"' || //
                 text.ptr[i + 1] != '"' || //
                 text.ptr[i + 2] != '"') {
            append_char();
            i += 1;
          }
          i += 2;
        } else {
          i += 1;
          while (text.ptr[i] != '"') {
            append_char();
            i += 1;
          }
        }
        break;
      }
      case State::SAW_LPAREN: {
        if (cur_has_child() || cur_non_empty()) next_item();
        push_item();
        break;
      }
      case State::SAW_RPAREN: {
        if (pop_item() == false) goto exit_loop;
        break;
      }
      case State::SAW_SEPARATOR: {
        break;
      }
      case State::SAW_PRINTABLE: {
        if (cur_has_child()) next_item();
        if (cur_non_empty() && prev_state != State::SAW_PRINTABLE) next_item();
        append_char();
        break;
      }
      }
      prev_state = state;
      i += 1;
    }
  exit_loop:
    (void)0;
    if (end_of_list != NULL) {
      *end_of_list = text.ptr + i + 1;
    }
    return root;
  error_parsing:
    return NULL;
  }
};

struct Value {
  enum class Value_t : i32 { UNKNOWN = 0, I32, F32, SYMBOL, BINDING, LAMBDA, SCOPE, MODE, ANY };
  i32 type;
  i32 any_type;
  union {
    string_ref str;
    f32        f;
    i32        i;
    List *     list;
    void *     any;
  };
  void ATTR_USED dump() {
    fprintf(stdout, "Value; {\n");
    switch (type) {
    case (i32)Value_t::I32: {
      fprintf(stdout, "  i32: %i\n", i);
      break;
    }
    case (i32)Value_t::F32: {
      fprintf(stdout, "  f32: %f\n", f);
      break;
    }
    case (i32)Value_t::SYMBOL: {
      fprintf(stdout, "  sym: %.*s\n", STRF(str));
      break;
    }
    case (i32)Value_t::BINDING: {
      fprintf(stdout, "  bnd:\n");
      list->dump(4);
      break;
    }
    case (i32)Value_t::LAMBDA: {
      fprintf(stdout, "  lmb:\n");
      list->dump(4);
      break;
    }
    case (i32)Value_t::SCOPE: {
      fprintf(stdout, "  scp\n");
      break;
    }
    case (i32)Value_t::ANY: {
      fprintf(stdout, "  any\n");
      break;
    }
    case (i32)Value_t::MODE: {
      fprintf(stdout, "  mod\n");
      break;
    }
    default: UNIMPLEMENTED;
    }
    fprintf(stdout, "}\n");
    fflush(stdout);
  }
};

struct Symbol_Table {
  struct Symbol {
    string_ref name;
    Value *    val;
  };
  struct Symbol_Frame {
    using Table_t = Hash_Table<string_ref, Value *, Default_Allocator, 0x10, 0x10>;
    Table_t       table;
    Symbol_Frame *prev;
    void          init() {
      table.init();
      prev = NULL;
    }
    void release() {
      table.release();
      prev = NULL;
    }
  };
  Pool<Symbol_Frame> table_storage;
  Symbol_Frame *     head;
  Symbol_Frame *     tail;

  void init() {
    table_storage = Pool<Symbol_Frame>::create(0x400);
    head          = table_storage.alloc(1);
    head->init();
    tail = head;
  }
  void release() {
    Symbol_Frame *cur = tail;
    while (cur != NULL) {
      Symbol_Frame *prev = cur->prev;
      cur->release();
      cur = prev;
    }
    table_storage.release();
  }
  Value *lookup_value(string_ref name) {
    Symbol_Frame *cur = tail;
    while (cur != NULL) {
      if (Value **val = cur->table.get_or_null(name)) return *val;
      cur = cur->prev;
    }
    return NULL;
  }
  Value *lookup_value(string_ref name, void *scope) {
    Symbol_Frame *cur = (Symbol_Frame *)scope;
    while (cur != NULL) {
      if (Value **val = cur->table.get_or_null(name)) return *val;
      cur = cur->prev;
    }
    return NULL;
  }
  void *get_scope() { return (void *)tail; }
  void  set_scope(void *scope) { tail = (Symbol_Frame *)scope; }
  void  enter_scope() {
    Symbol_Frame *new_table = table_storage.alloc(1);
    new_table->init();
    new_table->prev = tail;
    tail            = new_table;
  }
  void exit_scope() {
    Symbol_Frame *new_tail = tail->prev;
    ASSERT_DEBUG(new_tail != NULL);
    tail->release();
    table_storage.pop();
    tail = new_tail;
  }
  void ATTR_USED dump() {
    Symbol_Frame *cur = tail;
    while (cur != NULL) {
      fprintf(stdout, "--------new-table\n");
      cur->table.iter([&](Symbol_Frame::Table_t::Pair_t const &item) {
        fprintf(stdout, "symbol(\"%.*s\"):\n", STRF(item.key));
        item.value->dump();
      });
      cur = cur->prev;
    }
  }
  void add_symbol(string_ref name, Value *val) { tail->table.insert(name, val); }
};

//////////////////
// Global state //
//////////////////
struct Evaluator_State {
  Pool<char>   string_storage;
  Pool<List>   list_storage;
  Pool<Value>  value_storage;
  Symbol_Table symbol_table;
  bool         eval_error = false;

  void init() {
    string_storage = Pool<char>::create((1 << 20));
    list_storage   = Pool<List>::create((1 << 20));
    value_storage  = Pool<Value>::create((1 << 20));
    symbol_table.init();
  }

  void release() {
    string_storage.release();
    list_storage.release();
    value_storage.release();
    symbol_table.release();
  }

  void reset() {
    string_storage.reset();
    list_storage.reset();
    value_storage.reset();
    symbol_table.release();
    symbol_table.init();
  }

  void enter_scope() {
    string_storage.enter_scope();
    list_storage.enter_scope();
    value_storage.enter_scope();
    symbol_table.enter_scope();
  }

  void exit_scope() {
    string_storage.exit_scope();
    list_storage.exit_scope();
    value_storage.exit_scope();
    symbol_table.exit_scope();
  }

  Value *alloc_value() { return value_storage.alloc_zero(1); }

  string_ref move_cstr(string_ref old) {
    char *     new_ptr = string_storage.put(old.ptr, old.len + 1);
    string_ref new_ref = string_ref{new_ptr, old.len};
    new_ptr[old.len]   = '\0';
    return new_ref;
  }
};

struct Match {
  Value *val;
  bool   match;
  Match(Value *val) : val(val), match(true) {}
  Match(Value *val, bool match) : val(val), match(match) {}
  Value *unwrap() {
    if (!match) {
      ASSERT_ALWAYS(false);
      return NULL;
    }
    return val;
  }
};

struct Tmp_List_Allocator {
  List *alloc() {
    List *out = (List *)tl_alloc_tmp(sizeof(List));
    memset(out, 0, sizeof(List));
    return out;
  }
  char *alloc(size_t size) { return (char *)tl_alloc_tmp(size); }
};

struct IEvaluator;
typedef IEvaluator *(*Evaluator_Creator_t)();

static inline void push_warning(char const *fmt, ...) {
  fprintf(stdout, "[WARNING] ");
  va_list args;
  va_start(args, fmt);
  vfprintf(stdout, fmt, args);
  va_end(args);
  fprintf(stdout, "\n");
  fflush(stdout);
}

static inline void push_error(char const *fmt, ...) {
  fprintf(stdout, "[ERROR] ");
  va_list args;
  va_start(args, fmt);
  vfprintf(stdout, fmt, args);
  va_end(args);
  fprintf(stdout, "\n");
  fflush(stdout);
}
struct IEvaluator {
  Evaluator_State *  state        = NULL;
  IEvaluator *       prev         = NULL;
  virtual Match      eval(List *) = 0;
  virtual void       release()    = 0;
  static IEvaluator *get_head();
  static void        add_mode(string_ref name, Evaluator_Creator_t creat);
  static void        set_head(IEvaluator *);
  static IEvaluator *create_mode(string_ref name);
  static Match       global_eval(List *l) { return get_head()->eval(l); }
  Value *            eval_unwrap(List *l) { return global_eval(l).unwrap(); }
  Value *            eval_args(List *arg) {
    List * cur  = arg;
    Value *last = NULL;
    while (cur != NULL) {
      Value *a = eval_unwrap(cur);
      if (a != NULL && a->type == (i32)Value::Value_t::BINDING) {
        a = eval_args(a->list);
      }
      cur  = cur->next;
      last = a;
    }
    return last;
  }
  template <typename V> void eval_args_and_collect(List *l, V &values) {
    List *cur = l;
    while (cur != NULL) {
      Value *a = eval_unwrap(cur);
      if (a != NULL && a->type == (i32)Value::Value_t::BINDING) {
        eval_args_and_collect(a->list, values);
      } else {
        values.push(a);
      }
      cur = cur->next;
    }
  }
  Value *     alloc_value() { return state->alloc_value(); }
  string_ref  move_cstr(string_ref old) { return state->move_cstr(old); }
  void        set_error() { state->eval_error = true; }
  bool        is_error() { return state->eval_error; }
  static void parse_and_eval(string_ref text) {
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
      push_error("Couldn't parse");
      return;
    }
    root->dump_list_graph();

    IEvaluator::get_head()->state = &state;
    IEvaluator::get_head()->eval(root);
  }
};

#define ASSERT_EVAL(x)                                                                             \
  do {                                                                                             \
    if (!(x)) {                                                                                    \
      set_error();                                                                                 \
      push_error(#x);                                                                              \
      abort();                                                                                     \
      return NULL;                                                                                 \
    }                                                                                              \
  } while (0)
#define CHECK_ERROR()                                                                              \
  do {                                                                                             \
    if (is_error()) {                                                                              \
      abort();                                                                                     \
      return NULL;                                                                                 \
    }                                                                                              \
  } while (0)

namespace expr {
enum class Type {
  UNKNOWN = 0,
  BINOP,
  CALL,
  VALUE,
  SYMBOL,
  SCOPE,
  DEFUN,
  DECL,
  IF,
};
enum class ValueType {
  UNKNOWN = 0,
  I8,
  U8,
  I16,
  U16,
  I32,
  U32,
  I64,
  U64,
  F32,
  F64,
  BOOL,
};
struct Value {
  ValueType type;
  union {
    double v_f64;
    float  v_f32;
    u32    v_u32;
    u64    v_u64;
    i32    v_i32;
    i64    v_i64;
    bool   v_bool;
  };
};
struct Expr {
  union {
    Expr *child;
    Expr *lhs;
    Expr *cond;
    Value value;
  };
  string_ref token;
  union {
    Expr *rhs;
    Expr *argv;
    Expr *then_scope;
  };
  union {
    Expr *body_scope;
    Expr *else_scope;
  };
  Type type;
  char lscope, rscope;

  void toString(String_Builder &sb) {
    if (type == Type::BINOP) {
      if (lhs) lhs->toString(sb);
      sb.putstr(token);
      if (rhs) rhs->toString(sb);
    } else if (type == Type::SCOPE) {
      sb.putf("%c", lscope);
      child->toString(sb);
      sb.putf("%c", rscope);
    } else if (type == Type::CALL) {
      sb.putstr(token);
      child->toString(sb);
    } else if (type == Type::SYMBOL) {
      sb.putstr(token);
    } else if (type == Type::VALUE) {
      sb.putstr(token);
    } else if (type == Type::DEFUN) {
      sb.putf("defun ");
      sb.putstr(token);
      argv->toString(sb);
      body_scope->toString(sb);
    } else {
      UNIMPLEMENTED;
    }
  }
};
static Expr *tmp_alloc_expr() {
  Expr *out = (Expr *)tl_alloc_tmp(sizeof(Expr));
  memset(out, 0, sizeof(Expr));
  return out;
}
static bool isLiteral(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static bool isPrintable(char c) { return c >= 0x20 && c <= 0x7F; }
static bool isNumeral(char c) { return c >= '0' && c <= '9'; }
static void skipSpaces(char const *&cursor) {
  while (*cursor == ' ' || *cursor == '\n' || *cursor == '\t' || *cursor == '\r') {
    cursor++;
  }
}
static char        single_char_ops[] = {'+', '-', '*', '/', ',', '>', '<', '^', ';', '=', '.'};
static char const *two_char_ops[]    = {"<=", ">=", "!=", "+=", "-=", "*=", "/="};
static int         precedence[0x100];
static int         prec_init = [] {
  ito(ARRAY_SIZE(precedence)) precedence[i] = -1;
  // semicolon is used to chain expressions
  precedence[(int)';'] = 0;
  precedence[(int)','] = 1;
  // precedence[(int)':'] = 2;
  precedence[(int)'='] = 3;
  precedence[(int)'<'] = 5;
  precedence[(int)'>'] = 5;
  precedence[(int)'+'] = 10;
  precedence[(int)'-'] = 10;
  precedence[(int)'*'] = 20;
  precedence[(int)'/'] = 20;
  precedence[(int)'^'] = 30;
  precedence[(int)'.'] = 50;
  return 0;
}();
static int getPrecedence(string_ref token) {
  if (token.len == 1) {
    ASSERT_DEBUG(precedence[(int)token.ptr[0]] >= 0);
    return precedence[(int)token.ptr[0]];
  } else if (token.len == 2) {
    if (token.eq("<=")    //
        || token.eq(">=") //
        || token.eq("!=") //
        || token.eq("+=") //
        || token.eq("-=") //
        || token.eq("*=") //
        || token.eq("/=") //
        || token.eq("^=") //
    )
      return 5;
  }
  UNIMPLEMENTED;
}
static bool isLogicOp(string_ref c) {
  return c.eq(">") || c.eq("<") || c.eq("=") || c.eq(">=") || c.eq("<=") || c.eq("!=");
}
static bool isArithmeticOp(string_ref c) {
  return c.ptr[0] == '+' || c.ptr[0] == '-' || c.ptr[0] == '*' || c.ptr[0] == '/' ||
         c.ptr[0] == '^';
}
static bool isOp(char c) {
  for (auto o : single_char_ops)
    if (c == o) return true;
  return false;
}
static bool isOp(string_ref token) {
  if (token.len == 1) {
    return isOp(token.ptr[0]);
  } else if (token.len == 2) {
    for (auto o : two_char_ops)
      if (token.eq(o)) return true;
  }
  return false;
}
static Expr *parse_expression(char const *&cursor);
static Expr *parse_inner(char const *&cursor, char a, char b) {
  char const *s = cursor;
  skipSpaces(s);
  if (*s == a) {
    s++;
    Expr *inner = parse_expression(s);
    skipSpaces(s);
    if (inner == NULL || *s != b) return NULL;
    s++;
    Expr *expr   = tmp_alloc_expr();
    expr->type   = Type::SCOPE;
    expr->lscope = a;
    expr->rscope = b;
    expr->child  = inner;
    cursor       = s;
    return expr;
  }
  return NULL;
}
static bool parse_symbol(char const *&cursor, string_ref &token) {
  char const *cur = cursor;
  skipSpaces(cur);
  string_ref cur_token{cur, 0};
  while (isLiteral(*cur)) {
    cur++;
    cur_token.len++;
  }
  while (isLiteral(*cur) || isNumeral(*cur)) {
    cur++;
    cur_token.len++;
  }
  if (cur_token.len > 0) {
    cursor = cur;
    token  = cur_token;
    return true;
  }
  return false;
}
static Expr *parse_expression(char const *&cursor) {
  skipSpaces(cursor);
  Expr *lhs = NULL;
  if (*cursor == '(') {
    lhs = parse_inner(cursor, '(', ')');
    if (lhs == NULL) return NULL;
  } else if (isLiteral(*cursor)) {
    string_ref token;
    if (!parse_symbol(cursor, token)) return NULL;
    if (token.eq("defun")) {
      string_ref name;
      if (!parse_symbol(cursor, name)) return NULL;
      Expr *argv = parse_inner(cursor, '(', ')');
      if (argv == NULL) return NULL;
      Expr *body = parse_inner(cursor, '{', '}');
      if (body == NULL) return NULL;
      Expr *def       = tmp_alloc_expr();
      def->type       = Type::DEFUN;
      def->token      = name;
      def->argv       = argv;
      def->body_scope = body;
      return def;
    }
    skipSpaces(cursor);
    /*if (*cursor == ':') {
       string_ref type;
       if (!parse_symbol(type, type)) return NULL;
       Expr *decl      = tmp_alloc_expr();
       def->type       = Type::DECL;
       def->token      = name;
       def->rhs        = ;
       def->body_scope = body;
       return def;
     }*/
    if (*cursor == '(') {
      Expr *inner = parse_inner(cursor, '(', ')');
      if (inner == NULL) return NULL;
      Expr *call  = tmp_alloc_expr();
      call->type  = Type::CALL;
      call->token = token;
      call->child = inner;
      lhs         = call;
    } else {
      lhs        = tmp_alloc_expr();
      lhs->type  = Type::SYMBOL;
      lhs->token = token;
    }
  } else if (isNumeral(*cursor)) {
    string_ref cur_token{cursor, 1};
    cursor++;
    bool symbol              = false;
    bool has_dot             = false;
    bool only_ones_and_zeros = true;
    while (isLiteral(*cursor) || isNumeral(*cursor) || *cursor == '.') {
      if (*cursor == '.' && symbol) return NULL;
      if (*cursor == '.') has_dot = true;
      if (*cursor != '0' && *cursor != '1') only_ones_and_zeros = false;
      if (isLiteral(*cursor)) {
        symbol = true;
      }
      cursor++;
      cur_token.len++;
    }
    if (symbol) {
      lhs        = tmp_alloc_expr();
      lhs->type  = Type::SYMBOL;
      lhs->token = cur_token;
    } else {
      if (!has_dot && cur_token.len > 1 && cur_token.len <= 4 &&
          only_ones_and_zeros) { // Parse as a swizzle e.g. 001 100 101.
        lhs        = tmp_alloc_expr();
        lhs->type  = Type::SYMBOL;
        lhs->token = cur_token;
      } else {
        double num;
        bool   suc = parse_float(cur_token.ptr, (i32)cur_token.len, &num);
        if (!suc) return NULL;
        lhs              = tmp_alloc_expr();
        lhs->type        = Type::VALUE;
        lhs->token       = cur_token;
        lhs->value.type  = ValueType::F64;
        lhs->value.v_f64 = num;
      }
    }
  }
  skipSpaces(cursor);
  if (isOp(*cursor) || isOp({cursor, 2})) {
    Expr *op  = tmp_alloc_expr();
    op->type  = Type::BINOP;
    op->token = string_ref{cursor, 1};
    op->lhs   = lhs;
    if (isOp({cursor, 2})) {
      op->token.len++;
      if (!isOp(op->token)) return NULL;
      cursor += 2;
    } else {
      cursor++;
    }
    Expr *rhs = parse_expression(cursor);
    op->rhs   = rhs;
    if (op->lhs && op->rhs && rhs->type == Type::BINOP) {
      if (getPrecedence(rhs->token) < getPrecedence(op->token)) {
        Expr *a0 = op->lhs;
        Expr *a1 = rhs->lhs;
        Expr *a2 = rhs->rhs;
        ASSERT_DEBUG(a0 && a1 && a0);
        SWAP(op->token, rhs->token);
        rhs->lhs = a0;
        rhs->rhs = a1;
        op->lhs  = rhs;
        op->rhs  = a2;
      }
    }
    return op;
  }
  return lhs;
}
static bool fold(Expr *expr, double &res) {
  if (expr->type == Type::BINOP) {
    double a, b;
    if (expr->lhs && !fold(expr->lhs, a) || !fold(expr->rhs, b)) return false;
    if (expr->token.len == 1) {
      switch (expr->token.ptr[0]) {
      case '+': res = a + b; return true;
      case '-': res = (expr->lhs ? a : 0.0) - b; return true;
      case '*': res = a * b; return true;
      case '/': res = a / b; return true;
      case '^': res = ::pow(a, b); return true;
      default: return false;
      }
    }
    return false;
  } else if (expr->type == Type::VALUE) {
    res = expr->value.v_f64;
    return true;
  } else if (expr->type == Type::SCOPE)
    return fold(expr->child, res);
  return false;
}
static void __test_fold(char const *text, double cmp) {
  TMP_STORAGE_SCOPE;
  char const *cursor = text;
  Expr *      expr   = parse_expression(cursor);
  double      res;
  ASSERT_ALWAYS(fold(expr, res));
  ASSERT_ALWAYS(res == cmp);
}
static void __test_to_string(char const *text, char const *cmp) {
  TMP_STORAGE_SCOPE;
  char const *   cursor = text;
  Expr *         expr   = parse_expression(cursor);
  String_Builder sb;
  sb.init();
  defer(sb.release());
  expr->toString(sb);
  ASSERT_ALWAYS(expr && sb.get_str() == stref_s(cmp));
}
static int __test = [] {
  //__test_fold("(3.2 - 1.2) * 2.0 / 4.0", 1.0);
  //__test_fold("(3.2 - 1.2) * 2.0 / 2.0 ^ 2.0", 1.0);
  //__test_fold("(6.0 - (3.0 * 2.0 - 1.0)) * 2.0 / 2.0", 1.0);
  //__test_fold("3.0 * 2.0 - 1.0", 5.0);
  //__test_fold("-(-3.0 * (-2.0) + 1.0)", -5.0);
  //__test_to_string("a = b; 1 = 2 - 1;", "a=b;1=2-1;");
  //__test_to_string("a : i32 = b; a += c - 1;", "a:i32=b;a+=c-1;");
  //__test_to_string("a : i32 = b; a += c - 1;", "a:i32=b;a+=c-1;");
  //__test_to_string("defun foo (a : float3, b : float3) { dot(a, b) }",
  //"defun foo(a:float3,b:float3){dot(a,b)}");
  return 0;
}();
} // namespace expr

#endif // SCRIPT_HPP

#ifdef SCRIPT_IMPL
#ifndef SCRIPT_IMPL_GUARD
#  define SCRIPT_IMPL_GUARD

#  define ALLOC_VAL() (Value *)alloc_value()
#  define CALL_EVAL(x)                                                                             \
    eval_unwrap(x);                                                                                \
    CHECK_ERROR()
#  define ASSERT_SMB(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::SYMBOL);
#  define ASSERT_I32(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::I32);
#  define ASSERT_F32(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::F32);
#  define ASSERT_ANY(x) ASSERT_EVAL(x != NULL && x->type == (i32)Value::Value_t::ANY);

#  define EVAL_SMB(res, id)                                                                        \
    Value *res = eval_unwrap(l->get(id));                                                          \
    ASSERT_SMB(res)
#  define EVAL_I32(res, id)                                                                        \
    Value *res = eval_unwrap(l->get(id));                                                          \
    ASSERT_I32(res)
#  define EVAL_F32(res, id)                                                                        \
    Value *res = eval_unwrap(l->get(id));                                                          \
    ASSERT_F32(res)
#  define EVAL_ANY(res, id)                                                                        \
    Value *res = eval_unwrap(l->get(id));                                                          \
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
      bool is_imm32  = !l->quoted && parse_decimal_int(l->symbol.ptr, (i32)l->symbol.len, &imm32);
      bool is_immf32 = !l->quoted && parse_float(l->symbol.ptr, (i32)l->symbol.len, &immf32);
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
#endif // SCRIPT_IMPL_GUARD
#endif