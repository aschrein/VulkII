// utils.hpp - public domain aschrein 2020
#ifndef UTILS_H
#define UTILS_H

//#define TRACY_ENABLE

#include <codecvt>
#include <cstdlib>
#include <locale>
#include <malloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <time.h>
#include <type_traits>

#ifdef TRACY_ENABLE
#  include "3rdparty/tracy/Tracy.hpp"
#else
#  define TracyIsConnected false
#endif

static inline double time() { return ((double)clock()) / CLOCKS_PER_SEC; }

struct Timer {
  double cur_time;
  double old_time;
  double dt;
  void   init() {
    cur_time = time();
    old_time = time();
    dt       = 0.0;
  }
  void update() {
    cur_time = time();
    dt       = cur_time - old_time;
    old_time = cur_time;
  }
  void release() {}
};

#define ASSERT_ALWAYS(x)                                                                           \
  do {                                                                                             \
    if (!(x)) {                                                                                    \
      fprintf(stderr, "%s:%i [FAIL] at %s\n", __FILE__, __LINE__, #x);                             \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define ASSERT_PANIC(x) ASSERT_ALWAYS(x)
#define NOTNULL(x) ASSERT_ALWAYS((x) != NULL)
#define ARRAY_SIZE(_ARR) ((int)(sizeof(_ARR) / sizeof(*_ARR)))

#ifndef NDEBUG
#  define DEBUG_BUILD
#endif

#ifdef DEBUG_BUILD
#  define ASSERT_DEBUG(x) ASSERT_ALWAYS(x)
#else
#  define ASSERT_DEBUG(x)
#endif

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;
using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;
using i8  = int8_t;
using i32 = int32_t;
using f32 = float;
using f64 = double;
#undef MIN
#undef MAX
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MIN3(x, y, z) MIN(x, MIN(y, z))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MAX3(x, y, z) MAX(x, MAX(y, z))
#define CLAMP(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define OFFSETOF(class, field) ((unsigned int)(size_t) & (((class *)0)->field))
#define MEMZERO(x) memset(&x, 0, sizeof(x))
#define ASSERT_ISPOD(x) static_assert(std::is_pod<x>::value, "")
#define ito(N) for (uint32_t i = 0; i < N; ++i)
#define jto(N) for (uint32_t j = 0; j < N; ++j)
#define uto(N) for (uint32_t u = 0; u < N; ++u)
#define kto(N) for (uint32_t k = 0; k < N; ++k)
#define xto(N) for (uint32_t x = 0; x < N; ++x)
#define yto(N) for (uint32_t y = 0; y < N; ++y)
#define zto(N) for (uint32_t z = 0; z < N; ++z)

#if __linux__
// UNIX headers
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#  define DLL_EXPORT __attribute__((visibility("default")))
#  define ATTR_USED __attribute__((used))
#elif WIN32
#  pragma warning(disable : 4996)
#  define NOMINMAX
#  include <Windows.h>
#  undef min
#  undef max
#  define DLL_EXPORT __declspec(dllexport)
#  define ATTR_USED
#  define _CRT_SECURE_NO_WARNINGS

#  include <cfenv>

#  if !defined(alloca)
#    if defined(__GLIBC__) || defined(__sun) || defined(__APPLE__) || defined(__NEWLIB__)
#      include <alloca.h> // alloca (glibc uses <alloca.h>. Note that Cygwin may have _WIN32 defined, so the order matters here)
#    elif defined(_WIN32)
#      include <malloc.h> // alloca
#      if !defined(alloca)
#        define alloca _alloca // for clang with MS Codegen
#      endif                   // alloca
#    elif defined(__GLIBC__) || defined(__sun)
#      include <alloca.h> // alloca
#    else
#      include <stdlib.h> // alloca
#    endif
#  endif // alloca

#  if (defined(_MSC_VER) && !defined(snprintf))
#    define snprintf _snprintf
#  endif //(defined(_MSC_VER) && !defined(snprintf))

inline static void nop() {
#  if defined(_WIN32)
  __noop();
#  else
  __asm__ __volatile__("nop");
#  endif
}
inline static uint64_t get_thread_hash() {
  auto id = std::this_thread::get_id();
  return std::hash<std::thread::id>()(id);
}
struct Spin_Lock {
  std::atomic<u32> rw_flag;
  void             lock() {
#  ifdef TRACY_ENABLE
    ZoneScopedS(16);
#  endif
    u32 expected = 0;
    while (!rw_flag.compare_exchange_strong(expected, 1)) {
      expected = 0;
      while (rw_flag.load() != 0) {
        ito(16) nop(); // yield
      }
    }
  }
  void unlock() { rw_flag.store(0); }
};

#  ifdef UTILS_RENDERDOC

#    include "3rdparty/renderdoc_app.h"

class RenderDoc_CTX {
  private:
  RENDERDOC_API_1_4_1 *renderdoc_api = NULL;
  bool                 dll_not_found = false;

  ~RenderDoc_CTX() {
    if (renderdoc_api) {
      renderdoc_api->Shutdown();
    }
  }
  static RenderDoc_CTX &get() {
    static RenderDoc_CTX ctx;
    return ctx;
  }
  void init() {
    if (dll_not_found) return;
    if (renderdoc_api == NULL) {
#    ifdef WIN32
      HMODULE mod = GetModuleHandle("renderdoc.dll");
      if (mod) {
        pRENDERDOC_GetAPI getApi = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        if (getApi) {
          if (getApi(eRENDERDOC_API_Version_1_4_1, (void **)&renderdoc_api) == 1) {

          } else {
            // fprintf(stderr, "[RenderDoc] failed to retieve API functions!\n");
          }
        } else {
          // fprintf(stderr, "[RenderDoc] GetAPI not found!\n");
        }
      } else {
        dll_not_found = true;
        // fprintf(stderr, "[RenderDoc] module not found!\n");
      }
#    endif
    }
  }
  void _start() {
    if (renderdoc_api && renderdoc_api->IsFrameCapturing()) return;
#    ifdef WIN32
    if (renderdoc_api == NULL) {
      if (dll_not_found) return;
      init();
    }
#    endif
    if (renderdoc_api) {
      renderdoc_api->StartFrameCapture(NULL, NULL);
    }
  }

  void _end() {
    if (renderdoc_api) {
      if (!renderdoc_api->IsFrameCapturing()) return;
      renderdoc_api->EndFrameCapture(NULL, NULL);
    }
  }

  public:
  static void start() { get()._start(); }
  static void end() { get()._end(); }
};

#  endif

static u32 enable_fpe() {
  u32 fe_value  = ~(    //
      _EM_INVALID |    //
      _EM_DENORMAL |   //
      _EM_ZERODIVIDE | //
      _EM_OVERFLOW |   //
      _EM_UNDERFLOW |  //
      //_EM_INEXACT |    //
      0 //
  );
  u32 mask      = _MCW_EM;
  u32 old_state = 0;
  _clearfp();
  errno_t result = _controlfp_s(&old_state, fe_value, mask);
  ASSERT_DEBUG(result == 0);
  return old_state;
}

static u32 disable_fpe() {
  u32 fe_value  = ~(0);
  u32 mask      = _MCW_EM;
  u32 old_state = 0;
  _clearfp();
  errno_t result = _controlfp_s(&old_state, fe_value, mask);
  ASSERT_DEBUG(result == 0);
  return old_state;
}

static void restore_fpe(u32 new_mask) {
  u32 mask     = _MCW_EM;
  u32 old_mask = 0;
  _clearfp();
  errno_t result = _controlfp_s(&old_mask, new_mask, mask);
  ASSERT_DEBUG(result == 0);
}
#else
#  define DLL_EXPORT
#  define ATTR_USED
#endif

template <typename T> T copy(T const &in) { return in; }

template <typename M, typename K> bool contains(M const &in, K const &key) {
  return in.find(key) != in.end();
}

template <typename M> bool sets_equal(M const &a, M const &b) {
  if (a.size() != b.size()) return false;
  for (auto const &item : a) {
    if (!contains(b, item)) return false;
  }
  return true;
}

template <typename M> M get_intersection(M const &a, M const &b) {
  M out;
  for (auto const &item : a) {
    if (contains(b, item)) out.insert(item);
  }
  return out;
}

template <typename T, typename F> bool any(T set, F f) {
  for (auto const &item : set)
    if (f(item)) return true;
  return false;
}

#define UNIMPLEMENTED_(s)                                                                          \
  do {                                                                                             \
    fprintf(stderr, "%s:%i UNIMPLEMENTED %s\n", __FILE__, __LINE__, s);                            \
    abort();                                                                                       \
  } while (0)
#define UNIMPLEMENTED UNIMPLEMENTED_("")
#define TRAP                                                                                       \
  do {                                                                                             \
    fprintf(stderr, "%s:%i TRAP\n", __FILE__, __LINE__);                                           \
    fflush(stderr);                                                                                \
    abort();                                                                                       \
  } while (0)
#define NOCOMMIT (void)0

template <typename F> struct __Defer__ {
  F f;
  __Defer__(F f) : f(f) {}
  ~__Defer__() { f(); }
};

template <typename F> __Defer__<F> defer_func(F f) { return __Defer__<F>(f); }

#define DEFER_1(x, y) x##y
#define DEFER_2(x, y) DEFER_1(x, y)
#define DEFER_3(x) DEFER_2(x, __COUNTER__)
#define defer(code) auto DEFER_3(_defer_) = defer_func([&]() { code; })
#define static_defer(code) auto DEFER_3(_defer_) = defer_func([]() { code; })

#define STRINGIFY(a) _STRINGIFY(a)
#define _STRINGIFY(a) #a

#define PERF_HIST_ADD(name, val)
#define PERF_ENTER(name)
#define PERF_EXIT(name)
#define OK_FALLTHROUGH (void)0;
#define TMP_STORAGE_SCOPE                                                                          \
  tl_alloc_tmp_enter();                                                                            \
  defer(tl_alloc_tmp_exit(););
#define SWAP(x, y)                                                                                 \
  do {                                                                                             \
    auto tmp = x;                                                                                  \
    x        = y;                                                                                  \
    y        = tmp;                                                                                \
  } while (0)

#if __linux__
static inline size_t get_page_size() { return sysconf(_SC_PAGE_SIZE); }
#elif WIN32
static inline size_t get_page_size() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
}
#else
static inline size_t get_page_size() { return 1 << 12; }
#endif

static inline size_t page_align_up(size_t n) {
  return (n + get_page_size() - 1) & (~(get_page_size() - 1));
}

static inline size_t page_align_down(size_t n) { return (n) & (~(get_page_size() - 1)); }

static inline size_t get_num_pages(size_t size) { return page_align_up(size) / get_page_size(); }

#if __linux__
static inline void protect_pages(void *ptr, size_t num_pages) {
  mprotect(ptr, num_pages * get_page_size(), PROT_NONE);
}
static inline void unprotect_pages(void *ptr, size_t num_pages, bool exec = false) {
  mprotect(ptr, num_pages * get_page_size(), PROT_WRITE | PROT_READ | (exec ? PROT_EXEC : 0));
}

static inline void unmap_pages(void *ptr, size_t num_pages) {
  int err = munmap(ptr, num_pages * get_page_size());
  ASSERT_ALWAYS(err == 0);
}

static inline void map_pages(void *ptr, size_t num_pages) {
  void *new_ptr =
      mmap(ptr, num_pages * get_page_size(), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  ASSERT_ALWAYS((size_t)new_ptr == (size_t)ptr);
}
#elif WIN32
// TODO
static inline void protect_pages(void *ptr, size_t num_pages) {}
static inline void unprotect_pages(void *ptr, size_t num_pages, bool exec = false) {}
static inline void unmap_pages(void *ptr, size_t num_pages) {}
static inline void map_pages(void *ptr, size_t num_pages) {}
#else
// Noops
static inline void protect_pages(void *ptr, size_t num_pages) {}
static inline void unprotect_pages(void *ptr, size_t num_pages, bool exec = false) {}
static inline void unmap_pages(void *ptr, size_t num_pages) {}
static inline void map_pages(void *ptr, size_t num_pages) {}
#endif

template <typename T, typename V> struct Pair {
  T    first;
  V    second;
  bool operator==(Pair const &that) const { return first == that.first && second == that.second; }
};

template <typename T, typename V> Pair<T, V> make_pair(T t, V v) { return {t, v}; }

template <typename T = uint8_t> struct Pool {
  uint8_t *ptr;
  size_t   cursor;
  size_t   capacity;
  size_t   mem_length;
  size_t   stack_capacity;
  size_t   stack_cursor;

  static Pool create(size_t capacity) {
    ASSERT_DEBUG(capacity > 0);
    Pool   out;
    size_t STACK_CAPACITY = 0x20 * sizeof(size_t);
    out.mem_length        = get_num_pages(STACK_CAPACITY + capacity * sizeof(T)) * get_page_size();
#if __linux__
    out.ptr = (uint8_t *)mmap(NULL, out.mem_length, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE,
                              -1, 0);
    ASSERT_ALWAYS(out.ptr != MAP_FAILED);
#else
    out.ptr = (uint8_t *)malloc(out.mem_length);
    NOTNULL(out.ptr);
#endif
    out.capacity       = capacity;
    out.cursor         = 0;
    out.stack_capacity = STACK_CAPACITY;
    out.stack_cursor   = 0;
    return out;
  }

  T *back() { return (T *)(this->ptr + this->stack_capacity + this->cursor * sizeof(T)); }

  void advance(size_t size) {
    this->cursor += size;
    ASSERT_DEBUG(this->cursor < this->capacity);
  }

  void release() {
#if __linux__
    if (this->ptr) munmap(this->ptr, mem_length);
#else
    if (this->ptr) free(this->ptr);
#endif
    memset(this, 0, sizeof(Pool));
  }

  void push(T const &v) {
    T *ptr = alloc(1);
    memcpy(ptr, &v, sizeof(T));
  }

  bool has_items() { return this->cursor > 0; }

  T *at(uint32_t i) { return (T *)(this->ptr + this->stack_capacity + i * sizeof(T)); }

  T *alloc(size_t size) {
    ASSERT_DEBUG(size != 0);
    T *ptr = (T *)(this->ptr + this->stack_capacity + this->cursor * sizeof(T));
    this->cursor += size;
    ASSERT_DEBUG(this->cursor < this->capacity);
    return ptr;
  }

  T *try_alloc(size_t size) {
    ASSERT_DEBUG(size != 0);
    if (this->cursor + size > this->capacity) return NULL;
    T *ptr = (T *)(this->ptr + this->stack_capacity + this->cursor * sizeof(T));
    this->cursor += size;
    ASSERT_DEBUG(this->cursor < this->capacity);
    return ptr;
  }

  T *alloc_zero(size_t size) {
    T *mem = alloc(size);
    memset(mem, 0, size * sizeof(T));
    return mem;
  }

  T *alloc_align(size_t size, size_t alignment) {
    T *ptr = alloc(size + alignment);
    ptr    = (T *)(((size_t)ptr + alignment - 1) & (~(alignment - 1)));
    return ptr;
  }

  T *alloc_page_aligned(size_t size) {
    ASSERT_DEBUG(size != 0);
    size           = page_align_up(size) + get_page_size();
    T *ptr         = (T *)(this->ptr + this->stack_capacity + this->cursor * sizeof(T));
    T *aligned_ptr = (T *)(void *)page_align_down((size_t)ptr + get_page_size());
    this->cursor += size;
    ASSERT_DEBUG(this->cursor < this->capacity);
    return aligned_ptr;
  }

  void enter_scope() {
    // Save the cursor to the stack
    size_t *top = (size_t *)(this->ptr + this->stack_cursor);
    *top        = this->cursor;
    // Increment stack cursor
    this->stack_cursor += sizeof(size_t);
    ASSERT_DEBUG(this->stack_cursor < this->stack_capacity);
  }

  void exit_scope() {
    // Decrement stack cursor
    ASSERT_DEBUG(this->stack_cursor >= sizeof(size_t));
    this->stack_cursor -= sizeof(size_t);
    // Restore the cursor from the stack
    size_t *top  = (size_t *)(this->ptr + this->stack_cursor);
    this->cursor = *top;
  }

  void reset() {
    this->cursor       = 0;
    this->stack_cursor = 0;
  }

  T *put(T const *old_ptr, size_t count) {
    T *new_ptr = alloc(count);
    memcpy(new_ptr, old_ptr, count * sizeof(T));
    return new_ptr;
  }
  void pop() {
    ASSERT_DEBUG(cursor > 0);
    cursor -= 1;
  }
  bool has_space(size_t size) { return cursor + size <= capacity; }
};

template <typename T = u8> using Temporary_Storage = Pool<T>;

/** Allocates 'size' bytes using thread local allocator
 */
void *                   tl_alloc(size_t size);
template <typename T> T *tl_alloc_typed(size_t size) { return (T *)tl_alloc(size * sizeof(T)); }
/** Reallocates deleting `ptr` as a result
 */
void *tl_realloc(void *ptr, size_t oldsize, size_t newsize);
void  tl_free(void *ptr);
/** Allocates 'size' bytes using thread local temporal storage
 */
void *tl_alloc_tmp(size_t size);
/** Record the current state of thread local temporal storage
 */
void tl_alloc_tmp_enter();
/** Restore the previous state of thread local temporal storage
 */
void tl_alloc_tmp_exit();

struct string_ref {
  const char *ptr;
  size_t      len;
  string_ref  substr(size_t offset, size_t new_len) { return string_ref{ptr + offset, new_len}; }
  bool        eq(char const *str) const {
    size_t slen = strlen(str);
    if (ptr == NULL || str == NULL) return false;
    return len != slen ? false : strncmp(ptr, str, len) == 0 ? true : false;
  }
};

static inline i32 str_match(char const *cur, char const *patt) {
  i32 i = 0;
  while (true) {
    if (cur[i] == '\0' || patt[i] == '\0') return i;
    if (cur[i] == patt[i]) {
      i++;
    } else {
      return -1;
    }
  }
}

static inline i32 str_find(char const *cur, size_t maxlen, char c) {
  size_t i = 0;
  while (true) {
    if (i == maxlen) return -1;
    if (cur[i] == '\0') return -1;
    if (cur[i] == c) {
      return (i32)i;
    }
    i++;
  }
}

// for printf
#define STRF(str) (i32) str.len, str.ptr

static inline bool operator==(string_ref a, string_ref b) {
  if (a.ptr == NULL || b.ptr == NULL) return false;
  return a.len != b.len ? false : strncmp(a.ptr, b.ptr, a.len) == 0 ? true : false;
}

static inline uint64_t hash_of(uint64_t u) {
  uint64_t v = u * 3935559000370003845 + 2691343689449507681;
  v ^= v >> 21;
  v ^= v << 37;
  v ^= v >> 4;
  v *= 4768777513237032717;
  v ^= v << 20;
  v ^= v >> 41;
  v ^= v << 5;
  return v;
}

template <typename T> static void swap(T &a, T &b) {
  T c = a;
  a   = b;
  b   = c;
}

static inline uint64_t hash_of(uint32_t u) { return hash_of((uint64_t)u); }
static inline uint64_t hash_of(int32_t u) { return hash_of((uint64_t)u); }
static inline uint64_t hash_of(uint16_t u) { return hash_of((uint64_t)u); }

template <typename T> static uint64_t hash_of(T *ptr) { return hash_of((size_t)ptr); }

static inline uint64_t hash_of(string_ref a) {
  if (a.len == 0) return 0;
  uint64_t hash = 5381;
  for (size_t i = 0; i < a.len; i++) {
    hash =
        //(hash << 6) + (hash << 16) - hash + a.ptr[i];
        ((hash << 5) + hash) + a.ptr[i];
  }
  return hash;
}

template <typename T, typename V> u64 hash_of(Pair<T, V> const &p) {
  return hash_of(hash_of(p.first)) ^ hash_of(p.second);
}

template <> inline u64 hash_of(Pair<u32, u32> const &item) {
  return hash_of((u64)item.first | ((u64)item.second << 32ull));
}

/** String view of a static string
 */
static inline string_ref stref_s(char const *static_string, bool include_null = true) {
  if (static_string == NULL || static_string[0] == '\0') return string_ref{NULL, 0};
  ASSERT_DEBUG(static_string != NULL);
  string_ref out;
  out.ptr = static_string;
  out.len = strlen(static_string);
  if (!include_null) out.len--;
  ASSERT_DEBUG(out.len != 0);
  return out;
}

/** String view of a temporal string
  Uses thread local temporal storage
  */
static inline string_ref stref_tmp_copy(string_ref a) {
  string_ref out;
  out.len = a.len;
  ASSERT_DEBUG(out.len != 0);
  char *ptr = (char *)tl_alloc_tmp(out.len);
  memcpy(ptr, a.ptr, out.len);
  out.ptr = (char const *)ptr;
  return out;
}

/** String view of a temporal string
  Uses thread local temporal storage
  */
static inline string_ref stref_tmp(char const *tmp_string) {
  ASSERT_DEBUG(tmp_string != NULL);
  string_ref out;
  out.len = strlen(tmp_string);
  ASSERT_DEBUG(out.len != 0);
  char *ptr = (char *)tl_alloc_tmp(out.len);
  memcpy(ptr, tmp_string, out.len);
  out.ptr = (char const *)ptr;

  return out;
}

static inline string_ref stref_concat_tmp(string_ref a, string_ref b) {
  string_ref out;
  out.len = a.len + b.len;
  ASSERT_DEBUG(out.len != 0);
  char *ptr = (char *)tl_alloc_tmp(out.len);
  memcpy(ptr, a.ptr, a.len);
  memcpy(ptr + a.len, b.ptr, b.len);
  out.ptr = (char const *)ptr;
  return out;
}

static inline char const *stref_to_tmp_cstr(string_ref a) {
  ASSERT_DEBUG(a.ptr != NULL);
  char *ptr = (char *)tl_alloc_tmp(a.len + 1);
  memcpy(ptr, a.ptr, a.len);
  ptr[a.len] = '\0';
  return ptr;
}

static inline int32_t stref_find(string_ref a, string_ref b, size_t start = 0) {
  for (size_t i = start; i < a.len; i++) {
    for (size_t j = 0; j < b.len && i + j < a.len; j++) {
      if (a.ptr[i + j] != b.ptr[j]) break;
      if (j == b.len - 1) return (i32)(i - j);
    }
  }
  return -1;
}

static inline int32_t stref_find_last(string_ref a, string_ref b, size_t start = 0) {
  int32_t last_pos = -1;
  int32_t cursor   = stref_find(a, b, start);
  while (cursor >= 0) {
    last_pos = cursor;
    if ((size_t)cursor + 1 < a.len) cursor = stref_find_last(a, b, (size_t)(cursor + 1));
  }
  return last_pos;
}

static inline int32_t stref_find_last(string_ref a, char b) {
  int32_t last_pos = -1;
  ito(a.len) {
    if (a.ptr[i] == b) last_pos = (i32)i;
  }
  return last_pos;
}

static inline string_ref stref_substring(string_ref a, string_ref pattern) {
  i32 f = stref_find(a, pattern);
  if (f < 0) return a;
  return string_ref{a.ptr, (size_t)f};
}

static inline u32 smallest_pot(u32 x) {
  if ((x & (x - 1)) == 0) return x;
  for (int i = 1; i < 32; i *= 2) {
    x = x | (x >> i);
  }
  x = x + 1;
  return x;
}

static inline string_ref get_dir(string_ref path) {
  if (path.ptr[path.len - 1] == '/') path.len -= 1;
  int32_t sep = stref_find_last(path, '/');
  return path.substr(0, sep);
}

template <int N> struct inline_string {
  char buf[N];
  void init(string_ref str) {
    size_t len = MIN(str.len, N);
    memcpy(buf, str.ptr, len);
    if (len < N) buf[len] = '\0';
  }
  void init(char const *str) {
    size_t len = MIN(strlen(str), N);
    memcpy(buf, str, len);
    if (len < N) buf[len] = '\0';
  }
  string_ref ref() const { return string_ref{&buf[0], len()}; }
  u32        len() const {
    ito(N) {
      if (buf[i] == '\0') return i;
    }
    return N;
  }
};

template <int N>
static inline bool operator==(inline_string<N> const &a, inline_string<N> const &b) {
  if (a.len() != b.len()) return false;
  return strncmp(a.buf, b.buf, a.len()) == 0 ? true : false;
}

template <int N>
static inline bool operator!=(inline_string<N> const &a, inline_string<N> const &b) {
  return !(a == b);
}

struct string {
  char * ptr;
  size_t len;
  bool   nonempty() const { return ptr != NULL; }
  string copy() const {
    string out;
    out.ptr = (char *)tl_alloc(len);
    out.len = len;
    memcpy(out.ptr, ptr, len);
    return out;
  }
  void release() {
    if (ptr != NULL) tl_free(ptr);
    memset(this, 0, sizeof(string));
  }
  string_ref ref() const { return string_ref{ptr, len}; }
};

static inline bool operator==(string a, string b) {
  if (a.ptr == NULL || b.ptr == NULL) return false;
  return a.len != b.len ? false : strncmp(a.ptr, b.ptr, a.len) == 0 ? true : false;
}

template <int N> static inline uint64_t hash_of(inline_string<N> a) { return hash_of(a.ref()); }

static inline uint64_t hash_of(string a) { return hash_of(a.ref()); }

static inline string make_string(char const *static_string) {
  if (static_string == NULL || static_string[0] == '\0') return string{NULL, 0};
  ASSERT_DEBUG(static_string != NULL);
  string out;
  out.len = strlen(static_string);
  out.ptr = (char *)tl_alloc(out.len);
  memcpy(out.ptr, static_string, out.len);
  return out;
}

static inline string make_string(string_ref ref) {
  if (ref.len == 0) return string{NULL, 0};
  string out;
  out.len = ref.len;
  out.ptr = (char *)tl_alloc(out.len);
  memcpy(out.ptr, ref.ptr, out.len);
  return out;
}

struct String_Builder {
  Pool<char> tmp_buf;
  void       init() { tmp_buf = Pool<char>::create(1 << 20); }
  void       release() { tmp_buf.release(); }
  void       reset() { tmp_buf.reset(); }
  string_ref get_str() { return string_ref{(char const *)tmp_buf.at(0), tmp_buf.cursor}; }
  void       putstr(string_ref str) { putf("%.*s", STRF(str)); }
  i32        putf(char const *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    i32 len = vsprintf(tmp_buf.back(), fmt, args);
    va_end(args);
    ASSERT_ALWAYS(len > 0);
    tmp_buf.advance(len);
    return len;
  }
  void put_char(char c) { tmp_buf.put(&c, 1); }
};

static inline string_ref tmp_format(char const *fmt, ...) {
  static thread_local char tmp_buf[0x1000];
  // char *  buf = (char *)tl_alloc_tmp(strlen(fmt) * 2);
  va_list args;
  va_start(args, fmt);
  vsnprintf(tmp_buf, sizeof(tmp_buf), fmt, args);
  va_end(args);
  return stref_tmp(tmp_buf);
}

#if __linux__
static inline void make_dir_recursive(string_ref path) {
  TMP_STORAGE_SCOPE;
  if (path.ptr[path.len - 1] == '/') path.len -= 1;
  int32_t sep = stref_find_last(path, stref_s("/"));
  if (sep >= 0) {
    make_dir_recursive(path.substr(0, sep));
  }
  mkdir(stref_to_tmp_cstr(path), 0777);
}
#endif

static inline WCHAR *towstr_tmp(string_ref str) {
  std::wstring wide     = std::wstring(str.ptr, str.ptr + str.len);
  size_t       wide_len = sizeof(wchar_t) * wide.size();
  WCHAR const *src      = wide.c_str();
  WCHAR *      tmp      = (WCHAR *)tl_alloc_tmp(wide_len + sizeof(wchar_t));
  memcpy(tmp, wide.c_str(), wide_len);
  tmp[wide.size()] = L'\0';
  return tmp;
}

static inline void dump_file(char const *path, void const *data, size_t size) {
  FILE *file = fopen(path, "wb");
  ASSERT_ALWAYS(file);
  fwrite(data, 1, size, file);
  fclose(file);
}

static inline char *read_file_tmp(char const *filename) {
  FILE *text_file = fopen(filename, "rb");
  if (text_file == NULL) return NULL;
  fseek(text_file, 0, SEEK_END);
  long fsize = ftell(text_file);
  fseek(text_file, 0, SEEK_SET);
  size_t size = (size_t)fsize;
  char * data = (char *)tl_alloc_tmp((size_t)fsize + 1);
  fread(data, 1, (size_t)fsize, text_file);
  data[size] = '\0';
  fclose(text_file);
  return data;
}

static inline string_ref read_file_tmp_stref(char const *filename) {
  FILE *text_file = fopen(filename, "rb");
  if (text_file == NULL) return string_ref{NULL, 0};
  fseek(text_file, 0, SEEK_END);
  long fsize = ftell(text_file);
  fseek(text_file, 0, SEEK_SET);
  size_t size = (size_t)fsize;
  char * data = (char *)tl_alloc_tmp((size_t)fsize + 1);
  fread(data, 1, (size_t)fsize, text_file);
  data[size] = '\0';
  fclose(text_file);
  return string_ref{data, (size_t)fsize};
}

// Saves RGBA32_FLOAT. The extension must be .pfm. 'Xn View' (https://www.xnview.com/) can parse
// such files.
static inline void ATTR_USED write_image_rgba32_float_pfm(const char *file_name, void *data,
                                                          uint32_t width, u32 pitch,
                                                          uint32_t height, bool flip_y = false) {
  FILE *file = fopen(file_name, "wb");
  ASSERT_ALWAYS(file);
  fprintf(file, "PF\n%d %d\n%lf\n", width, height, -1.0f);
  ito(height) {
    jto(width) {
      float *src =
          &((float *)data)[pitch / sizeof(float) * (flip_y ? (height - 1 - i) : i) + j * 4];
      fwrite((void *)(src + 0), 1, 4, file);
      fwrite((void *)(src + 1), 1, 4, file);
      fwrite((void *)(src + 2), 1, 4, file);
    }
  }
  fclose(file);
}

static inline void ATTR_USED write_image_2d_i32_ppm(const char *file_name, void *data,
                                                    uint32_t pitch, uint32_t width,
                                                    uint32_t height) {
  FILE *file = fopen(file_name, "wb");
  ASSERT_ALWAYS(file);
  fprintf(file, "P6\n");
  fprintf(file, "%d %d\n", width, height);
  fprintf(file, "255\n");
  ito(height) {
    jto(width) {
      uint32_t pixel = *(uint32_t *)(void *)(((uint8_t *)data) + i * pitch + j * 4);
      uint8_t  r     = ((pixel >> 0) & 0xff);
      uint8_t  g     = ((pixel >> 8) & 0xff);
      uint8_t  b     = ((pixel >> 16) & 0xff);
      uint8_t  a     = ((pixel >> 24) & 0xff);
      if (a == 0) {
        r = ((i & 1) ^ (j & 1)) * 127;
        g = ((i & 1) ^ (j & 1)) * 127;
        b = ((i & 1) ^ (j & 1)) * 127;
      }
      fputc(r, file);
      fputc(g, file);
      fputc(b, file);
    }
  }
  fclose(file);
}

static inline void ATTR_USED write_image_2d_i24_ppm(const char *file_name, void *data,
                                                    uint32_t pitch, uint32_t width,
                                                    uint32_t height) {
  FILE *file = fopen(file_name, "wb");
  ASSERT_ALWAYS(file);
  fprintf(file, "P6\n");
  fprintf(file, "%d %d\n", width, height);
  fprintf(file, "255\n");
  ito(height) {
    jto(width) {
      uint8_t r = *(uint8_t *)(void *)(((uint8_t *)data) + i * pitch + j * 3 + 0);
      uint8_t g = *(uint8_t *)(void *)(((uint8_t *)data) + i * pitch + j * 3 + 1);
      uint8_t b = *(uint8_t *)(void *)(((uint8_t *)data) + i * pitch + j * 3 + 2);
      fputc(r, file);
      fputc(g, file);
      fputc(b, file);
    }
  }
  fclose(file);
}

static inline void ATTR_USED write_image_2d_i8_ppm(const char *file_name, void *data,
                                                   uint32_t pitch, uint32_t width,
                                                   uint32_t height) {
  FILE *file = fopen(file_name, "wb");
  ASSERT_ALWAYS(file);
  fprintf(file, "P6\n");
  fprintf(file, "%d %d\n", width, height);
  fprintf(file, "255\n");
  ito(height) {
    jto(width) {
      uint8_t r = *(uint8_t *)(void *)(((uint8_t *)data) + i * pitch + j);
      fputc(r, file);
      fputc(r, file);
      fputc(r, file);
    }
  }
  fclose(file);
}

// struct Allocator {
//  virtual void *    alloc(size_t)                                     = 0;
//  virtual void *    realloc(void *, size_t old_size, size_t new_size) = 0;
//  virtual void      free(void *)                                      = 0;
//  static Allocator *get_default() {
//    struct _Allocator : public Allocator {
//      virtual void *alloc(size_t size) override { return tl_alloc(size); }
//      virtual void *realloc(void *ptr, size_t old_size, size_t new_size) override {
//        return tl_realloc(ptr, old_size, new_size);
//      }
//      virtual void free(void *ptr) override { tl_free(ptr); }
//    };
//    static _Allocator alloc;
//    return &alloc;
//  }
//};

struct Default_Allocator {
  static void *alloc(size_t size) { return tl_alloc(size); }
  static void *realloc(void *ptr, size_t old_size, size_t new_size) {
    return tl_realloc(ptr, old_size, new_size);
  }
  static void free(void *ptr) { tl_free(ptr); }
};

//#define SWAP(a, b) do {auto tmp = a; a = b; b = tmp; } while (0)

template <typename T, typename F> void quicky_sort(T *arr, u32 size, F cmp) {
  if (size == 0 || size == 1) return;
  if (size == 2) {
    bool cmp1 = cmp(arr[0], arr[1]);
    if (cmp1) return;
    SWAP(arr[0], arr[1]);
    return;
  }
  u32 i     = 0;
  u32 j     = size - 1;
  T   pivot = arr[(j + i) / 2];
  while (true) {
    while (cmp(arr[i], pivot)) {
      i++;
      ASSERT_DEBUG(i <= j);
    }
    while (cmp(pivot, arr[j])) {
      j--;
      ASSERT_DEBUG(i <= j);
    }
    if (i >= j) break;
    SWAP(arr[i], arr[j]);
    i++;
    j--;
  }
  // tail call optimization
  u32 partition = j + 1;
  u32 d1        = partition;
  u32 d2        = size - partition;
  if (d1 < d2) {
    quicky_sort(arr + partition, size - partition, cmp);
    quicky_sort(arr, partition, cmp);
  } else {
    quicky_sort(arr, partition, cmp);
    quicky_sort(arr + partition, size - partition, cmp);
  }
}

struct Tmp_Allocator {
  static void *alloc(size_t size) { return tl_alloc_tmp(size); }
  static void *realloc(void *ptr, size_t oldsize, size_t newsize) {
    if (oldsize == newsize) return ptr;
    if (newsize < oldsize) {
      return ptr;
    } else {
      void *new_ptr = NULL;
      new_ptr       = tl_alloc_tmp(newsize);
      memcpy(new_ptr, ptr, oldsize);
      return new_ptr;
    }
  }
  static void free(void *) {
    // lol
  }
};

template <typename T, unsigned int N> struct InlineArray {
  T            elems[N];
  unsigned int size;
  T &          operator[](u32 i) { return elems[i]; }
  T const &    operator[](u32 i) const { return elems[i]; }
  void         push(T const &a) {
    ASSERT_DEBUG(size < N);
    elems[size++] = a;
  }
  bool contains(T elem) {
    ito(size) if (elems[i] == elem) return true;
    return false;
  }
  void init() { memset(this, 0, sizeof(*this)); }
  void memzero() { memset(elems, 0, sizeof(elems)); }
  void release() { size = 0; }
  T    pop() {
    ASSERT_DEBUG(size > 0);
    return elems[--size];
  }
  void reset() { release(); }
  void resize(size_t new_size) {
    ASSERT_DEBUG(new_size <= N);
    size = new_size;
  }
  bool isfull() const { return size == N; }
  void remove(T val) {
    ito(size) {
      if (elems[i] == val) {
        for (u32 j = i; j < size - 1; j++) {
          elems[j] = elems[j + 1];
        }
        i--;
        size--;
      }
    }
  }
};

template <typename T, size_t grow_k = 0x100,
          typename Allcator_t = Default_Allocator> //
struct Array {
  T *    ptr;
  size_t size;
  size_t capacity;
  Array  clone() const {
    Array out;
    out.init(ptr, size);
    return out;
  }
  void init(uint32_t capacity = 0) {
    if (capacity != 0)
      ptr = (T *)Allcator_t::alloc(sizeof(T) * capacity);
    else
      ptr = NULL;
    size           = 0;
    this->capacity = capacity;
  }
  void init(T const *data, uint32_t len) {
    ASSERT_DEBUG(len != 0 && data != NULL);
    ptr            = (T *)Allcator_t::alloc(sizeof(T) * len);
    size           = len;
    this->capacity = len;
    memcpy(ptr, data, len * sizeof(T));
  }
  bool contains(T const &item) {
    ito(size) if (ptr[i] == item) return true;
    return false;
  }
  void replace(T const &item, T const &with) { ito(size) if (ptr[i] == item) ptr[i] = with; }
  u32  get_size() { return this->size; }
  u32  has_items() { return get_size() != 0; }
  void release() {
    if (ptr != NULL) {
      Allcator_t::free(ptr);
    }
    memset(this, 0, sizeof(*this));
  }
  void resize(size_t new_size) {
    if (new_size > capacity) {
      uint64_t new_capacity = new_size;
      ptr      = (T *)Allcator_t::realloc(ptr, sizeof(T) * capacity, sizeof(T) * new_capacity);
      capacity = new_capacity;
    }
    ASSERT_DEBUG(ptr != NULL);
    size = new_size;
  }
  void reset() { size = 0; }
  void memzero() {
    if (capacity > 0) {
      memset(ptr, 0, sizeof(T) * capacity);
    }
  }
  void reserve(size_t cnt) {
    size_t new_capacity = size + cnt;
    if (new_capacity > capacity) {
      ptr      = (T *)Allcator_t::realloc(ptr, sizeof(T) * capacity, sizeof(T) * new_capacity);
      capacity = new_capacity;
    }
  }
  T *alloc(size_t num) {
    size_t old_cursor = size;
    resize(size + num);
    return ptr + old_cursor;
  }
  void push(T elem) {
    if (size + 1 > capacity) {
      size_t new_capacity = capacity + grow_k;
      ptr      = (T *)Allcator_t::realloc(ptr, sizeof(T) * capacity, sizeof(T) * new_capacity);
      capacity = new_capacity;
    }
    ASSERT_DEBUG(capacity >= size + 1);
    ASSERT_DEBUG(ptr != NULL);
    memcpy(ptr + size, &elem, sizeof(T));
    size += 1;
  }

  T &back() {
    ASSERT_DEBUG(size != 0);
    return ptr[size - 1];
  }

  void insert(size_t index, T item) {
    if (size == 0) {
      ASSERT_DEBUG(index == 0);
      push(item);
      return;
    }
    ASSERT_DEBUG(index < size);
    push({});
    if (size == 1) return;
    for (i32 i = size - 2; i >= 0; i--) {
      ptr[i + 1] = ptr[i];
    }
    ptr[index] = item;
  }

  T pop() {
    ASSERT_DEBUG(size != 0);
    ASSERT_DEBUG(ptr != NULL);
    T elem = ptr[size - 1];
    if (size + grow_k < capacity) {
      uint64_t new_capacity = capacity - grow_k;
      ptr      = (T *)Allcator_t::realloc(ptr, sizeof(T) * capacity, sizeof(T) * new_capacity);
      capacity = new_capacity;
    }
    ASSERT_DEBUG(size != 0);
    size -= 1;
    if (size == 0) {
      Allcator_t::free(ptr);
      ptr      = NULL;
      capacity = 0;
    }
    return elem;
  }
  T &operator[](size_t i) {
    ASSERT_DEBUG(i < size);
    ASSERT_DEBUG(ptr != NULL);
    return ptr[i];
  }
  T const &operator[](size_t i) const {
    ASSERT_DEBUG(i < size);
    ASSERT_DEBUG(ptr != NULL);
    return ptr[i];
  }
  T *      at(size_t i) { return ptr + i; }
  T const *at(size_t i) const { return ptr + i; }
};

template <typename T, size_t grow_k = 0x10,
          typename Allcator_t = Default_Allocator> //
struct AutoArray : public Array<T, grow_k, Allcator_t> {
  AutoArray() { Array<T, grow_k, Allcator_t>::init(); }
  AutoArray(AutoArray const &) = delete;
  AutoArray operator=(AutoArray const &) = delete;
  AutoArray operator=(AutoArray &&) = delete;
  AutoArray(AutoArray &&)           = delete;
  void cloneFrom(AutoArray const &a) { Array<T, grow_k, Allcator_t>::init(a.ptr, a.size); }
  ~AutoArray() { Array<T, grow_k, Allcator_t>::release(); }
};

template <typename T, u32 N, typename Allcator_t = Default_Allocator> //
struct SmallArray {
  T                           local[N];
  size_t                      size;
  Array<T, N * 3, Allcator_t> array;
  void                        init() {
    memset(this, 0, sizeof(*this));
    array.init();
  }
  void release() {
    array.release();
    memset(this, 0, sizeof(*this));
  }
  T &operator[](size_t i) {
    if (i < N)
      return local[i];
    else
      return array[i - N];
  }
  void push(T const &val) {
    if (size < N) {
      local[size++] = val;
    } else {
      array.push(val);
      size++;
    }
  }
  bool has(T elem) {
    ito(size) {
      if ((*this)[i] == elem) return true;
    }
    return false;
  }
};

template <typename K, typename Allcator_t = Default_Allocator, size_t grow_k = 0x10,
          size_t MAX_ATTEMPTS = 0x10>
struct Hash_Set {
  struct Hash_Pair {
    K        key;
    uint64_t hash;
  };
  using Array_t = Array<Hash_Pair, grow_k, Allcator_t>;
  Array_t arr;
  size_t  item_count;
  void    release() {
    arr.release();
    item_count = 0;
  }
  void init() {
    arr.init();
    item_count = 0;
  }
  void reset() {
    arr.memzero();
    item_count = 0;
  }
  void reserve(size_t num) { arr.resize(arr.size + num); }
  i32  find(K key) {
    if (item_count == 0) return -1;
    uint64_t key_hash = hash_of(key);
    uint64_t hash     = key_hash;
    ASSERT_DEBUG(hash != 0);
    uint64_t size = arr.capacity;
    if (size == 0) return -1;
    uint32_t attempt_id = 0;
    for (; attempt_id < MAX_ATTEMPTS; ++attempt_id) {
      uint64_t id = (hash) % size;
      /*if (arr.ptr[id].hash == 0)
        return -1;*/
      if (arr.ptr[id].hash == key_hash && arr.ptr[id].key == key) {
        return (i32)id;
      }
      hash = hash_of(hash);
    }
    return -1;
  }

  bool try_insert(K key) {
    uint64_t key_hash = hash_of(key);
    uint64_t hash     = key_hash;
    ASSERT_DEBUG(hash != 0);
    uint64_t size = arr.capacity;
    if (size == 0) {
      arr.resize(grow_k);
      arr.memzero();
      size = arr.capacity;
    }
    Hash_Pair pair;
    pair.key  = key;
    pair.hash = key_hash;
    for (uint32_t attempt_id = 0; attempt_id < MAX_ATTEMPTS; ++attempt_id) {
      uint64_t id = (hash) % size;
      if (arr.ptr[id].hash == 0) { // Empty slot
        arr.ptr[id] = pair;
        item_count += 1;
        return true;
      } else if (arr.ptr[id].hash == key_hash && arr.ptr[id].key == key) { // Override
        arr.ptr[id] = pair;
        return true;
      } else { // collision
        (void)0;
      }

      hash = hash_of(hash);
    }
    return false;
  }

  bool try_resize(size_t new_size) {
    ASSERT_DEBUG(new_size > 0);
    Array_t old_arr        = arr;
    size_t  old_item_count = item_count;
    {
      Array_t new_arr;
      new_arr.init();
      ASSERT_DEBUG(new_size > 0);
      new_arr.resize(new_size);
      new_arr.memzero();
      arr        = new_arr;
      item_count = 0;
    }
    uint32_t i = 0;
    for (; i < old_arr.capacity; ++i) {
      Hash_Pair pair = old_arr.ptr[i];
      if (pair.hash != 0) {
        bool suc = try_insert(pair.key);
        if (!suc) {
          arr.release();
          arr        = old_arr;
          item_count = old_item_count;
          return false;
        }
      }
    }
    old_arr.release();
    return true;
  }

  void remove(K key) {
    if (item_count == 0) return;
    while (true) {
      i32 id = find(key);
      if (id > -1) {
        ASSERT_DEBUG(item_count > 0);
        arr.ptr[id].hash = 0u;
        item_count -= 1;
        if (item_count == 0) {
          arr.release();
        } else if (arr.size + grow_k < arr.capacity) {
          try_resize(arr.capacity - grow_k);
        }

      } else {
        break;
      }
    }
  }

  bool insert(K key) {
    u32  iters = 0x10;
    bool suc   = false;
    while (!(suc = try_insert(key))) {
      u32    resize_iters = 6;
      size_t new_size     = arr.capacity + grow_k;
      bool   resize_suc   = false;
      size_t grow_rate    = grow_k << 1;
      while (!(resize_suc = try_resize(new_size))) {
        if (resize_iters == 0) break;
        new_size += grow_rate;
        grow_rate = grow_rate << 1;
        resize_iters -= 1;
      }
      (void)resize_suc;
      ASSERT_DEBUG(resize_suc == true);
      if (iters == 0) break;
      iters -= 1;
    }
    ASSERT_DEBUG(suc == true);
    return suc;
  }

  bool contains(K key) { return find(key) != -1; }
};

template <typename K, typename V> struct Map_Pair {
  K    key;
  V    value;
  bool operator==(Map_Pair const &that) const { return this->key == that.key; }
};

template <typename K, typename V> u64 hash_of(Map_Pair<K, V> const &item) {
  return hash_of(item.key);
}

template <typename K, typename V, typename Allcator_t = Default_Allocator, size_t grow_k = 0x10,
          size_t MAX_ATTEMPTS = 0x20>
struct Hash_Table {
  using Pair_t = Map_Pair<K, V>;
  Hash_Set<Map_Pair<K, V>, Allcator_t, grow_k, MAX_ATTEMPTS> set;
  void                                                       release() { set.release(); }
  void                                                       init() { set.init(); }

  i32 find(K key) { return set.find(Map_Pair<K, V>{key, {}}); }

  void reserve(size_t size) { set.arr.resize(size); }

  V &get(K key) {
    i32 id = set.find(Map_Pair<K, V>{key, {}});
    ASSERT_DEBUG(id >= 0);
    return set.arr[id].key.value;
  }

  V &get_ref(K key) {
    i32 id = set.find(Map_Pair<K, V>{key, {}});
    ASSERT_DEBUG(id >= 0);
    return set.arr[id].key.value;
  }

  V *get_or_null(K key) {
    if (set.item_count == 0) return 0;
    i32 id = set.find(Map_Pair<K, V>{key, {}});
    if (id < 0) return 0;
    return &set.arr[id].key.value;
  }

  void reset() { set.reset(); }

  void remove(K key) { return set.remove(Map_Pair<K, V>{key, {}}); }

  bool insert(K key, V value) { return set.insert(Map_Pair<K, V>{key, value}); }

  bool contains(K key) { return set.contains(Map_Pair<K, V>{key, {}}); }

  template <typename F> void iter(F f) {
    ito(set.arr.size) {
      auto &item = set.arr[i];
      if (item.hash != 0) {
        f(item.key);
      }
    }
  }
  template <typename F> void iter_values(F f) {
    ito(set.arr.size) {
      auto &item = set.arr[i];
      if (item.hash != 0) {
        f(item.key.value);
      }
    }
  }
  template <typename F> void iter_pairs(F f) {
    ito(set.arr.size) {
      auto &item = set.arr[i];
      if (item.hash != 0) {
        f(item.key.key, item.key.value);
      }
    }
  }

  Hash_Table clone() const {
    Hash_Table out;
    out.set.arr        = set.arr.clone();
    out.set.item_count = set.item_count;
    return out;
  }
};
struct PCG {
  u64 state = 0x853c49e6748fea9bULL;
  u64 inc   = 0xda3e39cb94b95bdbULL;
  u32 next() {
    uint64_t oldstate   = state;
    state               = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    int      rot        = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }
  f64 nextf() { return double(next()) / UINT32_MAX; }
};

template <typename K, typename Alloc_t = Default_Allocator> struct BinNode {
  BinNode *left  = NULL;
  BinNode *right = NULL;
  K        key;
  i32      depth = 1;
  ~BinNode()     = delete;
  explicit BinNode(K &&key) : key(std::move(key)) {}
  static BinNode *create(K &&key) {
    void *ptr = Alloc_t::alloc(sizeof(BinNode));
    return new (ptr) BinNode(std::move(key));
  }
  template <typename F> static void dump_dotgraph(BinNode *root, string_ref filename, F f) {
    if (root == NULL) return;
    TMP_STORAGE_SCOPE;
    FILE *dotgraph = fopen(stref_to_tmp_cstr(filename), "wb");
    fprintf(dotgraph, "digraph {\n");
    fprintf(dotgraph, "node [shape=record];\n");
    BinNode **stack        = (BinNode **)tl_alloc_tmp(sizeof(BinNode *) * (1 << 10));
    u32       stack_cursor = 0;
    BinNode * cur          = NULL;
    u32       null_id      = 0xffffffffu;
    stack[stack_cursor++]  = root;
    while (stack_cursor != 0) {
      cur = stack[--stack_cursor];
      ASSERT_ALWAYS(cur != NULL);
      fprintf(dotgraph, "%p", (void *)cur);
      f(dotgraph, "[label = \"%.*s\", shape = record];\n", cur);
      if (cur->right)
        fprintf(dotgraph, "%p -> %p [label = \"right\"];\n", (void *)cur,
                (size_t)(void *)cur->right);
      if (cur->left)
        fprintf(dotgraph, "%p -> %p [label = \"left\"];\n", (void *)cur, (size_t)(void *)cur->left);
      if (cur->right) stack[stack_cursor++] = cur->right;
      if (cur->left) stack[stack_cursor++] = cur->left;
    }
    fprintf(dotgraph, "}\n");
    fflush(dotgraph);
    fclose(dotgraph);
  }
  // returns new root for the subtree
  BinNode *put(K &&nk, bool replace = true) {
    if (replace && key == nk) {
      key = nk;
      return this;
    }
    if (nk < key) {
      if (left)
        left = left->put(std::move(nk));
      else {
        left = create(std::move(nk));
      }
    } else {
      if (right)
        right = right->put(std::move(nk));
      else {
        right = create(std::move(nk));
      }
    }
    return balance();
  }
  // returns new root for the subtree
  BinNode *remove(K const &k) {
    if (k == key) {
      if (right == NULL && left == NULL) {
        this->release();
        return NULL;
      }
      BinNode *new_root = NULL;
      if (get_diff() > 0)
        new_root = rotate_left();
      else
        new_root = rotate_right();
      return new_root->remove(k);
    }
    if (k < key) {
      if (left) {
        left = left->remove(k);
      } else {
        return this;
      }
    } else {
      if (right) {
        right = right->remove(k);
      } else {
        return this;
      }
    }
    return balance();
  }
  template <typename T> void traverse_keys(T t) {
    if (left) left->traverse_keys(t);
    t(key);
    if (right) right->traverse_keys(t);
  }
  enum {
    TRAVERSE_CONTINUE,
    TRAVERSE_BREAK,
  };
  template <typename T> int traverse_break(T t) {
    if (left)
      if (left->traverse_break(t) == TRAVERSE_BREAK) return TRAVERSE_BREAK;
    if (t(this) == TRAVERSE_BREAK) return TRAVERSE_BREAK;
    if (right)
      if (right->traverse_break(t) == TRAVERSE_BREAK) return TRAVERSE_BREAK;
    return TRAVERSE_CONTINUE;
  }
  // returns new root for the subtree
  BinNode *balance() {
    update_depth();
    i32 diff = get_diff();
    if (std::abs(diff) > 1) {
      if (diff > 0) {
        if (right->get_diff() < 0) right = right->rotate_right();
        return rotate_left();
      } else {
        if (left->get_diff() > 0) left = left->rotate_left();
        return rotate_right();
      }
    }
    return this;
  }
  i32 get_diff() {
    i32 ld = left ? left->depth : 0;
    i32 rd = right ? right->depth : 0;
    return rd - ld;
  }
  // returns new root for the subtree
  BinNode *rotate_left() {
    ASSERT_DEBUG(right);
    BinNode *new_root = this->right;
    this->right       = new_root->left;
    new_root->left    = this;
    update_depth();
    new_root->update_depth();
    return new_root;
  }
  // returns new root for the subtree
  BinNode *rotate_right() {
    ASSERT_DEBUG(left);
    BinNode *new_root = this->left;
    this->left        = left->right;
    new_root->right   = this;
    update_depth();
    new_root->update_depth();
    return new_root;
  }
  void update_depth() {
    depth = 1;
    if (right) {
      depth = MAX(depth, right->depth + 1);
    }
    if (left) {
      depth = MAX(depth, left->depth + 1);
    }
    ASSERT_DEBUG(depth > 0);
  }
  BinNode *find(K const &k) {
    if (key == k) return this;
    if (k > key)
      return right ? right->find(k) : NULL;
    else
      return left ? left->find(k) : NULL;
  }
  BinNode *lower_bound(K const &k) {
    if (key == k) return this;
    if (k > key) {
      if (right) {
        BinNode *lb = right->lower_bound(k);
        return lb ? lb : this;
      } else
        return this;
    } else if (left) {
      return left->lower_bound(k);
    } else
      return NULL;
  }
  BinNode *upper_bound(K const &k) {
    if (key == k) return this;
    if (k < key) {
      if (left) {
        BinNode *ub = left->upper_bound(k);
        return ub ? ub : this;
      } else
        return this;
    } else if (right) {
      return right->upper_bound(k);
    } else
      return NULL;
  }
  void release() {
    if (left) left->release();
    if (right) right->release();
    Alloc_t::free(this);
  }
};

class Util_Allocator {
  private:
  struct Alloc_Key {
    size_t offset;
    size_t size;
    bool   operator==(Alloc_Key const &that) const { return offset == that.offset; }
    bool   operator>(Alloc_Key const &that) const { return offset > that.offset; }
    bool   operator<(Alloc_Key const &that) const { return offset < that.offset; }
  };
  using BinNode_t      = BinNode<Alloc_Key>;
  BinNode_t *free_root = NULL;

  // Removed crap shite
  Util_Allocator(Util_Allocator const &) = delete;
  Util_Allocator(Util_Allocator &&)      = delete;
  Util_Allocator &operator=(Util_Allocator const &) = delete;
  Util_Allocator &operator=(Util_Allocator &&) = delete;

  public:
  ~Util_Allocator() {
    if (free_root) free_root->release();
  }
  Util_Allocator(size_t size) { free_root = BinNode_t::create({0, size}); }
  ptrdiff_t alloc(size_t alignment, size_t size) {
    ASSERT_DEBUG(size);
    if (alignment == 0) alignment = 1;
    ASSERT_DEBUG(alignment && ((alignment & (alignment - 1)) == 0)); // pot
    if (free_root == NULL) return -1;
    ptrdiff_t offset = -1;
    free_root->traverse_break([&](BinNode_t *node) {
      size_t end            = node->key.offset + node->key.size;
      size_t aligned_offset = (node->key.offset + alignment - 1) & (~(alignment - 1));
      if (aligned_offset >= end) return BinNode_t::TRAVERSE_CONTINUE;
      size_t usable_space = end - aligned_offset;
      if (usable_space >= size) {
        offset            = aligned_offset;
        size_t key_offset = node->key.offset;
        size_t key_size   = node->key.size;
        free_root         = free_root->remove({node->key});
        if (aligned_offset != key_offset) {
          size_t diff = aligned_offset - key_offset;
          ASSERT_DEBUG(diff < key_size && diff > 0);
          if (free_root == NULL) {
            free_root = BinNode_t::create({key_offset, diff});
          } else
            free_root = free_root->put({key_offset, diff});
        }
        size_t new_size = usable_space - size;
        if (new_size) {
          if (free_root == NULL) {
            free_root = BinNode_t::create({aligned_offset + size, new_size});
          } else
            free_root = free_root->put({aligned_offset + size, new_size});
        }
        return BinNode_t::TRAVERSE_BREAK;
      }
      return BinNode_t::TRAVERSE_CONTINUE;
    });
    return offset;
  }
  bool has_free_space(size_t alignment, size_t size) {
    ASSERT_DEBUG(size);
    if (alignment == 0) alignment = 1;
    ASSERT_DEBUG(alignment && ((alignment & (alignment - 1)) == 0)); // pot
    if (free_root == NULL) return false;
    ptrdiff_t offset = -1;
    free_root->traverse_break([&](BinNode_t *node) {
      size_t end            = node->key.offset + node->key.size;
      size_t aligned_offset = (node->key.offset + alignment - 1) & (~(alignment - 1));
      if (aligned_offset >= end) return BinNode_t::TRAVERSE_CONTINUE;
      size_t usable_space = end - aligned_offset;
      if (usable_space >= size) {
        offset = node->key.offset;
        return BinNode_t::TRAVERSE_BREAK;
      }
      return BinNode_t::TRAVERSE_CONTINUE;
    });
    return offset >= 0;
  }
  u32 get_num_nodes() {
    if (free_root == NULL) return 0;
    u32 size = 0;
    free_root->traverse_keys([&](Alloc_Key const &key) { size += 1; });
    return size;
  }
  size_t get_free_space() {
    if (free_root == NULL) return 0;
    size_t size = 0;
    free_root->traverse_keys([&](Alloc_Key const &key) { size += key.size; });
    return size;
  }
  void free(size_t offset, size_t size) {
    if (free_root == NULL) {
      free_root = BinNode_t::create({offset, size});
      return;
    }
    size_t new_offset = offset;
    size_t new_size   = size;
    // Try merging the lower bound node
    if (free_root) {
      BinNode_t *lb = free_root->lower_bound({offset, 0});
      if (lb) {
        ASSERT_DEBUG(lb->key.offset + lb->key.size <= new_offset);
        if (lb->key.offset + lb->key.size == new_offset) {
          new_offset = lb->key.offset;
          new_size   = lb->key.size + new_size;
          free_root  = free_root->remove(lb->key);
        }
      }
    }
    // Try merging the upper bound node
    if (free_root) {
      BinNode_t *ub = free_root->upper_bound({offset, 0});
      if (ub) {
        ASSERT_DEBUG(ub->key.offset > new_offset);
        if (ub->key.offset == new_offset + new_size) {
          new_size  = ub->key.size + new_size;
          free_root = free_root->remove(ub->key);
        }
      }
    }
    // Insert new node
    if (free_root == NULL) {
      free_root = BinNode_t::create({new_offset, new_size});
    } else
      free_root = free_root->put({new_offset, new_size});
  }
};

#endif

#ifdef UTILS_TL_IMPL
#ifndef UTILS_TL_IMPL_H
#  define UTILS_TL_IMPL_H
#  include <string.h>

#  ifndef UTILS_TL_TMP_SIZE
#    define UTILS_TL_TMP_SIZE 1 << 26
#  endif

struct Thread_Local {
  Temporary_Storage<> temporary_storage;
  bool                initialized = false;
  ~Thread_Local() { temporary_storage.release(); }
#  ifdef UTILS_TL_IMPL_DEBUG
  i64 allocated = 0;
#  endif
};

// TODO(aschrein): Change to __thread?
thread_local Thread_Local g_tl{};

Thread_Local *get_tl() {
  if (g_tl.initialized == false) {
    g_tl.initialized       = true;
    g_tl.temporary_storage = Temporary_Storage<>::create(UTILS_TL_TMP_SIZE);
  }
  return &g_tl;
}

void *tl_alloc_tmp(size_t size) { return get_tl()->temporary_storage.alloc(size); }

void tl_alloc_tmp_enter() { get_tl()->temporary_storage.enter_scope(); }
void tl_alloc_tmp_exit() { get_tl()->temporary_storage.exit_scope(); }

void *tl_alloc(size_t size) {
#  ifdef UTILS_TL_IMPL_DEBUG
  get_tl()->allocated += (i64)size;
  void *ptr          = malloc(size + sizeof(size_t));
  ((size_t *)ptr)[0] = size;
  return ((u8 *)ptr + 8);
#  elif UTILS_TL_IMPL_TRACY
  ZoneScopedS(16);
  void *ptr = malloc(size);
  TracyAllocS(ptr, size, 16);
  return ptr;
#  else
  return malloc(size);
#  endif
}

static inline void *_tl_realloc(void *ptr, size_t oldsize, size_t newsize) {
  return realloc(ptr, newsize);
#  if 0
  if (oldsize == newsize) return ptr;
  size_t min_size = oldsize < newsize ? oldsize : newsize;
  void * new_ptr  = NULL;
  if (newsize != 0) new_ptr = malloc(newsize);
  if (min_size != 0) {
    memcpy(new_ptr, ptr, min_size);
  }
  if (ptr != NULL) free(ptr);
  return new_ptr;
#  endif
}

#  ifdef UTILS_TL_IMPL_DEBUG
static inline void assert_tl_alloc_zero() {
  ASSERT_ALWAYS(get_tl()->allocated == 0);
  ASSERT_ALWAYS(get_tl()->temporary_storage.cursor == 0);
  ASSERT_ALWAYS(get_tl()->temporary_storage.stack_cursor == 0);
}
#  endif

void *tl_realloc(void *ptr, size_t oldsize, size_t newsize) {
#  ifdef UTILS_TL_IMPL_DEBUG
  if (ptr == NULL) {
    ASSERT_ALWAYS(oldsize == 0);
    return tl_alloc(newsize);
  }
  get_tl()->allocated -= (i64)oldsize;
  get_tl()->allocated += (i64)newsize;
  void *old_ptr = (u8 *)ptr - sizeof(size_t);
  ASSERT_ALWAYS(((size_t *)old_ptr)[0] == oldsize);
  void *new_ptr          = _tl_realloc(old_ptr, oldsize + sizeof(size_t), newsize + sizeof(size_t));
  ((size_t *)new_ptr)[0] = newsize;
  return ((u8 *)new_ptr + sizeof(size_t));
#  elif UTILS_TL_IMPL_TRACY
  ZoneScopedS(16);
  size_t minsize = MIN(oldsize, newsize);
  void * new_ptr = NULL;
  if (newsize > 0) {
    new_ptr = malloc(newsize);
    TracyAllocS(new_ptr, newsize, 16);
    if (minsize > 0) {
      memcpy(new_ptr, ptr, minsize);
    }
  }
  TracyFreeS(ptr, 16);
  return new_ptr;
#  else
  return _tl_realloc(ptr, oldsize, newsize);
#  endif
}

void tl_free(void *ptr) {
#  ifdef UTILS_TL_IMPL_DEBUG
  size_t size = ((size_t *)((u8 *)ptr - sizeof(size_t)))[0];
  get_tl()->allocated -= (i64)size;
  free(((u8 *)ptr - sizeof(size_t)));
  return;
#  elif UTILS_TL_IMPL_TRACY
  ZoneScopedS(16);
  TracyFreeS(ptr, 16);
  free(ptr);
#  else
  free(ptr);
#  endif
}
#endif // UTILS_TL_IMPL_H
#endif // UTILS_TL_IMPL
