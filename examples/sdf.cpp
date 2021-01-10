#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <atomic>
//#include <functional>
#include <imgui.h>
#include <mutex>
#include <thread>

#ifndef ZoneScopedS
#  define ZoneScopedS(x)
#  define ZoneScoped
#endif

inline void nop() {
#if defined(_WIN32)
  __noop();
#else
  __asm__ __volatile__("nop");
#endif
}
inline static uint64_t get_thread_id() {
  auto id = std::this_thread::get_id();
  return std::hash<std::thread::id>()(id);
}
struct Spin_Lock {
  std::atomic<bool> flag = {0};
  void              lock() {
    for (;;) {
      if (!flag.exchange(true, std::memory_order_acquire)) {
        return;
      }
      while (flag.load(std::memory_order_relaxed)) {
        ito(16) nop();
      }
    }
  }
  void unlock() { flag.store(false, std::memory_order_release); }
};

// Poor man's queue
// Not thread safe in all scenarios but kind of works in mine
// @Cleanup
template <typename Job_t> struct Queue {
  template <typename Job_t> struct Batch {
    Job_t *          job_queue;
    u32              capacity;
    std::atomic<u32> head;
    Spin_Lock        spinlock;
    void             lock() { spinlock.lock(); }
    void             unlock() { spinlock.unlock(); }
    void             init() {
      ZoneScoped;
      head      = 0;
      capacity  = 1 << 18;
      job_queue = (Job_t *)tl_alloc(sizeof(Job_t) * capacity);
    }
    void release() { tl_free(job_queue); }

    bool has_items() { return head.load() != 0; }
    bool try_dequeue(Job_t &job) {
      ZoneScoped;
      lock();
      defer(unlock());
      if (!has_items()) return false;
      u32 old_head = head.fetch_sub(1);
      job          = job_queue[old_head - 1];
      return true;
    }
    bool try_dequeue(Job_t *jobs, u32 *max_size) {
      ZoneScoped;
      lock();
      defer(unlock());
      if (!has_items()) {
        *max_size = 0;
        return false;
      }
      u32 num_items = MIN(*max_size, head.load());
      u32 old_head  = head.fetch_sub(num_items);
      memcpy(jobs, job_queue + old_head - num_items, sizeof(Job_t) * num_items);
      *max_size = num_items;
      assert(jobs[0].id != 0);
      return true;
    }
    void dequeue(Job_t *out, u32 &count) {
      ZoneScoped;
      lock();
      defer(unlock());
      if (head < count) {
        count = head;
      }
      u32 old_head = head.fetch_sub(count);
      memcpy(out, job_queue + head, count * sizeof(out[0]));
    }
    void enqueue(Job_t job) {
      ZoneScoped;
      lock();
      defer(unlock());
      u32 old_head        = head.fetch_add(1);
      job_queue[old_head] = job;
      ASSERT_PANIC(head <= capacity);
    }
    void enqueue(Job_t const *jobs, u32 num) {
      ZoneScoped;
      u32 old_head = 0;
      //{
      lock();
      defer(unlock());
      old_head = head.fetch_add(num);
      ASSERT_PANIC(head <= capacity);
      //}
      memcpy(job_queue + old_head, jobs, num * sizeof(Job_t));
    }
    bool has_job() { return head != 0u; }
    void reset() { head = 0u; }
  };

  static constexpr u32 NUM_BATCHES = 512;
  Batch<Job_t>         batches[NUM_BATCHES];
  void                 init() {
    ZoneScoped;
    ito(NUM_BATCHES) { batches[i].init(); }
  }
  void release() {
    ZoneScoped;
    ito(NUM_BATCHES) { batches[i].release(); }
  }
  bool has_items_any() {
    ito(NUM_BATCHES) if (batches[i].has_items()) return true;
    return false;
  }
  bool has_items(u32 id) { return batches[id % NUM_BATCHES].has_items(); }
  bool try_dequeue(u32 id, Job_t &job) { return batches[id % NUM_BATCHES].try_dequeue(job); }
  bool try_dequeue(u32 id, Job_t *jobs, u32 *max_size) {
    return batches[id % NUM_BATCHES].try_dequeue(jobs, max_size);
  }
  void dequeue(u32 id, Job_t *out, u32 &count) {
    return batches[id % NUM_BATCHES].dequeue(out, count);
  }
  void enqueue(u32 id, Job_t job) { return batches[id % NUM_BATCHES].enqueue(job); }
  void enqueue(u32 id, Job_t const *jobs, u32 num) {
    return batches[id % NUM_BATCHES].enqueue(jobs, num);
  }
  bool has_job(u32 id) { return batches[id % NUM_BATCHES].has_job(); }
};
struct Thread_Job {
  u32 id;
  void (*job)(u32);
};
struct Thread_Pool {
  u32                     num_threads = 0;
  std::thread *           thpool[0x100];
  std::atomic<bool>       threads_working;
  std::mutex              cv_mux;
  std::condition_variable cv;
  std::mutex              finished_mux;
  std::condition_variable finished_cv;
  std::atomic<u32>        workers_in_progress;

  Queue<Thread_Job> queue;

  void init_thread_pool() {
    threads_working = true;
    queue.init();
    num_threads = std::thread::hardware_concurrency();
#if defined(_WIN32)
    {
      SYSTEM_INFO sysinfo;
      GetSystemInfo(&sysinfo);
      num_threads = sysinfo.dwNumberOfProcessors;
    }
#endif

    fprintf(stdout, "Launching %i threads\n", num_threads);
    for (u32 thread_id = 0; thread_id < num_threads; thread_id++) {
      thpool[thread_id] = new std::thread([&, thread_id] {
      // Set affinity
#if defined(_WIN32)
        SetThreadAffinityMask(GetCurrentThread(), (u64)1 << (u64)thread_id);
#endif // _WIN32
        u32 batch_hash = thread_id;
        while (threads_working) {
          batch_hash = (u32)hash_of(batch_hash);
          if (queue.has_items(batch_hash) == false) {
            finished_cv.notify_one();
            ito(32) nop();
            if (!threads_working) break;
            continue;
          }
          workers_in_progress++;
          defer({ workers_in_progress--; });
          u32        num_jobs = 1;
          Thread_Job job;
          if (!queue.try_dequeue(batch_hash, &job, &num_jobs)) continue;
          job.job(job.id);
        }
      });
      cv.notify_one();
    }
  }
  void release() {
    wait();
    threads_working = false;

    ito(num_threads) {
      cv.notify_all();
      thpool[i]->join();
      delete thpool[i];
    }
  }
  void wait() {
    std::unique_lock<std::mutex> lk(finished_mux);
    finished_cv.wait(lk,
                     [&] { return queue.has_items_any() == false && workers_in_progress == 0; });
  }
} thread_pool;

template <typename T = f32> struct ReLuActivation {
  static T           f(T i) { return i > (T)0 ? i : (T)0; }
  static T           deriv(T i) { return i > (T)0 ? (T)1 : (T)0; }
  static char const *str() { return "relu"; }
};

template <typename T = f32> struct TanhActivation {
  static T f(T i) { return std::tanh(i); }
  static T deriv(T i) {
    T a = std::tanh(i);
    return (T)1 - a * a;
  }
  static char const *str() { return "tanh"; }
};

template <typename T = f32> struct NoOpActivation {
  static T           f(T i) { return i; }
  static T           deriv(T i) { return (T)1; }
  static char const *str() { return ""; }
};

struct BaseNNLayer {
  AutoArray<f32> inputs;
  AutoArray<f32> input_errors;
  AutoArray<f32> output_derivs;
  AutoArray<f32> outputs;
  AutoArray<f32> coefficients;
  u32            input_size;
  u32            output_size;
  u32            row_size;
  static PCG &   get_pcg() {
    static PCG pcg;
    return pcg;
  }
  void __init(u32 input_size, u32 output_size) {
    this->input_size  = input_size;
    this->output_size = output_size;

    row_size = input_size + 1;
    coefficients.resize(row_size * output_size);
    inputs.resize(row_size);
    inputs[input_size] = 1.0f;
    input_errors.resize(row_size);
    outputs.resize(output_size);
    output_derivs.resize(output_size);

    ito(coefficients.size) {
      coefficients[i] = // 1.0f;
          get_pcg().nextf() * 2.0f - 1.0f;
    }
  }
  u32                  getOutputSize() { return output_size; }
  f32                  getOutput(i32 i) { return f(outputs[i]); }
  virtual f32          f(f32 x)     = 0;
  virtual f32          deriv(f32 x) = 0;
  virtual char const * fStr() const = 0;
  virtual BaseNNLayer *clone()      = 0;
  void                 cloneInto(BaseNNLayer *layer) {
    layer->inputs.cloneFrom(inputs);
    layer->input_errors.cloneFrom(input_errors);
    layer->output_derivs.cloneFrom(output_derivs);
    layer->outputs.cloneFrom(outputs);
    layer->coefficients.cloneFrom(coefficients);
    layer->input_size  = input_size;
    layer->output_size = output_size;
    layer->row_size    = row_size;
  }
  void init(f32 const *in) { ito(input_size) inputs[i] = in[i]; }
  void eval() {
    ito(outputs.size) {
      f32 result = (f32)0;
      jto(inputs.size) { result += inputs[j] * coefficients[row_size * i + j]; }
      outputs[i] = result;
      // fprintf(stdout, "[%f] ", result);
    };
  }
  float rexp(float x) { return 1.0f - std::exp(-std::abs(x)); }
  // bool  do_update(float x) { return rexp(x) > get_pcg().nextf(); }
  void solve(f32 const *error, f32 const dt) {
    ito(outputs.size) output_derivs[i] = deriv(outputs[i]);
    ito(inputs.size) {
      f32 result = (f32)0;
      jto(outputs.size) {
        f32 dfdx = output_derivs[j];
        result += error[j] * dfdx * coefficients[row_size * j + i];
      }
      input_errors[i] = result;
      // fprintf(stdout, "[%f] ", result);
    }
    // inputs[input_size] += dt * input_errors[input_size];
    // fprintf(stdout, "c=[");
    ito(outputs.size) {
      f32 dfdx = output_derivs[i];
      jto(inputs.size) {
        // if (do_update(error[i]))
        coefficients[row_size * i + j] += dt * dfdx * inputs[j] * error[i];
        // fprintf(stdout, "%f ", coefficients[row_size * i + j]);
      }
    }
    // fprintf(stdout, "] ");
  }
  f32 *getErrors() { return &input_errors[0]; }
  void provide(BaseNNLayer *layer) {
    ASSERT_ALWAYS(outputs.size == layer->input_size);
    ito(outputs.size) layer->inputs[i] = f(outputs[i]);
  }
  void reset() {
    inputs[input_size] = 1.0f;
    ito(coefficients.size) {
      coefficients[i] = // 1.0f;
          get_pcg().nextf() * 2.0f - 1.0f;
    }
  }
  void regulateL1(float dt) {
    ito(coefficients.size) {
      if (isnan(coefficients[i])) coefficients[i] = 0.0f;
      if (isinf(coefficients[i])) coefficients[i] = 0.0f;
      coefficients[i] -= dt * sign(coefficients[i]);
    }
  }
  void regulateL2(float dt) {
    ito(coefficients.size) {
      if (isnan(coefficients[i])) coefficients[i] = 0.0f;
      if (isinf(coefficients[i])) coefficients[i] = 0.0f;
      coefficients[i] -= dt * coefficients[i];
    }
  }
  void provide(f32 *dst) { ito(outputs.size) dst[i] = f(outputs[i]); }
  // f32 &  get(u32 from, u32 to) { return coefficients[from * row_size + to]; }
  void release() { delete this; }
  void on_gui(Gizmo_Layer *gl, float3 origin, float stride, float width) {
    float3 forward   = float3(1.0f, 0.0f, 0.0f);
    float3 up        = float3(0.0f, 0.0f, 1.0f);
    float3 v0_offset = float3(0.0f, 0.0f, -((inputs.size - 1) * stride) / 2.0f);
    float3 v1_offset = float3(0.0f, 0.0f, -(outputs.size * stride) / 2.0f);
    ito(outputs.size) {
      jto(inputs.size) {
        float3 v0 = origin + up * f32(j) * stride;
        float3 v1 = origin + up * f32(i) * stride + forward * width;
        float  k  = coefficients[i * row_size + j];
        float  a  = max(0.0f, -k);
        k         = max(0.0f, k);
        gl->draw_cylinder(v0 + v0_offset, v1 + v1_offset, 1.0e-2f, float3(k, (k + a) / 2.0f, a));
      }
    }
  }
};

template <typename F = ReLuActivation<f32>> struct NNLayer : public BaseNNLayer {
  NNLayer(u32 input_size, u32 output_size) { BaseNNLayer::__init(input_size, output_size); }
  f32          f(f32 x) override { return F::f(x); }
  f32          deriv(f32 x) override { return F::deriv(x); }
  char const * fStr() const override { return F::str(); }
  BaseNNLayer *clone() override {
    NNLayer<F> *layer = new NNLayer<F>(input_size, output_size);
    cloneInto(layer);
    return layer;
  }
};

class NN {
  enum Activation { eUnknown, eReLu, eTanh, eDelay, eNoOp };

  Activation parse_activation(string_ref token) {
    if (token.eq("ReLu")) {
      return eReLu;
    } else if (token.eq("Tanh")) {
      return eTanh;
    } else if (token.eq("NoOp")) {
      return eNoOp;
    } else if (token.eq("Delay")) {
      return eDelay;
    } else {
      UNIMPLEMENTED;
    }
  }
  void init_traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      init_traverse(l->child);
      init_traverse(l->next);
    } else {
      if (l->cmp_symbol("layer")) {
        u32        num_inputs  = l->get(1)->parse_int();
        u32        num_outputs = l->get(2)->parse_int();
        Activation activation  = parse_activation(l->get(3)->symbol);
        if (activation == eReLu) {
          layers.push(new NNLayer<ReLuActivation<f32>>(num_inputs, num_outputs));
        } else if (activation == eTanh) {
          layers.push(new NNLayer<TanhActivation<f32>>(num_inputs, num_outputs));
        } else if (activation == eNoOp) {
          layers.push(new NNLayer<NoOpActivation<f32>>(num_inputs, num_outputs));
        } else {
          UNIMPLEMENTED;
        }
      } else {
      }
    }
  }

  public:
  void on_gui(Gizmo_Layer *gl) {
    ito(layers.size) {
      // ImGui::SameLine();
      layers[i]->on_gui(gl, float3(2.0f, 0.0f, 0.0f) + f32(i) * float3(1.0f, 0.0f, 0.0f), 1.0f,
                        1.0f);
    }
  }

  static NN *create(List *l) {
    NN *n = new NN;
    n->init(l);
    return n;
  }

  NN *clone() {
    NN *out = new NN;
    ito(layers.size) out->layers.push(layers[i]->clone());
    return out;
  }

  void release() {
    ito(layers.size) layers[i]->release();
    delete this;
  }

  void eval(f32 const *inputs, f32 *outputs) {
    layers[0]->init(inputs);
    ito(layers.size) {
      layers[i]->eval();
      if (i != layers.size - 1) layers[i]->provide(layers[i + 1]);
    }
    layers[layers.size - 1]->provide(outputs);
  }
  void reset() { ito(layers.size) layers[i]->reset(); }
  void regulateL1(float dt) { ito(layers.size) layers[i]->regulateL1(dt); }
  void regulateL2(float dt) { ito(layers.size) layers[i]->regulateL2(dt); }
  void solve(f32 const *error, f32 const dt) {
    layers[layers.size - 1]->solve(error, dt);
    for (int i = layers.size - 2; i >= 0; i--) {
      layers[i]->solve(layers[i + 1]->getErrors(), dt);
    }
  }

  AutoArray<BaseNNLayer *> layers;

  private:
  void init(List *l) { init_traverse(l); }

  NN() {}
  ~NN() {}
};

// static int __test_nn_1 = [] {
//  TMP_STORAGE_SCOPE;
//  List *l = List::parse(stref_s(R"(
//  (layer 1 1 NoOp)
//  )"),
//                        Tmp_List_Allocator{});
//  ASSERT_ALWAYS(l);
//  NN<f32> *nn = NN<f32>::create(l);
//  int      N  = 10000;
//  jto(N) {
//    int   i = (j % 100);
//    float a = float(i);
//    float c = float(i) + 25.0f;
//    float res;
//    //fprintf(stdout, "input=[%f] ", a);
//    nn->eval(&a, &res);
//    //fprintf(stdout, "result=[%f] ", res);
//    float error = c - res;
//    //nn->regulateL1(1.0e-3f);
//    nn->solve(&error, 1.0e-4);
//
//    //fprintf(stdout, "error=[%f]\n", error);
//    // fprintf(stdout, "###\n");
//  }
//
//  defer(nn->release());
//
//  return 0;
//}();
// static int __test_nn_2 = [] {
//  TMP_STORAGE_SCOPE;
//  List *l = List::parse(stref_s(R"(
//  (layer 2 1 ReLu)
//  (layer 1 1 ReLu)
//  )"),
//                        Tmp_List_Allocator{});
//  ASSERT_ALWAYS(l);
//  NN<f32> *nn = NN<f32>::create(l);
//  struct TestData {
//    float a, b;
//    float c;
//  };
//  int             N = 10000;
//  Array<TestData> td;
//  td.init(N);
//  defer(td.release());
//  PCG pcg;
//  ito(N) {
//    float a = pcg.nextf();
//    float b = pcg.nextf();
//    float c = a > b ? 1.0 : 0.0;
//    //float c = a > 0.5f ? (b > 0.5 ? 1.0f : 0.0f) : (b > 0.5f ? 0.0f : 1.0f);
//    td.push({a, b, c});
//  }
//  ito(N) {
//    float res;
//    nn->eval(&td[i].a, &res);
//    float error = td[i].c - res;
//    nn->solve(&error, 0.003f);
//     //nn->regulateL1(0.001f);
//    fprintf(stdout, "error=[%f]\n", error);
//  }
//  defer(nn->release());
//
//  return 0;
//}();

Config       g_config;
Scene *      g_scene     = Scene::create();
Gizmo_Layer *gizmo_layer = NULL;
NN *         nn          = NULL;
struct TestData {
  float x, y, z;
  float dist;
};
int             N = 10000;
Array<TestData> td;

template <typename T> class GPUBuffer {
  private:
  rd::IDevice *factory;
  Array<T>      cpu_array;
  Resource_ID   gpu_buffer;
  Resource_ID   cpu_buffer;
  size_t        gpu_buffer_size;

  public:
  void init(rd::IDevice *factory) {
    this->factory = factory;
    cpu_array.init();
    gpu_buffer.reset();
    cpu_buffer.reset();
    gpu_buffer_size = 0;
  }
  void push(T a) { cpu_array.push(a); }
  void clear() {
    cpu_array.release();
    factory->release_resource(gpu_buffer);
    factory->release_resource(cpu_buffer);
    gpu_buffer.reset();
  }
  Resource_ID get() { return gpu_buffer; }
  void        reset() { cpu_array.reset(); }
  void        flush(rd::ICtx *ctx = NULL) {
    if (gpu_buffer.is_null() || cpu_array.size * sizeof(T) < gpu_buffer_size) {
      if (cpu_buffer) factory->release_resource(cpu_buffer);
      if (gpu_buffer) factory->release_resource(gpu_buffer);
      gpu_buffer_size = cpu_array.size * sizeof(T);
      {
        rd::Buffer_Create_Info info;
        MEMZERO(info);
        info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits =
            (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        info.size  = cpu_array.size * sizeof(T);
        gpu_buffer = factory->create_buffer(info);
      }
      {
        rd::Buffer_Create_Info info;
        MEMZERO(info);
        info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
        info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
        info.size       = cpu_array.size * sizeof(T);
        cpu_array       = factory->create_buffer(info);
      }
    }
    void *ptr = factory->map_buffer(cpu_buffer);
    memcpy(ptr, cpu_array.ptr, gpu_buffer_size);
    factory->unmap_buffer(cpu_buffer);
    if (ctx == NULL) {
      ctx = factory->start_compute_pass();
      ctx->copy_buffer(cpu_buffer, 0, gpu_buffer, 0, gpu_buffer_size);
      factory->end_compute_pass(ctx);
    } else {
      ctx->copy_buffer(cpu_buffer, 0, gpu_buffer, 0, gpu_buffer_size);
    }
  }
  void release() {
    if (cpu_buffer) factory->release_resource(cpu_buffer);
    if (gpu_buffer) factory->release_resource(gpu_buffer);
    cpu_array.release();
  }
};

void build_nn_shader(String_Builder &sb, NN *nn) {
  sb.putf("void eval_sdf(\n");
  ito(nn->layers[0]->input_size) {
    sb.putf("in float input_%i,\n", i);
    sb.putf("\n");
  }
  ito(nn->layers[nn->layers.size - 1]->output_size) { sb.putf("out float output_%i,\n", i); }
  // if (i != nn->layers[nn->layers.size - 1]->output_size - 1) sb.putf(",");
  sb.putf("in int exit_layer, in int exit_point");
  sb.putf(") {\n");
  ito(nn->layers.size) {
    auto layer = nn->layers[i];
    jto(layer->input_size) { sb.putf("float n_%i_%i = 0.0;\n", i, j); }
    sb.putf("float n_%i_%i = 1.0;\n", i, layer->input_size);
  }
  ito(nn->layers[nn->layers.size - 1]->output_size) { sb.putf("output_%i = 0.0;\n", i); }
  ito(nn->layers[0]->input_size) { sb.putf("n_0_%i = input_%i;\n", i, i); }
  int offset = 0;
  ito(nn->layers.size) {
    auto layer = nn->layers[i];
    jto(layer->input_size + 1) {
      kto(layer->output_size) {
        sb.putf("float c_%i_%i_%i = buffer_load(params, %i);\n", i, j, k, offset);
        offset++;
      }
    }
  }
  ito(nn->layers.size - 1) {
    auto layer = nn->layers[i];
    kto(layer->output_size) {
      jto(layer->input_size + 1) {
        sb.putf("n_%i_%i += n_%i_%i * c_%i_%i_%i;\n", i + 1, k, i, j, i, j, k);
      }
      sb.putf("n_%i_%i = %s(n_%i_%i);\n", i + 1, k, layer->fStr(), i + 1, k);
      sb.putf("if (exit_layer == %i && exit_point == %i) {output_0 = n_%i_%i; return; }\n", i, k,
              i + 1, k);
    }
  }
  {
    i32  i     = nn->layers.size - 1;
    auto layer = nn->layers[nn->layers.size - 1];
    kto(layer->output_size) {
      jto(layer->input_size + 1) {
        sb.putf("output_%i += n_%i_%i * c_%i_%i_%i;\n", k, i, j, i, j, k);
      }
      sb.putf("output_%i = %s(output_%i);\n", k, layer->fStr(), k);
    }
  }
  sb.putf("}\n");
}

void render_mesh(Mesh const &mesh, float4x4 const &model, rd::IDevice *factory, rd::ICtx *ctx) {
  if (mesh.vertexCount) {
    ctx->push_state();
    defer(ctx->pop_state());
    ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name world_transform))
)
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
)

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_NORMAL))

@(ENTRY)
  PIXEL_NORMAL   = NORMAL;
  @(EXPORT_POSITION mul4(viewproj, mul4(world_transform, float4(POSITION / 16.0 - float3_splat(1.0), 1.0))));
@(END)
)"),
                                                  NULL, 0));

    ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_NORMAL))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  @(EXPORT_COLOR 0 abs(float4(PIXEL_NORMAL, 1.0)));
@(END)
)"),
                                                  NULL, 0));
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits         = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits       = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size             = sizeof(float3) * mesh.vertexCount;
      Resource_ID vertex_buffer = factory->create_buffer(buf_info);
      factory->release_resource(vertex_buffer);
      {
        float3 *vertices = (float3 *)factory->map_buffer(vertex_buffer);
        memcpy(vertices, mesh.vertices, sizeof(float3) * mesh.vertexCount);
        factory->unmap_buffer(vertex_buffer);
      }
      ctx->IA_set_vertex_buffer(0, vertex_buffer, 0, 12, rd::Input_Rate::VERTEX);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits         = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits       = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size             = sizeof(float3) * mesh.vertexCount;
      Resource_ID vertex_buffer = factory->create_buffer(buf_info);
      factory->release_resource(vertex_buffer);
      {
        float3 *vertices = (float3 *)factory->map_buffer(vertex_buffer);
        memcpy(vertices, mesh.normals, sizeof(float3) * mesh.vertexCount);
        factory->unmap_buffer(vertex_buffer);
      }
      ctx->IA_set_vertex_buffer(1, vertex_buffer, 0, 12, rd::Input_Rate::VERTEX);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits        = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits      = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size            = mesh.faceCount * 3 * sizeof(u32);
      Resource_ID index_buffer = factory->create_buffer(buf_info);
      factory->release_resource(index_buffer);
      {
        u32 *indices = (u32 *)factory->map_buffer(index_buffer);
        memcpy(indices, mesh.faces, mesh.faceCount * 3 * sizeof(u32));
        factory->unmap_buffer(index_buffer);
      }
      ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
    }
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      ctx->IA_set_attribute(info);
    }
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 1;
      info.offset   = 0;
      info.type     = rd::Attriute_t::NORMAL;
      ctx->IA_set_attribute(info);
    }
    rd::RS_State rs_state;
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CW;
    rs_state.cull_mode    = rd::Cull_Mode::NONE;
    ctx->RS_set_state(rs_state);
    struct PC {
      float4x4 world_transform;
    } pc;
    struct Uniform {
      afloat4x4 viewproj;
    };
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits          = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits        = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size              = sizeof(Uniform);
      Resource_ID uniform_buffer = factory->create_buffer(buf_info);
      factory->release_resource(uniform_buffer);
      Uniform *ptr  = (Uniform *)factory->map_buffer(uniform_buffer);
      ptr->viewproj = gizmo_layer->get_camera().viewproj();
      factory->unmap_buffer(uniform_buffer);
      ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
    }

    // ito(nn->layers.size) {
    // BaseNNLayer *layer = nn->layers[i];
    // jto(layer->output_size) {
    // u32 i              = 0;
    // u32 j              = 0;
    pc.world_transform = model;
    /*    pc.g_exit_layer    = i;
        pc.g_exit_point    = j;*/
    ctx->push_constants(&pc, 0, sizeof(pc));
    // gizmo_layer->render_linebox(
    ctx->draw_indexed(mesh.faceCount * 3, 1, 0, 0, 0);
    //}
    //}
  }
}

void render_layers(NN *_nn, rd::IDevice *factory, rd::ICtx *ctx) {
  int num_meshes = 0;
  jto(nn->layers.size) {
    kto(nn->layers[j]->getOutputSize()) { num_meshes++; }
  }
  std::atomic<i32> cnt;
  Mesh *           meshes = new Mesh[num_meshes];
  defer(delete[] meshes);
  jto(nn->layers.size) {
    kto(nn->layers[j]->getOutputSize()) {
      NN *nn = _nn->clone();
      defer(nn->release());
      constexpr int N = 32;
      float *       f = new float[N * N * N];
      defer(delete[] f);
      zto(N) {
        yto(N) {
          xto(N) {
            float _x = (float(x) / (N - 1)) * 2.0f - 1.0f;
            float _y = (float(y) / (N - 1)) * 2.0f - 1.0f;
            float _z = (float(z) / (N - 1)) * 2.0f - 1.0f;
            float res;
            float in[] = {_x, _y, _z};
            nn->eval(in, &res);
            float c                  = nn->layers[j]->getOutput(k);
            f[z * N * N + y * N + x] = c;
          }
        }
      }
      Mesh mesh      = march(f, N, N, N, 0.0f);
      mesh.offset[0] = 2.2f * (j + 1);
      mesh.offset[1] = 2.2f * (k + 0);
      mesh.offset[2] = 0.0f;
      meshes[cnt++]  = mesh;
    }
  }
  ito(num_meshes) {
    Mesh &mesh = meshes[i];
    defer({
      delete[] mesh.vertices;
      delete[] mesh.faces;
    });
    render_mesh(mesh, translate(float4x4(1.0f), float3(mesh.offset[0], mesh.offset[1], 0.0f)),
                factory, ctx);
  }
  // translate(float4x4(1.0f), float3(2.2f * (layer + 1), 2.2f * output, 0.0f))
}

void __test_thread(u32 id) { printf("thread %i is launched\n", id); }

class GBufferPass {
  public:
  Resource_ID normal_rt;
  Resource_ID depth_rt;

  public:
  void render_sdf(rd::ICtx *ctx, NN *nn, float3 origin) {
    String_Builder sb;
    sb.init();
    defer(sb.release());
  }

  void init() { MEMZERO(*this); }
  void render(rd::IDevice *factory) {
    float4x4 bvh_visualizer_offset = translate(float4x4(1.0f), float3(-10.0f, 0.0f, 0.0f));
    g_scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GfxSufraceComponent>() == NULL) {
          GfxSufraceComponent::create(factory, mn);
        }
        render_bvh(bvh_visualizer_offset, mn->getComponent<GfxSufraceComponent>()->getBVH(),
                   gizmo_layer);
      }
    });

    u32 width  = g_config.get_u32("g_buffer_width");
    u32 height = g_config.get_u32("g_buffer_height");
    {
      rd::Image_Create_Info rt0_info;

      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      normal_rt = get_or_create_image(factory, rt0_info, normal_rt);
    }
    {
      rd::Image_Create_Info rt0_info;

      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::D32_OR_R32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT;
      depth_rt            = get_or_create_image(factory, rt0_info, depth_rt);
    }
    {
      rd::Render_Pass_Create_Info info;
      MEMZERO(info);
      info.width  = width;
      info.height = height;
      rd::RT_View rt0;
      MEMZERO(rt0);
      rt0.image             = normal_rt;
      rt0.format            = rd::Format::NATIVE;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.5f;
      rt0.clear_color.g     = 0.5f;
      rt0.clear_color.b     = 0.5f;
      rt0.clear_color.a     = 1.0f;
      info.rts.push(rt0);

      info.depth_target.image             = depth_rt;
      info.depth_target.clear_depth.clear = true;
      info.depth_target.format            = rd::Format::NATIVE;

      rd::ICtx *ctx = factory->start_render_pass(info);
      ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float4x4)  (name world_transform))
)

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))
@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_OUTPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(ENTRY)
  PIXEL_POSITION   = POSITION;
  PIXEL_NORMAL     = NORMAL;
  PIXEL_TEXCOORD0  = TEXCOORD0;
  @(EXPORT_POSITION mul4(viewproj, mul4(world_transform, float4(POSITION, 1.0))));
@(END)
)"),
                                                    NULL, 0));
      ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_INPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  float4 color = float4(PIXEL_NORMAL, 1.0);
  @(EXPORT_COLOR 0 color);
@(END)
)"),
                                                    NULL, 0));
      static u32 attribute_to_location[] = {
          0xffffffffu, 0, 1, 2, 3, 4, 5, 6, 7, 8,
      };
      setup_default_state(ctx, 1);
      rd::DS_State ds_state;
      MEMZERO(ds_state);
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = true;
      ds_state.enable_depth_write = true;
      ctx->DS_set_state(ds_state);
      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
      ctx->RS_set_state(rs_state);
      float4x4 viewproj = gizmo_layer->get_camera().viewproj();
      ctx->push_constants(&viewproj, 0, sizeof(float4x4));
      g_scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
          float4x4             model = mn->get_transform();
          ctx->push_constants(&model, 64, sizeof(model));
          ito(gs->getNumSurfaces()) {
            GfxSurface *s = gs->getSurface(i);
            s->draw(ctx, attribute_to_location);
          }
        }
      });
      /*if (0) {
        static int id = 1;
        ito(1000) {
          Thread_Job job;
          job.job = __test_thread;
          job.id  = id;
          thread_pool.queue.enqueue(id, &job, 1);
          id++;
        }
        thread_pool.wait();
      }*/
      if (0) {
        render_layers(nn, factory, ctx);
      }
      if (g_config.get_bool("render SDF")) {
        ctx->push_state();
        defer(ctx->pop_state());
        ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name world_transform))
  (add_field (type int)  (name g_exit_layer))
  (add_field (type int)  (name g_exit_point))
)
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float3) (name camera_pos))
  (add_field (type float3) (name camera_look))
  (add_field (type float3) (name camera_up))
  (add_field (type float3) (name camera_right))
)

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_WORLD_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_OBJECT_POSITION))

@(ENTRY)
  PIXEL_WORLD_POSITION   = mul4(world_transform, float4(POSITION, 1.0)).xyz;
  PIXEL_OBJECT_POSITION   = POSITION;
  @(EXPORT_POSITION mul4(viewproj, mul4(world_transform, float4(POSITION, 1.0))));
@(END)
)"),
                                                      NULL, 0));
        String_Builder sb;
        sb.init();
        defer(sb.release());
        sb.putf(R"(
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 1)
  (type float)
  (name params)
)

float relu(float x) {
  return x < 0.0 ? 0.0 : x;
}
)");
        build_nn_shader(sb, nn);

        sb.putf(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name world_transform))
  (add_field (type int)  (name g_exit_layer))
  (add_field (type int)  (name g_exit_point))
)
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float3) (name camera_pos))
  (add_field (type float3) (name camera_look))
  (add_field (type float3) (name camera_up))
  (add_field (type float3) (name camera_right))
)

@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_WORLD_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_OBJECT_POSITION))

float dot2( in float2 v ) { return dot(v,v); }
float dot2( in float3 v ) { return dot(v,v); }
float ndot( in float2 a, in float2 b ) { return a.x*b.x - a.y*b.y; }

///////////////////////////////////////////////

struct Camera {
    float3 pos;
    float3 look;
    float3 up;
    float3 right;
    float  fov;
};

struct Ray {
    float3 o;
    float3 d;
};

struct Collision {
    float  t;
    float3 pos;
    float dist;
    //float3 normal;
};

float eval_sdf(float3 ro) {
    float res = 0;
    eval_sdf(ro.x, ro.y, ro.z, res, g_exit_layer, g_exit_point);
    return res;
}

float3 eval_normal(float3 pos) {
    float2 e = float2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*eval_sdf( pos + e.xyy ) + 
					  e.yyx*eval_sdf( pos + e.yyx ) + 
					  e.yxy*eval_sdf( pos + e.yxy ) + 
					  e.xxx*eval_sdf( pos + e.xxx ) );
}

Collision scene(
    float3 ro,
    float3 rd) {
    Collision col;
    col.t = 99999.0;
    const uint MAX_ITER = 32u;
    float tmin = 1.0e-2;
    float tmax = 8.0;
    float t = tmin;
    for (uint i = 0u; i < MAX_ITER; i++) {
        float3 p = ro + rd * t;
        float d = eval_sdf(p);
        col.dist = d;
        if (d < 1.0e-3) {
            col.t = t;
            col.pos = p;
            //col.normal = eval_normal(p);
            break;
        }
        t += d;
        if (t > tmax)
            break;
    }
    return col;
}


@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  // Camera cam;
  // cam.pos   = camera_pos;
  // cam.look  = camera_look;
  // cam.up    = camera_up;
  // cam.right = camera_right;
  Ray r;
  r.o = PIXEL_OBJECT_POSITION;
  r.d = normalize(PIXEL_WORLD_POSITION - camera_pos);
  Collision col = scene(r.o, r.d);
  if (col.t > 1000.0)
      discard;
    
  float4 color = float4(
          float3(abs(col.pos)),
          1.0);
  //float res;
  //eval_sdf( PIXEL_OBJECT_POSITION.x,
  //          PIXEL_OBJECT_POSITION.y,
  //          PIXEL_OBJECT_POSITION.z, res, g_exit_layer, g_exit_point);
  //float4 color = float4(abs(res));
  @(EXPORT_COLOR 0 color);
@(END)
)");
        ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, sb.get_str(), NULL, 0));
        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits         = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits       = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
          buf_info.size             = sizeof(float3) * 8;
          Resource_ID vertex_buffer = factory->create_buffer(buf_info);
          factory->release_resource(vertex_buffer);
          {
            float3 *vertices     = (float3 *)factory->map_buffer(vertex_buffer);
            float3  dvertices[8] = {float3(-1, -1, -1), float3(1, -1, -1), float3(1, 1, -1),
                                   float3(-1, 1, -1),  float3(-1, -1, 1), float3(1, -1, 1),
                                   float3(1, 1, 1),    float3(-1, 1, 1)};
            ito(8) {
              /* float x     = ((i >> 0) & 1) * 2.0f - 1.0f;
               float y     = ((i >> 1) & 1) * 2.0f - 1.0f;
               float z     = ((i >> 2) & 1) * 2.0f - 1.0f;*/
              // vertices[i] = {x, y, z};
              vertices[i] = dvertices[i];
            }
            factory->unmap_buffer(vertex_buffer);
          }
          ctx->IA_set_vertex_buffer(0, vertex_buffer, 0, 12, rd::Input_Rate::VERTEX);
        }
        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits        = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits      = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
          buf_info.size            = 6 * 2 * 3 * sizeof(u32);
          Resource_ID index_buffer = factory->create_buffer(buf_info);
          factory->release_resource(index_buffer);
          {
            u32 *indices             = (u32 *)factory->map_buffer(index_buffer);
            u32  dindices[6 * 2 * 3] = {0, 1, 3, 3, 1, 2, 1, 5, 2, 2, 5, 6, 5, 4, 6, 6, 4, 7,
                                       4, 0, 7, 7, 0, 3, 3, 2, 7, 7, 2, 6, 4, 5, 0, 0, 5, 1};
            ito(ARRAYSIZE(dindices)) indices[i] = dindices[i];
            factory->unmap_buffer(index_buffer);
          }
          ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
        }
        {
          rd::Attribute_Info info;
          MEMZERO(info);
          info.binding  = 0;
          info.format   = rd::Format::RGB32_FLOAT;
          info.location = 0;
          info.offset   = 0;
          info.type     = rd::Attriute_t::POSITION;
          ctx->IA_set_attribute(info);
        }
        rd::RS_State rs_state;
        MEMZERO(rs_state);
        rs_state.polygon_mode = rd::Polygon_Mode::FILL;
        rs_state.front_face   = rd::Front_Face::CW;
        rs_state.cull_mode    = rd::Cull_Mode::NONE;
        ctx->RS_set_state(rs_state);
        struct PC {
          float4x4 world_transform;
          i32      g_exit_layer;
          i32      g_exit_point;
        } pc;
        struct Uniform {
          afloat4x4 viewproj;
          afloat3   camera_pos;
          afloat3   camera_look;
          afloat3   camera_up;
          afloat3   camera_right;
        };
        {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits          = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits        = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
          buf_info.size              = sizeof(Uniform);
          Resource_ID uniform_buffer = factory->create_buffer(buf_info);
          factory->release_resource(uniform_buffer);
          Uniform *ptr      = (Uniform *)factory->map_buffer(uniform_buffer);
          ptr->camera_pos   = gizmo_layer->get_camera().pos;
          ptr->camera_look  = gizmo_layer->get_camera().look;
          ptr->camera_up    = gizmo_layer->get_camera().up;
          ptr->camera_right = gizmo_layer->get_camera().right;
          ptr->viewproj     = gizmo_layer->get_camera().viewproj();
          factory->unmap_buffer(uniform_buffer);
          ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
        }
        {
          int size = 0;
          ito(nn->layers.size) {
            auto layer = nn->layers[i];
            jto(layer->input_size + 1) {
              kto(layer->output_size) { size++; }
            }
          }
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.mem_bits         = (u32)rd::Memory_Bits::HOST_VISIBLE;
          buf_info.usage_bits       = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
          buf_info.size             = size * sizeof(float);
          Resource_ID params_buffer = factory->create_buffer(buf_info);
          factory->release_resource(params_buffer);
          float *ptr = (float *)factory->map_buffer(params_buffer);
          ito(nn->layers.size) {
            auto layer = nn->layers[i];
            jto(layer->input_size + 1) {
              kto(layer->output_size) {
                ptr[0] = layer->coefficients[layer->row_size * k + j];
                ptr++;
              }
            }
          }
          factory->unmap_buffer(params_buffer);
          ctx->bind_storage_buffer(0, 1, params_buffer, 0, sizeof(float) * size);
        }
        // ito(nn->layers.size)
        {

          // BaseNNLayer *layer = nn->layers[i];
          // jto(layer->output_size)
          {
            pc.world_transform =
                translate(float4x4(1.0f), float3(2.2f * (nn->layers.size + 1), 0.0f, 0.0f));
            pc.g_exit_layer = -1;
            pc.g_exit_point = -1;
            ctx->push_constants(&pc, 0, sizeof(pc));
            // gizmo_layer->render_linebox(
            ctx->draw_indexed(36, 1, 0, 0, 0);
          }
        }
      }
      auto g_camera = gizmo_layer->get_camera();
      ito(nn->layers.size) {
        BaseNNLayer *layer = nn->layers[i];
        jto(layer->output_size) {
          float3 world_offset = float4(2.2f * (i + 1), 2.2f * j, 0.0f, 0.0f);
          gizmo_layer->render_linebox(world_offset - float3(1.0f, 1.0f, 1.0f),
                                      world_offset + float3(1.0f, 1.0f, 1.0f),
                                      float3(0.0f, 0.0f, 0.0f));
        }
      }
      {
        float dx = 1.0e-1f * g_camera.distance;
        gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
        gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{dx, 0.0f, 0.0f},
                                   dx * 0.04f, float3{1.0f, 0.0f, 0.0f});
        gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, dx, 0.0f},
                                   dx * 0.04f, float3{0.0f, 1.0f, 0.0f});
        gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, 0.0f, dx},
                                   dx * 0.04f, float3{0.0f, 0.0f, 1.0f});
      }
      gizmo_layer->render(factory, ctx);
      factory->end_render_pass(ctx);
    }
    if (g_config.get_bool("render network")) {
      auto           ctx = factory->start_compute_pass();
      String_Builder sb;
      sb.init();
      defer(sb.release());
      sb.putf(R"(
@(DECLARE_BUFFER
  (type READ_ONLY)
  (set 0)
  (binding 1)
  (type float)
  (name params)
)

float relu(float x) {
  return x < 0.0 ? 0.0 : x;
}
)");
      build_nn_shader(sb, nn);

      sb.putf(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4)  (name world_offset))
  (add_field (type float4)  (name scale))
  (add_field (type int)     (name g_exit_layer))
  (add_field (type int)     (name g_exit_point))
)

@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float3)    (name camera_pos))
  (add_field (type float3)    (name camera_look))
  (add_field (type float3)    (name camera_up))
  (add_field (type float3)    (name camera_right))
)

@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 2)
  (format RGBA32_FLOAT)
  (name out_image)
)

@(GROUP_SIZE 8 8 8)
@(ENTRY)
  int2 dim = imageSize(out_image);
  //if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
  //  return;
  float res = 0;
  float3 ro = float3(GLOBAL_THREAD_INDEX.xyz) * scale.xyz - float3_splat(1.0);
  eval_sdf(ro.x, ro.y, ro.z, res, g_exit_layer, g_exit_point);
  if (res > 0.0 && res < 1.0e-2) {
    int N    = 1;
/*float3 dr = 1.0 * scale.xyz / float(N * 2 + 1); 
for (int x = -N; x <= N; x++) {
for (int y = -N; y <= N; y++) {
for (int z = -N; z <= N; z++) {
ro = float3(x, y, z) * dr + float3(GLOBAL_THREAD_INDEX.xyz) * scale.xyz - float3_splat(1.0);
eval_sdf(ro.x, ro.y, ro.z, res, g_exit_layer, g_exit_point);
if (abs(res) > 1.0e-4)
continue;*/
          float4 pp = mul4(viewproj, float4(ro + world_offset.xyz, 1.0));
          pp.xyz /= pp.w;
          if (pp.x > 1.0 || pp.x < -1.0 || pp.y > 1.0 || pp.y < -1.0)
            return;
          i32 x = i32(0.5 + dim.x * (pp.x + 1.0) / 2.0);
          i32 y = i32(0.5 + dim.y * (pp.y + 1.0) / 2.0);
          for (int ix = x - 1; ix <= x + 1; ix++)
          for (int iy = y - 1; iy <= y + 1; iy++)
          image_store(out_image, int2(ix, iy), float4(0.0, 0.0, 0.0, 1.0));
/*
}
}
}
*/
    
  }
@(END)
)");
      ctx->CS_set_shader(factory->create_shader_raw(rd::Stage_t::COMPUTE, sb.get_str(), NULL, 0));
      struct PC {
        float4 world_offset;
        float4 scale;
        i32    g_exit_layer;
        i32    g_exit_point;
      } pc;
      struct Uniform {
        afloat4x4 viewproj;
        afloat3   camera_pos;
        afloat3   camera_look;
        afloat3   camera_up;
        afloat3   camera_right;
      };
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits          = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits        = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
        buf_info.size              = sizeof(Uniform);
        Resource_ID uniform_buffer = factory->create_buffer(buf_info);
        factory->release_resource(uniform_buffer);
        Uniform *ptr      = (Uniform *)factory->map_buffer(uniform_buffer);
        ptr->camera_pos   = gizmo_layer->get_camera().pos;
        ptr->camera_look  = gizmo_layer->get_camera().look;
        ptr->camera_up    = gizmo_layer->get_camera().up;
        ptr->camera_right = gizmo_layer->get_camera().right;
        ptr->viewproj     = gizmo_layer->get_camera().viewproj();
        factory->unmap_buffer(uniform_buffer);
        ctx->bind_uniform_buffer(0, 0, uniform_buffer, 0, sizeof(Uniform));
      }

      int size = 0;
      ito(nn->layers.size) {
        auto layer = nn->layers[i];
        jto(layer->input_size + 1) {
          kto(layer->output_size) { size++; }
        }
      }
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits         = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits       = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
      buf_info.size             = size * sizeof(float);
      Resource_ID params_buffer = factory->create_buffer(buf_info);
      factory->release_resource(params_buffer);
      float *ptr = (float *)factory->map_buffer(params_buffer);
      ito(nn->layers.size) {
        auto layer = nn->layers[i];
        jto(layer->input_size + 1) {
          kto(layer->output_size) {
            ptr[0] = layer->coefficients[layer->row_size * k + j];
            ptr++;
          }
        }
      }
      factory->unmap_buffer(params_buffer);
      ctx->bind_storage_buffer(0, 1, params_buffer, 0, sizeof(float) * size);
      ctx->image_barrier(normal_rt, (u32)rd::Access_Bits::SHADER_WRITE,
                         rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
      ctx->bind_rw_image(0, 2, 0, normal_rt, rd::Image_Subresource::top_level(),
                         rd::Format::NATIVE);
      ito(nn->layers.size) {
        BaseNNLayer *layer = nn->layers[i];
        jto(layer->output_size) {
          pc.world_offset = float4(2.2f * (i + 1), 2.2f * j, 0.0f, 0.0f);
          pc.scale        = float4(1.0f, 1.0f, 1.0f, 0.0f) / 16.0f;
          pc.g_exit_layer = i;
          pc.g_exit_point = j;
          ctx->push_constants(&pc, 0, sizeof(pc));
          ctx->dispatch(4, 4, 4);
        }
      }
      factory->end_compute_pass(ctx);
    }
  }
  void release(rd::IDevice *factory) { factory->release_resource(normal_rt); }
};

class Event_Consumer : public IGUI_Pass {
  GBufferPass gbuffer_pass;

  public:
  void init(rd::Pass_Mng *pmng) override { //
    IGUI_Pass::init(pmng);
  }
  void init_traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      init_traverse(l->child);
      init_traverse(l->next);
    } else {
      if (l->cmp_symbol("camera")) {
        gizmo_layer->get_camera().traverse(l->next);
      } else if (l->cmp_symbol("config")) {
        g_config.traverse(l->next);
      } else if (l->cmp_symbol("scene")) {
        g_scene->restore(l);
      }
    }
  }

  void on_gui_traverse_nodes(List *l, int &id) {
    if (l == NULL) return;
    id++;
    ImGui::PushID(id);
    defer(ImGui::PopID());
    if (l->child) {
      ImGui::Indent();
      on_gui_traverse_nodes(l->child, id);
      ImGui::Unindent();
      on_gui_traverse_nodes(l->next, id);
    } else {
      if (l->next == NULL) return;
      if (l->cmp_symbol("scene")) {
        on_gui_traverse_nodes(l->next, id);
        return;
      } else if (l->cmp_symbol("node")) {
        ImGui::LabelText("Node", "%.*s", STRF(l->get(1)->symbol));
        on_gui_traverse_nodes(l->get(2), id);
        return;
      }

      string_ref  type = l->next->symbol;
      char const *name = stref_to_tmp_cstr(l->symbol);
      if (type == stref_s("float3")) {
        float x    = l->get(2)->parse_float();
        float y    = l->get(3)->parse_float();
        float z    = l->get(4)->parse_float();
        float f[3] = {x, y, z};
        if (ImGui::DragFloat3(name, (float *)&f[0], 1.0e-2f)) {
          // if (f[0] != x) {
          // DebugBreak();
          //}
          l->get(2)->symbol = tmp_format("%f", f[0]);
          l->get(3)->symbol = tmp_format("%f", f[1]);
          l->get(4)->symbol = tmp_format("%f", f[2]);
        }
      } else if (type == stref_s("model")) {
        ImGui::LabelText("model", stref_to_tmp_cstr(l->get(2)->symbol));
      } else {
        UNIMPLEMENTED;
      }
    }

    // ImGui::LabelText("Name", "%.*s", STRF(node->get_name()));
    // ImGui::DragFloat3("Offset", (float *)&node->offset, 1.0e-2f);
    // glm::vec3 euler = glm::eulerAngles(node->rotation);
    // euler *= 180.0f / PI;
    // ImGui::DragFloat3("Rotation", (float *)&euler, 1.0e-1f);
    // euler *= PI / 180.0f;
    //// float EPS = 1.0e-3f;
    //// if (euler.x < 2.0f * PI + EPS) euler.x += 2.0f * PI;
    //// if (euler.x > 2.0f * PI - EPS) euler.x -= 2.0f * PI;
    //// if (euler.y < 2.0f * PI + EPS) euler.y += 2.0f * PI;
    //// if (euler.y > 2.0f * PI - EPS) euler.y -= 2.0f * PI;
    //// if (euler.z < 2.0f * PI + EPS) euler.z += 2.0f * PI;
    //// if (euler.z > 2.0f * PI - EPS) euler.z -= 2.0f * PI;
    // node->rotation = glm::quat(glm::vec3(euler.x, euler.y, euler.z));
    //// quat(euler.x, float3(1.0f, 0.0f, 0.0f)) *
    //// quat(euler.y, float3(0.0f, 1.0f, 0.0f)) *
    //// quat(euler.z, float3(0.0f, 0.0f, 1.0f));
    // if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
    //}

    // ito(node->get_children().size) {
    //  if (node->get_children()[i]) on_gui_traverse_nodes(node->get_children()[i]);
    //}
  }

  void on_gui(rd::IDevice *factory) override { //
    // bool show = true;
    // ShowExampleAppCustomNodeGraph(&show);
    // ImGui::TestNodeGraphEditor();
    // ImGui::Begin("Text");
    // te.Render("Editor");
    // ImGui::End();
    ImGui::Begin("Scene");
    {
      String_Builder sb;
      sb.init();
      defer(sb.release());
      g_scene->save(sb);
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(sb.get_str(), Tmp_List_Allocator());
      if (cur) {
        int id = 0;
        on_gui_traverse_nodes(cur, id);
        g_scene->restore(cur);
      }
    }
    ImGui::End();
    ImGui::Begin("Config");
    g_config.on_imgui();
// ImGui::LabelText("clear pass", "%f ms", hr.clear_timestamp.duration);
// ImGui::LabelText("pre  pass", "%f ms", hr.prepass_timestamp.duration);
// ImGui::LabelText("resolve pass", "%f ms", hr.resolve_timestamp.duration);
#if 1
    if (1) {
      static PCG pcg;

      static float rate             = 0.03f;
      static float regrate          = 0.00f;
      static float regrate2         = 0.00f;
      static int   inters_per_frame = 1000;
      float        res;

      ito(inters_per_frame) {
        int iter = (pcg.next()) % td.size;
        nn->eval(&td[iter].x, &res);
        float error = (td[iter].dist - res);
        nn->solve(&error, rate / 10.0f);
        nn->regulateL1(regrate / 1000.0f);
        nn->regulateL2(regrate2 / 1000.0f);
      }
      ImGui::DragFloat("learning rate", &rate, 1.0e-3f);
      ImGui::DragFloat("reg rate L1", &regrate, 1.0e-3f);
      ImGui::DragFloat("reg rate L2", &regrate2, 1.0e-3f);

      // ImGui::DragFloat("error", &error);
      ImGui::DragInt("iters per frame", &inters_per_frame);
      if (ImGui::Button("reset")) nn->reset();
    }
#endif
    ImGui::End();
    ImGui::Begin("main viewport");
    gizmo_layer->per_imgui_window();
    auto wsize = get_window_size();
    ImGui::Image(bind_texture(gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                 ImVec2(wsize.x, wsize.y));
    auto wpos = ImGui::GetCursorScreenPos();
    // auto iinfo      = factory->get_image_info(hr.hair_img);
    // g_camera.aspect = float(iinfo.height) / iinfo.width;

    ImGui::End();
    //{
    //  ImGui::Begin("NN");
    //
    //  ImGui::End();
    //}
  }
  void on_init(rd::IDevice *factory) override { //
    TMP_STORAGE_SCOPE;
    gizmo_layer = Gizmo_Layer::create(factory);
    // new XYZDragGizmo(gizmo_layer, &pos);
    g_config.init(stref_s(R"(
(
 (add u32  g_buffer_width 512 (min 4) (max 1024))
 (add u32  g_buffer_height 512 (min 4) (max 1024))
 (add bool forward 1)
 (add bool "render SDF" 1)
 (add bool "render network" 1)
 (add bool "render samples" 1)
 (add bool "depth test" 1)
 (add f32  strand_size 1.0 (min 0.1) (max 16.0))
)
)"));

    // g_scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/monkey.gltf"));
    g_scene->load_mesh(stref_s("mesh"), stref_s("models/scene.gltf"));

    gbuffer_pass.init();

    List *l = List::parse(stref_s(R"(
  (layer 3 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 1 Tanh)
  )"),
                          Tmp_List_Allocator{});
    ASSERT_ALWAYS(l);
    nn = NN::create(l);

    td.init(N);
    // defer(td.release());
    PCG            pcg;
    Random_Factory rf;
    // Image2D *img = load_image(stref_s("images/101.png"));
    // assert(img);
    // defer(img->release());
    g_scene->update();
    // thread_pool.init_thread_pool();
    if (1) {
      g_scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            if (mn->getComponent<GfxSufraceComponent>() == NULL) {
              GfxSufraceComponent::create(factory, mn);
            }
            auto *c       = mn->getComponent<GfxSufraceComponent>();
            float size    = 0.0f;
            auto  aabb    = mn->getAABB();
            size          = MAX3(abs(aabb.max.x - aabb.min.x), abs(aabb.max.y - aabb.min.y),
                        abs(aabb.max.z - aabb.min.z));
            auto   t      = mn->get_transform();
            int    N      = 128;
            float  dx     = (mn->getAABB().max - mn->getAABB().min).x / N;
            float  dy     = (mn->getAABB().max - mn->getAABB().min).y / N;
            float  dz     = (mn->getAABB().max - mn->getAABB().min).z / N;
            float3 origin = mn->getAABB().min;
            zto(N) {
              yto(N) {
                xto(N) {
                  float    rx = origin.x + dx * x;
                  float    ry = origin.x + dy * y;
                  float    rz = origin.x + dz * z;
                  float3   p(rx, ry, rz);
                  TestData item;
                  item.x    = p.x;
                  item.y    = p.y;
                  item.z    = p.z;
                  item.dist = c->getBVH()->distance(p);
                  td.push(item);
                }
              }
            }
            // ito(mn->getNumSurfaces()) {
            //  if (mn->getSurface(i)->mesh.num_indices < 3) continue;
            //  int N = (1 << 17) / (mn->getSurface(i)->mesh.num_indices / 3);
            //  if (N < 1) {
            //    abort();
            //  }
            //  kto(mn->getSurface(i)->mesh.num_indices / 3) {

            //    Triangle_Full ftri = mn->getSurface(i)->mesh.fetch_triangle(k);

            //    float3 v0 = ftri.v0.position;
            //    float3 v1 = ftri.v1.position;
            //    float3 v2 = ftri.v2.position;

            //    jto(N) {
            //      float t0 = pcg.nextf();
            //      float t1 = pcg.nextf();
            //      if (t1 > 1.0 - t0) {
            //        t1 = 1.0 - t1;
            //        t0 = 1.0 - t0;
            //      }
            //      float  t2  = 1.0f - t0 - t1;
            //      float3 v   = v0 * t0 + v1 * t1 + v2 * t2;
            //      v          = transform(t, v);
            //      float3   p = v + rf.rand_sphere_center_r3() * size / 1.0f;
            //      TestData item;
            //      item.x    = p.x;
            //      item.y    = p.y;
            //      item.z    = p.z;
            //      item.dist = c->getBVH()->distance(p);
            //      td.push(item);
            //    }
            //  }

            //}
          }
        }
      });
    }
    char *state = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }

    // float a = pcg.nextf(); // * 2.0f - 1.0f;
    // float b = pcg.nextf(); /// * 2.0f - 1.0f;
    //// float c = a > b ? 1.0 : 0.0;
    //// float2 uv = float2(a, 1.0 - b);
    //// float  c  = img->sample(uv).r;
    //// pow(abs(1.0 - abs(1.0 - sqrt(a * a + b * b))), 4.0f);
    //// a > 0.0f ? (b > 0.0 ? 1.0f : -1.0f) : (b > 0.0f ? -1.0f : 1.0f);
    //// sin(a*6.0f)  + cos(b*6.0f);
    // td.push({a, b, c});

    // ito(N) {
    //  float res;
    //  nn->eval(&td[i].a, &res);
    //  float error = td[i].c - res;
    //  nn->solve(&error, 0.003f);
    //  // nn->regulateL1(0.001f);
    //  fprintf(stdout, "error=[%f]\n", error);
    //}
    // defer(nn->release());
  }
  void on_release(rd::IDevice *factory) override { //
    // thread_pool.release();
    FILE *scene_dump = fopen("scene_state", "wb");
    fprintf(scene_dump, "(\n");
    defer(fclose(scene_dump));
    gizmo_layer->get_camera().dump(scene_dump);
    g_config.dump(scene_dump);
    {
      String_Builder sb;
      sb.init();
      g_scene->save(sb);
      fwrite(sb.get_str().ptr, 1, sb.get_str().len, scene_dump);
      sb.release();
    }
    fprintf(scene_dump, ")\n");
    gizmo_layer->release();
    g_scene->release();
    IGUI_Pass::release(factory);
  }
  void consume(void *_event) override { //
    IGUI_Pass::consume(_event);
  }
  void on_frame(rd::IDevice *factory) override { //
    g_scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GizmoComponent>() == NULL) {
          GizmoComponent::create(gizmo_layer, mn);
        }
      }
    });
    if (g_config.get_bool("render samples")) {
      float stride = 2.2;
      ito(td.size) {
        gizmo_layer->draw_sphere(float3{td[i].x, td[i].y, td[i].z}, 0.002f,
                                 float3{td[i].dist, 0.0f, -td[i].dist});
      }
      float3 offset = float3(stride, 0.0f, 0.0f);
      if (0) {
        zto(33) {
          yto(33) {
            xto(33) {
              float _x = (float(x) / 32.0) * 2.0f - 1.0f;
              float _y = (float(y) / 32.0) * 2.0f - 1.0f;
              float _z = (float(z) / 32.0) * 2.0f - 1.0f;
              float res;
              float in[] = {_x, _y, _z};
              nn->eval(in, &res);
              float EPS = 2.0e-2;
              // jto(nn->layers.size)
              {
                u32 j = nn->layers.size - 1;
                // kto(nn->layers[j]->getOutputSize())
                {
                  u32   k = nn->layers[j]->getOutputSize() - 1;
                  float c = nn->layers[j]->getOutput(k);
                  if (abs(c) < EPS)
                    gizmo_layer->draw_sphere(offset + float3{_x + stride * j, _y + stride * k, _z},
                                             0.04f, float3{c, 0.0f, -c});
                }
              }
            }
          }
        }
      }
    }
    // nn->on_gui(gizmo_layer);
    g_scene->get_root()->update();
    gbuffer_pass.render(factory);
    IGUI_Pass::on_frame(factory);
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  IGUI_Pass *   gui  = new Event_Consumer;
  gui->init(pmng);
  pmng->set_event_consumer(gui);
  pmng->loop();
  return 0;
}