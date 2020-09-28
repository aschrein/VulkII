#include "rendering.hpp"
#include "rendering_utils.hpp"

#include <imgui.h>

Config       g_config;
Scene *      g_scene     = Scene::create();
Gizmo_Layer *gizmo_layer = NULL;

template <typename T> class GPUBuffer {
  private:
  rd::IFactory *factory;
  Array<T>      cpu_array;
  Resource_ID   gpu_buffer;
  Resource_ID   cpu_buffer;
  size_t        gpu_buffer_size;

  public:
  void init(rd::IFactory *factory) {
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
  void        flush(rd::Imm_Ctx *ctx = NULL) {
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

class GBufferPass {
  public:
  Resource_ID normal_rt;
  Resource_ID depth_rt;

  public:
  void init() { MEMZERO(*this); }
  void render(rd::IFactory *factory) {
    g_scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GfxSufraceComponent>() == NULL) {
          GfxSufraceComponent::create(factory, mn);
        }
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
      rt0_info.format     = rd::Format::D32_FLOAT;
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

      rd::Imm_Ctx *ctx = factory->start_render_pass(info);
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
      auto g_camera = gizmo_layer->get_camera();
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
  }
  void release(rd::IFactory *factory) { factory->release_resource(normal_rt); }
};

template <typename T = f32> struct ReLuActivation {
  static T f(T i) { return i > (T)0 ? i : (T)0; }
  static T deriv(T i) { return i > (T)0 ? (T)1 : (T)0; }
};

template <typename T = f32> struct TanhActivation {
  static T f(T i) { return std::tanh(i); }
  static T deriv(T i) {
    T a = std::tanh(i);
    return (T)1 - a * a;
  }
};

template <typename T = f32> struct NoOpActivation {
  static T f(T i) { return i; }
  static T deriv(T i) { return (T)1; }
};

template <typename T = f32> struct DelayActivation {
  static constexpr T eps = 1.0e-1f;
  static T           f(T i) { return i < -eps ? -eps : (i > eps ? eps : i / eps); }
  static T           deriv(T i) { return i < -eps ? (T)0 : (i > eps ? (T)0 : (T)1 / eps); }
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
  u32         getOutputSize() { return output_size; }
  f32         getOutput(i32 i) { return f(outputs[i]); }
  virtual f32 f(f32 x)     = 0;
  virtual f32 deriv(f32 x) = 0;
  void        init(f32 const *in) { ito(input_size) inputs[i] = in[i]; }
  void        eval() {
    ito(outputs.size) {
      f32 result = (f32)0;
      jto(inputs.size) { result += inputs[j] * coefficients[row_size * i + j]; }
      outputs[i] = result;
      // fprintf(stdout, "[%f] ", result);
    };
  }
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
    inputs[input_size] = get_pcg().nextf();
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
  f32 f(f32 x) override { return F::f(x); }
  f32 deriv(f32 x) override { return F::deriv(x); }
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
        } else if (activation == eDelay) {
          layers.push(new NNLayer<DelayActivation<f32>>(num_inputs, num_outputs));
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
class Event_Consumer : public IGUI_Pass {
  GBufferPass gbuffer_pass;
  float3      pos;
  NN *        nn = NULL;
  struct TestData {
    float a, b;
    float c;
  };
  int             N = 1000;
  Array<TestData> td;

  public:
  void init(rd::Pass_Mng *pmng) override { //
    IGUI_Pass::init(pmng);
    TMP_STORAGE_SCOPE;
    List *l = List::parse(stref_s(R"(
  (layer 2 4 Tanh)
  (layer 4 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 8 Tanh)
  (layer 8 4 Tanh)
  (layer 4 1 Tanh)
  )"),
                          Tmp_List_Allocator{});
    ASSERT_ALWAYS(l);
    nn = NN::create(l);

    td.init(N);
    // defer(td.release());
    PCG pcg;
    ito(N) {
      float a = pcg.nextf() * 2.0f - 1.0f;
      float b = pcg.nextf() * 2.0f - 1.0f;
      // float c = a > b ? 1.0 : 0.0;
      float c = //pow(abs(1.0 - abs(1.0 - sqrt(a * a + b * b))), 4.0f);
       //a > 0.0f ? (b > 0.0 ? 1.0f : -1.0f) : (b > 0.0f ? -1.0f : 1.0f);
        sin(a*1.0f)  + cos(b*1.0f);
      td.push({a, b, c});
    }
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

  void on_gui(rd::IFactory *factory) override { //
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
    {
      static PCG pcg;

      static float rate             = 1.f;
      static float regrate          = 0.00f;
      static int   inters_per_frame = 1000;
      float        res;

      ito(inters_per_frame) {
        int iter = (pcg.next()) % td.size;
        nn->eval(&td[iter].a, &res);
        float error = (td[iter].c - res);
        nn->solve(&error, rate / 10.0f);
        nn->regulateL1(regrate / 10.0f);
      }
      ImGui::DragFloat("learning rate", &rate, 1.0e-3f);
      ImGui::DragFloat("reg rate", &regrate, 1.0e-3f);

      // ImGui::DragFloat("error", &error);
      ImGui::DragInt("iters per frame", &inters_per_frame);
      if (ImGui::Button("reset")) nn->reset();
    }
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
  void on_init(rd::IFactory *factory) override { //
    TMP_STORAGE_SCOPE;
    gizmo_layer = Gizmo_Layer::create(factory);
    // new XYZDragGizmo(gizmo_layer, &pos);
    g_config.init(stref_s(R"(
(
 (add u32  g_buffer_width 512 (min 4) (max 1024))
 (add u32  g_buffer_height 512 (min 4) (max 1024))
 (add bool forward 1)
 (add bool "depth test" 1)
 (add f32  strand_size 1.0 (min 0.1) (max 16.0))
)
)"));

    // g_scene->load_mesh(stref_s("mesh"), stref_s("models/human_skull_and_neck/scene.gltf"));
    char *state = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
    gbuffer_pass.init();
  }
  void on_release(rd::IFactory *factory) override { //
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
  void on_frame(rd::IFactory *factory) override { //
    g_scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        if (mn->getComponent<GizmoComponent>() == NULL) {
          GizmoComponent::create(gizmo_layer, mn);
        }
      }
    });
    {
      float stride = 2.2;
      ito(td.size) {
        gizmo_layer->draw_sphere(float3{td[i].a - stride, td[i].b, 0.0f}, 0.2f,
                                 float3{td[i].c, 0.0f, -td[i].c});
      }

      yto(33) {
        xto(33) {
          float _x = (float(x) / 32.0) * 2.0f - 1.0f;
          float _y = (float(y) / 32.0) * 2.0f - 1.0f;
          float res;
          float in[] = {_x, _y};
          nn->eval(in, &res);

          jto(nn->layers.size) {
            kto(nn->layers[j]->getOutputSize()) {
              float c = nn->layers[j]->getOutput(k);
              gizmo_layer->draw_sphere(float3{_x + stride * j, _y + stride * k, 0.0f}, 0.2f,
                                       float3{c, 0.0f, -c});
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
