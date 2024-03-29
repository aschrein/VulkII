#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC
//#include "marching_cubes/marching_cubes.h"
#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "script.hpp"

#include <atomic>
//#include <functional>
#include <3rdparty/half.hpp>
#include <condition_variable>
#include <imgui.h>
#include <mutex>
#include <thread>

#include <embree3/rtcore_builder.h>

struct RenderingContext {
  rd::IDevice *factory     = NULL;
  Config *     config      = NULL;
  Scene *      scene       = NULL;
  Gizmo_Layer *gizmo_layer = NULL;
  void         dump() {
    FILE *scene_dump = fopen("scene_state", "wb");
    fprintf(scene_dump, "(\n");
    defer(fclose(scene_dump));
    if (gizmo_layer) gizmo_layer->get_camera().dump(scene_dump);
    config->dump(scene_dump);
    if (scene) {
      String_Builder sb;
      sb.init();
      scene->save(sb);
      fwrite(sb.get_str().ptr, 1, sb.get_str().len, scene_dump);
      sb.release();
    }
    fprintf(scene_dump, ")\n");
  }
};

struct GBuffer {
  Resource_ID normal;
  Resource_ID depth;
};

class BakerPass {
  public:
  static constexpr char const *NAME = "Baking Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(pass);                                                                                  \
  RESOURCE(frame_buffer);                                                                          \
  RESOURCE(position_rt);                                                                           \
  RESOURCE(normal_rt);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  u32                         width  = 0;
  u32                         height = 0;
  bool                        dirty  = true;
  rd::Render_Pass_Create_Info info{};
  rd::Graphics_Pipeline_State gfx_state{};

  public:
  void release(rd::IDevice *factory) {
    timestamps.release(factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  void init(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.init(dev);
    Resource_ID vs = dev->create_shader(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
  float4x4 world_transform;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
  [[vk::location(3)]] float3 src_pos : TEXCOORD2;
};

struct VSInput {
  [[vk::location(0)]] float3 pos     : POSITION;
  [[vk::location(1)]] float3 normal  : NORMAL;
  [[vk::location(2)]] float2 uv      : TEXCOORD0;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.normal  = mul(pc.world_transform, float4(input.normal.xyz, 0.0f)).xyz;
  output.uv      = input.uv;
  output.pos     = float4(input.uv * 2.0 - 1.0, 0.0f, 1.0f);
  //output.pos     = float4(input.uv, 0.0f, 1.0f);
  output.src_pos = input.pos;
  return output;
}
)"),
                                        NULL, 0);
    Resource_ID ps = dev->create_shader(rd::Stage_t::PIXEL, stref_s(R"(
struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
  [[vk::location(3)]] float3 src_pos : TEXCOORD2;
};

struct PSOut {
  float4 rt0 : SV_TARGET0;
  float4 rt1 : SV_TARGET1;
};

PSOut main(in PSInput input) {
  PSOut ps_out;
  ps_out.rt0 = float4(input.src_pos.xyz, 1.0f);
  ps_out.rt1 = float4(input.normal.xyz, 1.0f);
  return ps_out;
}
)"),
                                        NULL, 0);
    dev->release_resource(vs);
    dev->release_resource(ps);
    signature = [=] {
      rd::Binding_Space_Create_Info set_info{};
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    pass = [=] {
      rd::Render_Pass_Create_Info info{};
      rd::RT_Ref                  rt0{};
      rt0.format            = rd::Format::RGBA32_FLOAT;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);

      rt0.format            = rd::Format::RGBA32_FLOAT;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);

      info.depth_target.enabled = false;
      return dev->create_render_pass(info);
    }();

    pso = [=] {
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state{};
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = false;
      ds_state.enable_depth_write = false;
      gfx_state.DS_set_state(ds_state);
      rd::Blend_State bs{};
      bs.enabled = false;
      bs.color_write_mask =
          (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
          (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
      gfx_state.OM_set_blend_state(0, bs);
      gfx_state.OM_set_blend_state(1, bs);
      gfx_state.VS_set_shader(vs);
      gfx_state.PS_set_shader(ps);
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 1;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 1;
        info.offset   = 0;
        info.type     = rd::Attriute_t::NORMAL;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 2;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 2;
        info.offset   = 0;
        info.type     = rd::Attriute_t::TEXCOORD0;
        gfx_state.IA_set_attribute(info);
      }
      gfx_state.IA_set_vertex_binding(0, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(1, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(2, 8, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();
  }
  void update_frame_buffer(RenderingContext rctx) {
    auto dev = rctx.factory;
    if (frame_buffer.is_valid()) dev->release_resource(frame_buffer);
    if (position_rt.is_valid()) dev->release_resource(position_rt);
    if (normal_rt.is_valid()) dev->release_resource(normal_rt);

    position_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = Image2D::get_num_mip_levels(width, height);
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();

    normal_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = Image2D::get_num_mip_levels(width, height);
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
    frame_buffer = [=] {
      rd::Frame_Buffer_Create_Info info{};
      rd::RT_View                  rt0{};
      rt0.image  = position_rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);
      rt0.image = normal_rt;
      info.rts.push(rt0);

      info.depth_target.enabled = false;
      return dev->create_frame_buffer(pass, info);
    }();
  }
  void render(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.update(dev);
    u32 width  = rctx.config->get_u32("baking.size");
    u32 height = rctx.config->get_u32("baking.size");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
      dirty = true;
    }
    if (dirty == false) return;
    dirty = false;
    struct PushConstants {
      float4x4 viewproj;
      float4x4 world_transform;
    } pc;

    float4x4 viewproj = rctx.gizmo_layer->get_camera().viewproj();

    rd::ICtx *ctx = dev->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "Baking Pass");
      timestamps.begin_range(ctx);
      ctx->start_render_pass();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      pc.viewproj = viewproj;

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      table->push_constants(&viewproj, 0, sizeof(float4x4));
      ctx->bind_table(table);
      ctx->bind_graphics_pso(pso);
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
          float4x4             world = mn->get_transform();
          pc.world_transform         = world;
          table->push_constants(&pc, 0, sizeof(pc));
          ito(gs->getNumSurfaces()) {
            GfxSurface *s = gs->getSurface(i);
            s->draw(ctx, gfx_state);
          }
        }
      });
      ctx->end_render_pass();
      timestamps.end_range(ctx);
    }

    Resource_ID e = dev->end_render_pass(ctx);
    Mip_Builder::create_image(dev, position_rt);
    Mip_Builder::create_image(dev, normal_rt);
    timestamps.commit(e);
  }
};

class BufferThing {
#define RESOURCE_LIST                                                                              \
  RESOURCE(cs0);                                                                                   \
  RESOURCE(cs1);                                                                                   \
  RESOURCE(cs2);                                                                                   \
  RESOURCE(buffer);                                                                                \
  RESOURCE(buffer1);                                                                               \
  RESOURCE(readback);                                                                              \
  RESOURCE(signature);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  public:
  void init(rd::IDevice *dev) {
    buffer = [dev] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = sizeof(u32) * 16 * 1024 * 1024;
      return dev->create_buffer(buf_info);
    }();
    // Allocate a buffer.
    buffer1 = [dev] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = sizeof(u32) * 16 * 1024 * 1024;
      return dev->create_buffer(buf_info);
    }();

    signature = [dev] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
      set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = 4;
      return dev->create_signature(table_info);
    }();

    cs0 = dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x % 1024;
    for (uint i = 0; i < 100; i++)
      BufferOut.Store<uint>(id * 4, BufferOut.Load<uint>(id * 4) + 1);
}
)"),
                                                                NULL, 0));
    cs1 = dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer BufferIn : register(u1, space0);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x % 1024;
    for (uint i = 0; i < 100; i++)
      BufferOut.Store<uint>(id * 4, BufferOut.Load<uint>(id * 4) + 1);
}
)"),
                                                                NULL, 0));
    cs2 = dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut: register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer BufferIn : register(u1, space0);
  
struct CullPushConstants
{
  uint val;
};
[[vk::push_constant]] ConstantBuffer<CullPushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;
  
[numthreads(64, 1, 1)]
  void main(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x % 1024;
    for (uint i = 0; i < 100; i++)
      BufferOut.Store<uint>(id * 4, BufferOut.Load<uint>(id * 4) * pc.val);
}
)"),
                                                                NULL, 0));
  }
  void test_buffers(rd::IDevice *dev) {
    Resource_ID wevent_0{};
    Resource_ID wevent_1{};
    {
      rd::ICtx *ctx = dev->start_async_compute_pass();
      {
        TracyVulkIINamedZone(ctx, "Async Compute Example 1");
        ctx->bind_compute(cs0);
        rd::IBinding_Table *table = dev->create_binding_table(signature);
        defer(table->release());
        table->bind_UAV_buffer(0, 0, buffer, 0, sizeof(u32) * 1024);
        table->bind_UAV_buffer(0, 1, buffer, 0, sizeof(u32) * 1024);
        ctx->bind_table(table);
        ctx->dispatch(1024 * 32, 1, 1);
      }
      wevent_0 = dev->end_async_compute_pass(ctx);
    }
    // dev->wait_idle();
    u32 val = 2;
    {
      rd::ICtx *ctx = dev->start_async_compute_pass();
      {
        TracyVulkIINamedZone(ctx, "Async Compute Example 2");
        // ctx->wait_for_event(wevent_0);
        ctx->bind_compute(cs1);
        rd::IBinding_Table *table = dev->create_binding_table(signature);
        defer(table->release());
        table->bind_UAV_buffer(0, 0, buffer1, 0, sizeof(u32) * 1024);
        table->bind_UAV_buffer(0, 1, buffer1, 0, sizeof(u32) * 1024);
        ctx->bind_table(table);
        // ctx->buffer_barrier(buffer, rd::Buffer_Access::UAV);
        ctx->dispatch(1024 * 32, 1, 1);
      }
      wevent_1 = dev->end_async_compute_pass(ctx);
    }
    {
      rd::ICtx *ctx = dev->start_async_copy_pass();
      {
        // ctx->wait_for_event(wevent_0);
        // ctx->wait_for_event(wevent_1);
        TracyVulkIINamedZone(ctx, "Async Copy Example");
        ctx->copy_buffer(buffer, 0, buffer1, 0, sizeof(u32) * 16 * 1024 * 1024);
      }
      wevent_1 = dev->end_async_copy_pass(ctx);
    }
  }
  void release(rd::IDevice *factory) {
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
};

class GIComputeGBufferPass {
  public:
  static constexpr char const *NAME = "G.I. Compute GBuffer Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(sampler_state);                                                                         \
  RESOURCE(clear_pso);                                                                             \
  RESOURCE(prepare_indirect_pso);                                                                  \
  RESOURCE(indirect_args_buffer);                                                                  \
  RESOURCE(pso);                                                                                   \
  RESOURCE(counter_grid_0);                                                                        \
  RESOURCE(counter_grid_1);                                                                        \
  RESOURCE(normal_rt);                                                                             \
  RESOURCE(depth_rt);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  u32 grid_id = 0;
  u32 width   = 0;
  u32 height  = 0;

  Resource_ID prev_grid{};
  Resource_ID cur_grid{};

  public:
#include "examples/shaders/declarations.hlsl"

  TimeStamp_Pool timestamps = {};

  void init(RenderingContext rctx) {
    TMP_STORAGE_SCOPE;
    auto dev = rctx.factory;
    timestamps.init(dev);
    sampler_state = [&] {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::LINEAR;
      info.mip_mode       = rd::Filter::LINEAR;
      info.max_lod        = 1000.0f;
      info.anisotropy     = true;
      info.max_anisotropy = 16.0f;
      return dev->create_sampler(info);
    }();
    signature = [=] {
      rd::Binding_Table_Create_Info table_info{};
      {
        rd::Binding_Space_Create_Info set_info{};
        set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
        set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
        set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
        set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
        set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
        set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
        set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
        set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
        table_info.spaces.push(set_info);
      }
      {
        rd::Binding_Space_Create_Info set_info{};
        set_info.bindings.push({rd::Binding_t::UNIFORM_BUFFER, 1});
        table_info.spaces.push(set_info);
      }
      {
        rd::Binding_Space_Create_Info set_info{};
        set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
        table_info.spaces.push(set_info);
      }
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();

    Resource_ID cs{};
    pso = dev->create_compute_pso(
        signature, cs = dev->create_shader(
                       rd::Stage_t::COMPUTE,
                       read_file_tmp_stref("examples/shaders/gi_compute_rasterization.hlsl"), NULL,
                       0, stref_s("main_tri_per_lane")));
    dev->release_resource(cs);

    indirect_args_buffer = [&] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_INDIRECT_ARGUMENTS |
                            (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
      buf_info.size =
          sizeof(rd::Dispatch_Indirect_Args) * COUNTER_GRID_RESOLUTION * COUNTER_GRID_RESOLUTION;
      return dev->create_buffer(buf_info);
    }();

    cs                   = {};
    prepare_indirect_pso = dev->create_compute_pso(
        signature, cs = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
struct IndirectArgs {
  i32 dimx;
  i32 dimy;
  i32 dimz;
};

struct PushConstants {
  u32  min_triangle_size;
  u32  max_triangle_size;
  u32  min_resolution;
  u32  max_resolution;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

[[vk::binding(4, 0)]] RWTexture2D<uint>  counter_grid         : register(u4, space0);
[[vk::binding(6, 0)]] RWByteAddressBuffer indirect_arg_buffer : register(u6, space0);

[numthreads(16, 16, 1)]
  void main(uint3 tid : SV_DispatchThreadID)
{
  u32 width, height;
  counter_grid.GetDimensions(width, height);
  if (tid.x >= width || tid.y >= height)
    return;
  u32 cnt = counter_grid[tid.xy];
  IndirectArgs args = indirect_arg_buffer.Load<IndirectArgs>(12 * (tid.x + tid.y * width));
  if (args.dimx == 0) {
    args.dimx = 1;
  } else if (cnt > pc.max_triangle_size) {
    args.dimx = args.dimx + 1;
  } else if (cnt < pc.min_triangle_size) {
    args.dimx = args.dimx - 1;
  }
  args.dimx = max(pc.min_resolution, min(pc.max_resolution, args.dimx));
  args.dimy = args.dimx;
  args.dimz = 1;
  indirect_arg_buffer.Store<IndirectArgs>(12 * (tid.x + tid.y * width), args);
}
)"),
                                           NULL, 0));
    dev->release_resource(cs);
    cs        = {};
    clear_pso = dev->create_compute_pso(signature,
                                        cs = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
struct PushConstants {
  u32      clear_counter_grid;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

[[vk::binding(2, 0)]] RWTexture2D<float4> normal_target  : register(u2, space0);
[[vk::binding(3, 0)]] RWTexture2D<uint>  depth_target    : register(u3, space0);
[[vk::binding(4, 0)]] RWTexture2D<uint>  counter_grid    : register(u4, space0);

[numthreads(16, 16, 1)]
  void main(uint3 tid : SV_DispatchThreadID)
{
  if (pc.clear_counter_grid) {
    uint width, height;
    counter_grid.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
      return;
    counter_grid[tid.xy]  = 0;
  } else {

    uint width, height;
    normal_target.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
      return;
    normal_target[tid.xy] = float4_splat(0.0f);
    depth_target[tid.xy]  = 0;
  }
}
)"),
                                                                NULL, 0));
    dev->release_resource(cs);
    counter_grid_0 = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format = rd::Format::R32_UINT;
      rt0_info.width  = COUNTER_GRID_RESOLUTION;
      rt0_info.height = COUNTER_GRID_RESOLUTION;
      rt0_info.depth  = 1;
      rt0_info.layers = 1;
      rt0_info.levels = 1;
      rt0_info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
    counter_grid_1 = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format = rd::Format::R32_UINT;
      rt0_info.width  = COUNTER_GRID_RESOLUTION;
      rt0_info.height = COUNTER_GRID_RESOLUTION;
      rt0_info.depth  = 1;
      rt0_info.layers = 1;
      rt0_info.levels = 1;
      rt0_info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
  }
  void update_frame_buffer(RenderingContext rctx) {
    auto dev = rctx.factory;
    if (normal_rt.is_valid()) dev->release_resource(normal_rt);
    if (depth_rt.is_valid()) dev->release_resource(depth_rt);

    normal_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
    depth_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::R32_UINT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
  }
  void render(RenderingContext rctx, Resource_ID pos_tex, Resource_ID normal_tex) {
    auto dev = rctx.factory;
    timestamps.update(dev);
    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    float4x4       viewproj = rctx.gizmo_layer->get_camera().viewproj();
    FrameConstants fc{};
    fc.viewproj         = viewproj;
    Resource_ID cbuffer = [&] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size        = sizeof(fc);
      return dev->create_buffer(buf_info);
    }();
    dev->release_resource(cbuffer);
    {
      memcpy(dev->map_buffer(cbuffer), &fc, sizeof(fc));
      dev->unmap_buffer(cbuffer);
    }
    if (grid_id == 0) {
      cur_grid  = counter_grid_0;
      prev_grid = counter_grid_1;
    } else {
      cur_grid  = counter_grid_1;
      prev_grid = counter_grid_0;
    }
    grid_id = (grid_id + 1) & 1;
    {
      rd::ICtx *ctx = dev->start_compute_pass();
      {
        TracyVulkIINamedZone(ctx, "G.I. Compute Clear GBuffer Pass");
        ctx->bind_compute(clear_pso);
        rd::IBinding_Table *table = dev->create_binding_table(signature);
        defer(table->release());
        table->bind_UAV_texture(0, 2, 0, this->normal_rt, rd::Image_Subresource::top_level(),
                                rd::Format::NATIVE);
        table->bind_UAV_texture(0, 3, 0, this->depth_rt, rd::Image_Subresource::top_level(),
                                rd::Format::NATIVE);
        table->bind_UAV_texture(0, 4, 0, cur_grid, rd::Image_Subresource::top_level(),
                                rd::Format::NATIVE);
        ctx->bind_table(table);
        struct PushConstants {
          u32 clear_counter_grid;
        } pc;
        pc.clear_counter_grid = 0;
        table->push_constants(&pc, 0, sizeof(pc));
        ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
        pc.clear_counter_grid = 1;
        table->push_constants(&pc, 0, sizeof(pc));
        ctx->dispatch((COUNTER_GRID_RESOLUTION + 15) / 16, (COUNTER_GRID_RESOLUTION + 15) / 16, 1);
      }
      dev->end_compute_pass(ctx);
    }
    Resource_ID feedback_buffer = [&] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
      buf_info.size        = 1 << 20;
      return dev->create_buffer(buf_info);
    }();
    dev->release_resource(feedback_buffer);
    rd::ICtx *ctx = dev->start_compute_pass();
    timestamps.begin_range(ctx);
    {
      TracyVulkIINamedZone(ctx, "G.I. Prepare Compute GBuffer Pass");

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      table->bind_UAV_texture(0, 4, 0, prev_grid, rd::Image_Subresource::top_level(),
                              rd::Format::NATIVE);
      table->bind_UAV_buffer(0, 6, indirect_args_buffer, 0, 0);
      ctx->buffer_barrier(indirect_args_buffer, rd::Buffer_Access::UAV);
      ctx->bind_compute(prepare_indirect_pso);
      ctx->bind_table(table);
      struct PushConstants {
        u32 min_triangle_size;
        u32 max_triangle_size;
        u32 min_resolution;
        u32 max_resolution;
      } pc;
      pc.min_triangle_size = rctx.config->get_u32("GI.min_triangle_size", 4, 1, 64);
      pc.max_triangle_size = rctx.config->get_u32("GI.max_triangle_size", 32, 1, 64);
      pc.min_resolution    = rctx.config->get_u32("GI.min_resolution", 1, 1, 128);
      pc.max_resolution    = rctx.config->get_u32("GI.max_resolution", 16, 1, 128);
      table->push_constants(&pc, 0, sizeof(pc));
      ctx->dispatch((COUNTER_GRID_RESOLUTION + 16 - 1) / 16,
                    (COUNTER_GRID_RESOLUTION + 16 - 1) / 16, 1);
      ctx->buffer_barrier(indirect_args_buffer, rd::Buffer_Access::INDIRECT_ARGS);
    }
    {
      TracyVulkIINamedZone(ctx, "G.I. Compute GBuffer Pass");

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      table->bind_cbuffer(1, 0, cbuffer, 0, 0);
      table->bind_texture(0, 0, 0, pos_tex, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
      table->bind_texture(0, 1, 0, normal_tex, rd::Image_Subresource::top_level(),
                          rd::Format::NATIVE);

      table->bind_UAV_texture(0, 2, 0, this->normal_rt, rd::Image_Subresource::top_level(),
                              rd::Format::NATIVE);
      table->bind_UAV_texture(0, 3, 0, this->depth_rt, rd::Image_Subresource::top_level(),
                              rd::Format::NATIVE);
      table->bind_UAV_texture(0, 4, 0, cur_grid, rd::Image_Subresource::top_level(),
                              rd::Format::NATIVE);
      table->bind_UAV_buffer(0, 6, indirect_args_buffer, 0, 0);
      table->bind_UAV_texture(0, 7, 0, prev_grid, rd::Image_Subresource::top_level(),
                              rd::Format::NATIVE);

      table->bind_sampler(0, 5, sampler_state);
      ctx->image_barrier(pos_tex, rd::Image_Access::SAMPLED);
      ctx->image_barrier(normal_tex, rd::Image_Access::SAMPLED);
      ctx->bind_compute(prepare_indirect_pso);
      ctx->bind_table(table);
      GI_PushConstants pc{};
      if (rctx.config->get_bool("G.I.color_triangles")) {
        pc.flags |= GI_RASTERIZATION_FLAG_PIXEL_COLOR_TRIANGLES;
      }

      ctx->bind_compute(pso);
      yto(COUNTER_GRID_RESOLUTION) {
        xto(COUNTER_GRID_RESOLUTION) {
          // pc.cell_x = rctx.config->get_u32("G.I.grid_size");
          pc.cell_x = x;
          pc.cell_y = y;
          pc.model  = float4x4(1.0f);
          table->push_constants(&pc, 0, sizeof(pc));
          ctx->dispatch_indirect(indirect_args_buffer, sizeof(rd::Dispatch_Indirect_Args) *
                                                           (x + y * COUNTER_GRID_RESOLUTION));
        }
      }
    }
    timestamps.end_range(ctx);
    Resource_ID e = dev->end_compute_pass(ctx);
    timestamps.commit(e);
  }
  GBuffer get_gbuffer() {
    GBuffer out{};
    out.normal = normal_rt;
    out.depth  = depth_rt;
    return out;
  }
  void release(rd::IDevice *factory) {
    timestamps.release(factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
};

class ComputeGBufferPass {
  public:
  static constexpr char const *NAME = "Compute GBuffer Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(clear_pso);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(normal_rt);                                                                             \
  RESOURCE(depth_rt);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  u32 width  = 0;
  u32 height = 0;

  public:
#include "examples/shaders/declarations.hlsl"

  TimeStamp_Pool timestamps = {};

  void init(RenderingContext rctx) {
    TMP_STORAGE_SCOPE;
    auto dev = rctx.factory;
    timestamps.init(dev);

    signature = [=] {
      rd::Binding_Table_Create_Info table_info{};
      {
        rd::Binding_Space_Create_Info set_info{};
        set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
        set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
        set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
        set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
        table_info.spaces.push(set_info);
      }
      {
        rd::Binding_Space_Create_Info set_info{};
        set_info.bindings.push({rd::Binding_t::UNIFORM_BUFFER, 1});
        table_info.spaces.push(set_info);
      }
      {
        rd::Binding_Space_Create_Info set_info{};
        set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
        table_info.spaces.push(set_info);
      }
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();

    Resource_ID cs{};
    pso = dev->create_compute_pso(
        signature,
        cs = dev->create_shader(rd::Stage_t::COMPUTE,
                                read_file_tmp_stref("examples/shaders/compute_rasterization.hlsl"),
                                NULL, 0));
    dev->release_resource(cs);

    cs        = {};
    clear_pso = dev->create_compute_pso(signature,
                                        cs = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(

[[vk::binding(2, 0)]] RWTexture2D<float4> normal_target   : register(u2, space0);
[[vk::binding(3, 0)]] RWTexture2D<uint>  depth_target    : register(u3, space0);

[numthreads(16, 16, 1)]
  void main(uint3 tid : SV_DispatchThreadID)
{
  uint width, height;
  normal_target.GetDimensions(width, height);
  if (tid.x >= width || tid.y >= height)
    return;
  normal_target[tid.xy] = float4_splat(0.0f);
  depth_target[tid.xy]  = 0;
}
)"),
                                                                NULL, 0));
    dev->release_resource(cs);
  }
  void update_frame_buffer(RenderingContext rctx) {
    auto dev = rctx.factory;
    if (normal_rt.is_valid()) dev->release_resource(normal_rt);
    if (depth_rt.is_valid()) dev->release_resource(depth_rt);

    normal_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
    depth_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::R32_UINT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
  }
  void render(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.update(dev);
    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    float4x4       viewproj = rctx.gizmo_layer->get_camera().viewproj();
    FrameConstants fc{};
    fc.viewproj         = viewproj;
    Resource_ID cbuffer = [&] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
      buf_info.size        = sizeof(fc);
      return dev->create_buffer(buf_info);
    }();
    dev->release_resource(cbuffer);
    {
      memcpy(dev->map_buffer(cbuffer), &fc, sizeof(fc));
      dev->unmap_buffer(cbuffer);
    }
    PushConstants pc{};
    if (rctx.config->get_bool("RASTERIZATION_FLAG_CULL_PIXELS")) {
      pc.flags |= RASTERIZATION_FLAG_CULL_PIXELS;
    }
    {
      rd::ICtx *ctx = dev->start_compute_pass();
      {
        TracyVulkIINamedZone(ctx, "Compute Clear GBuffer Pass");
        ctx->bind_compute(clear_pso);
        rd::IBinding_Table *table = dev->create_binding_table(signature);
        defer(table->release());
        table->bind_UAV_texture(0, 2, 0, this->normal_rt, rd::Image_Subresource::top_level(),
                                rd::Format::NATIVE);
        table->bind_UAV_texture(0, 3, 0, this->depth_rt, rd::Image_Subresource::top_level(),
                                rd::Format::NATIVE);
        ctx->bind_table(table);
        ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
      }
      dev->end_compute_pass(ctx);
    }
    Resource_ID feedback_buffer = [&] {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
      buf_info.size        = 1 << 20;
      return dev->create_buffer(buf_info);
    }();
    dev->release_resource(feedback_buffer);
    rd::ICtx *ctx = dev->start_compute_pass();
    {
      TracyVulkIINamedZone(ctx, "Compute GBuffer Pass");
      timestamps.begin_range(ctx);
      ctx->bind_compute(pso);
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
          float4x4             world = mn->get_transform();
          ito(gs->getNumSurfaces()) {
            GfxSurface *        s     = gs->getSurface(i);
            rd::IBinding_Table *table = dev->create_binding_table(signature);
            defer(table->release());
            table->bind_cbuffer(1, 0, cbuffer, 0, 0);
            table->bind_UAV_buffer(2, 0, feedback_buffer, 0, 0);
            table->bind_UAV_buffer(0, 0, s->buffer, 0, 0);
            table->bind_UAV_buffer(0, 1, s->buffer, s->index_offset, 0);
            table->bind_UAV_texture(0, 2, 0, this->normal_rt, rd::Image_Subresource::top_level(),
                                    rd::Format::NATIVE);
            table->bind_UAV_texture(0, 3, 0, this->depth_rt, rd::Image_Subresource::top_level(),
                                    rd::Format::NATIVE);
            pc.normal_offset   = s->get_attribute_offset(rd::Attriute_t::NORMAL);
            pc.normal_stride   = s->get_attribute_stride(rd::Attriute_t::NORMAL);
            pc.position_offset = s->get_attribute_offset(rd::Attriute_t::POSITION);
            pc.position_stride = s->get_attribute_stride(rd::Attriute_t::POSITION);
            pc.first_vertex    = 0;
            pc.index_count     = s->total_indices;
            pc.index_offset    = 0;
            pc.index_stride    = s->get_bytes_per_index();
            pc.model           = mn->get_transform();
            table->push_constants(&pc, 0, sizeof(pc));
            ctx->bind_table(table);
            ctx->dispatch(
                (s->total_indices + RASTERIZATION_GROUP_SIZE - 1) / RASTERIZATION_GROUP_SIZE, 1, 1);
          }
        }
      });
      timestamps.end_range(ctx);
    }

    Resource_ID e = dev->end_compute_pass(ctx);
    timestamps.commit(e);
  }
  GBuffer get_gbuffer() {
    GBuffer out{};
    out.normal = normal_rt;
    out.depth  = depth_rt;
    return out;
  }
  void release(rd::IDevice *factory) {
    timestamps.release(factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
};

class GBufferPass {
  public:
  static constexpr char const *NAME = "GBuffer Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(pass);                                                                                  \
  RESOURCE(frame_buffer);                                                                          \
  RESOURCE(normal_rt);                                                                             \
  RESOURCE(depth_rt);                                                                              \
  RESOURCE(gbuffer_vs);                                                                            \
  RESOURCE(gbuffer_ps);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  u32 width  = 0;
  u32 height = 0;
  // BufferThing bthing{};

  rd::Render_Pass_Create_Info info{};
  rd::Graphics_Pipeline_State gfx_state{};

  public:
  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  void init(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.init(dev);
    // bthing.init(dev);
    gbuffer_vs = dev->create_shader(rd::Stage_t::VERTEX, stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
  float4x4 world_transform;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
};

struct VSInput {
  [[vk::location(0)]] float3 pos     : POSITION;
  [[vk::location(1)]] float3 normal  : NORMAL;
  [[vk::location(4)]] float2 uv      : TEXCOORD0;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.normal = mul(pc.world_transform, float4(input.normal.xyz, 0.0f)).xyz;
  output.uv     = input.uv;
  output.pos    = mul(pc.viewproj, mul(pc.world_transform, float4(input.pos, 1.0f)));
  return output;
}
)"),
                                    NULL, 0);
    gbuffer_ps = dev->create_shader(rd::Stage_t::PIXEL, stref_s(R"(
struct PSInput {
  [[vk::location(0)]] float4 pos     : SV_POSITION;
  [[vk::location(1)]] float3 normal  : TEXCOORD0;
  [[vk::location(2)]] float2 uv      : TEXCOORD1;
};

float4 main(in PSInput input) : SV_TARGET0 {
  return float4(input.normal.xyz, 1.0f);
}
)"),
                                    NULL, 0);
    signature  = [=] {
      rd::Binding_Space_Create_Info set_info{};
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    pass = [=] {
      rd::Render_Pass_Create_Info info{};
      rd::RT_Ref                  rt0{};
      rt0.format            = rd::Format::RGBA32_FLOAT;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);

      info.depth_target.enabled           = true;
      info.depth_target.clear_depth.clear = true;
      info.depth_target.format            = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_render_pass(info);
    }();

    pso = [=] {
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state{};
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CCW;
      rs_state.cull_mode    = rd::Cull_Mode::BACK;
      gfx_state.RS_set_state(rs_state);
      ds_state.cmp_op             = rd::Cmp::GE;
      ds_state.enable_depth_test  = true;
      ds_state.enable_depth_write = true;
      gfx_state.DS_set_state(ds_state);
      rd::Blend_State bs{};
      bs.enabled = false;
      bs.color_write_mask =
          (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
          (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
      gfx_state.OM_set_blend_state(0, bs);
      gfx_state.VS_set_shader(gbuffer_vs);
      gfx_state.PS_set_shader(gbuffer_ps);
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 1;
        info.format   = rd::Format::RGB32_FLOAT;
        info.location = 1;
        info.offset   = 0;
        info.type     = rd::Attriute_t::NORMAL;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 2;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 2;
        info.offset   = 0;
        info.type     = rd::Attriute_t::TEXCOORD0;
        gfx_state.IA_set_attribute(info);
      }
      gfx_state.IA_set_vertex_binding(0, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(1, 12, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(2, 8, rd::Input_Rate::VERTEX);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();
  }
  void update_frame_buffer(RenderingContext rctx) {
    auto dev = rctx.factory;
    if (frame_buffer.is_valid()) dev->release_resource(frame_buffer);
    if (normal_rt.is_valid()) dev->release_resource(normal_rt);
    if (depth_rt.is_valid()) dev->release_resource(depth_rt);

    normal_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
    depth_rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format = rd::Format::D32_OR_R32_FLOAT;
      rt0_info.width  = width;
      rt0_info.height = height;
      rt0_info.depth  = 1;
      rt0_info.layers = 1;
      rt0_info.levels = 1;
      rt0_info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_DT | (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      return dev->create_image(rt0_info);
    }();
    frame_buffer = [=] {
      rd::Frame_Buffer_Create_Info info{};
      rd::RT_View                  rt0{};
      rt0.image  = normal_rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);

      info.depth_target.enabled = true;
      info.depth_target.image   = depth_rt;
      info.depth_target.format  = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_frame_buffer(pass, info);
    }();
  }
  void render(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.update(dev);
    // float4x4 bvh_visualizer_offset = glm::translate(float4x4(1.0f), float3(-10.0f, 0.0f,
    // 0.0f));
    // bthing.test_buffers(dev);
    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    struct PushConstants {
      float4x4 viewproj;
      float4x4 world_transform;
    } pc;

    float4x4 viewproj = rctx.gizmo_layer->get_camera().viewproj();

    rd::ICtx *ctx = dev->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "GBuffer Pass");
      timestamps.begin_range(ctx);
      ctx->start_render_pass();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      pc.viewproj = viewproj;

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      table->push_constants(&viewproj, 0, sizeof(float4x4));
      ctx->bind_table(table);
      ctx->bind_graphics_pso(pso);

      if (rctx.config->get_bool("ras.render_meshlets")) {
        rctx.scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            if (auto *sc = mn->getComponent<GfxMeshletSufraceComponent>()) {
              ito(mn->getNumSurfaces()) {
                GfxMeshletSurface *gfx_meshlets = sc->get_meshlets(i);
                gfx_meshlets->iterate([](Meshlet const &meshlet) {

                });
              }
            }
          }
        });
      }
      if (rctx.config->get_bool("ras.render_geometry")) {
        rctx.scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            GfxSufraceComponent *gs    = mn->getComponent<GfxSufraceComponent>();
            float4x4             world = mn->get_transform();
            pc.world_transform         = world;
            table->push_constants(&pc, 0, sizeof(pc));
            ito(gs->getNumSurfaces()) {
              GfxSurface *s = gs->getSurface(i);
              s->draw(ctx, gfx_state);
            }
          }
        });
      }
      ctx->end_render_pass();
      timestamps.end_range(ctx);
    }

    Resource_ID e = dev->end_render_pass(ctx);
    timestamps.commit(e);
  }
  GBuffer get_gbuffer() {
    GBuffer out{};
    out.normal = normal_rt;
    out.depth  = depth_rt;
    return out;
  }
  void release(rd::IDevice *factory) {
    timestamps.release(factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
};

class GizmoPass {
  public:
  static constexpr char const *NAME = "Gizmo Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(pso);                                                                                   \
  RESOURCE(pass);                                                                                  \
  RESOURCE(frame_buffer);                                                                          \
  RESOURCE(rt);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE
  void release(rd::IDevice *factory) {
    timestamps.release(factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST

  u32         width  = 0;
  u32         height = 0;
  Resource_ID last_depth_rt{};

  public:
  TimeStamp_Pool timestamps = {};
  struct PushConstants {
    float4x4 viewproj;
    float4x4 world_transform;
  };
  void init(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.init(dev);
    pass = [=] {
      rd::Render_Pass_Create_Info info{};
      rd::RT_Ref                  rt0{};
      rt0.format            = rd::Format::RGBA32_FLOAT;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);

      info.depth_target.enabled           = true;
      info.depth_target.clear_depth.clear = false;
      info.depth_target.format            = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_render_pass(info);
    }();
  }
  void update_frame_buffer(RenderingContext rctx, Resource_ID depth_rt) {
    auto dev = rctx.factory;
    if (frame_buffer.is_valid()) dev->release_resource(frame_buffer);
    if (rt.is_valid()) dev->release_resource(rt);

    rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |      //
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();

    frame_buffer = [=] {
      rd::Frame_Buffer_Create_Info info{};
      rd::RT_View                  rt0{};
      rt0.image  = rt;
      rt0.format = rd::Format::RGBA32_FLOAT;
      info.rts.push(rt0);

      info.depth_target.enabled = true;
      info.depth_target.image   = depth_rt;
      info.depth_target.format  = rd::Format::D32_OR_R32_FLOAT;
      return dev->create_frame_buffer(pass, info);
    }();
  }
  void render(RenderingContext rctx, Resource_ID depth_rt) {
    auto dev = rctx.factory;
    timestamps.update(dev);
    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height || last_depth_rt.data != depth_rt.data) {
      this->width   = width;
      this->height  = height;
      last_depth_rt = depth_rt;
      update_frame_buffer(rctx, depth_rt);
    }
    if (rctx.config->get_bool("gizmo.enable")) {
      auto g_camera = rctx.gizmo_layer->get_camera();
      {
        float dx = 1.0e-1f * g_camera.distance;
        rctx.gizmo_layer->draw_sphere(g_camera.look_at, dx * 0.04f, float3{1.0f, 1.0f, 1.0f});
        rctx.gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{dx, 0.0f, 0.0f},
                                        dx * 0.04f, float3{1.0f, 0.0f, 0.0f});
        rctx.gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, dx, 0.0f},
                                        dx * 0.04f, float3{0.0f, 1.0f, 0.0f});
        rctx.gizmo_layer->draw_cylinder(g_camera.look_at, g_camera.look_at + float3{0.0f, 0.0f, dx},
                                        dx * 0.04f, float3{0.0f, 0.0f, 1.0f});
      }

      if (rctx.config->get_bool("gizmo.render_bounds")) {
        rctx.scene->traverse([&](Node *node) {
          AABB     aabb = node->getAABB();
          float4x4 t(1.0f);
          rctx.gizmo_layer->render_linebox(transform(t, aabb.min), transform(t, aabb.max),
                                           float3(1.0f, 0.0f, 0.0f));
        });
      }
      if (rctx.config->get_bool("gizmo.render_meshlets")) {
        ///////////////
        // [WARNING] //
        ///////////////
        // Edge reference count
        static thread_local u8 ref_cnt[1 << 20]{};
        ///////////////
        rctx.scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
            if (auto *sc = mn->getComponent<MeshletSufraceComponent>()) {
              ito(mn->getNumSurfaces()) {
                Raw_Meshlets_Opaque *meshlets = sc->get_meshlets(i);
                jto(meshlets->meshlets.size) {
                  Meshlet &m = meshlets->meshlets[j];
                  ASSERT_DEBUG(sizeof(ref_cnt) > sizeof(u8) * m.index_count * m.index_count);
                  memset(ref_cnt, 0, sizeof(u8) * m.index_count * m.index_count);
                  kto(m.index_count / 3) {
                    u32 i0 = meshlets->index_data[m.index_offset + k * 3 + 0];
                    u32 i1 = meshlets->index_data[m.index_offset + k * 3 + 1];
                    u32 i2 = meshlets->index_data[m.index_offset + k * 3 + 2];
                    ref_cnt[i0 * m.index_count + i1]++;
                    ref_cnt[i1 * m.index_count + i0]++;

                    ref_cnt[i1 * m.index_count + i2]++;
                    ref_cnt[i2 * m.index_count + i1]++;

                    ref_cnt[i2 * m.index_count + i0]++;
                    ref_cnt[i0 * m.index_count + i2]++;
                  }
                  kto(m.index_count / 3) {
                    u32    i0    = meshlets->index_data[m.index_offset + k * 3 + 0];
                    u32    i1    = meshlets->index_data[m.index_offset + k * 3 + 1];
                    u32    i2    = meshlets->index_data[m.index_offset + k * 3 + 2];
                    i32    refs0 = ref_cnt[i0 * m.index_count + i1];
                    i32    refs1 = ref_cnt[i1 * m.index_count + i2];
                    i32    refs2 = ref_cnt[i2 * m.index_count + i0];
                    float3 p0    = meshlets->fetch_position(m.vertex_offset + i0);
                    float3 p1    = meshlets->fetch_position(m.vertex_offset + i1);
                    float3 p2    = meshlets->fetch_position(m.vertex_offset + i2);
                    if (refs0 == 1) rctx.gizmo_layer->draw_line(p0, p1, float3(1.0f, 1.0f, 0.0f));
                    if (refs1 == 1) rctx.gizmo_layer->draw_line(p1, p2, float3(1.0f, 1.0f, 0.0f));
                    if (refs2 == 1) rctx.gizmo_layer->draw_line(p2, p0, float3(1.0f, 1.0f, 0.0f));
                  }
                }
              }
            }
          }
        });
      }
      if (rctx.config->get_bool("gizmo.render_bvh")) {
        rctx.scene->traverse([&](Node *node) {
          if (MeshNode *mn = node->dyn_cast<MeshNode>()) {

            if (auto *sc = mn->getComponent<BVHSufraceComponent>()) {
              if (sc->getBVH()) {
                render_bvh(float4x4(1.0f), sc->getBVH(), rctx.gizmo_layer);
              }
            }
          }
        });
      }
    }

    struct PushConstants {
      float4x4 viewproj;
      float4x4 world_transform;
    } pc;

    float4x4 viewproj = rctx.gizmo_layer->get_camera().viewproj();

    rd::ICtx *ctx = dev->start_render_pass(pass, frame_buffer);
    {
      TracyVulkIINamedZone(ctx, "Gizmo Pass");
      timestamps.begin_range(ctx);
      ctx->start_render_pass();

      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, width, height);
      rctx.gizmo_layer->render(ctx, width, height);
      ctx->end_render_pass();
      timestamps.end_range(ctx);
    }
    rctx.gizmo_layer->reset();
    Resource_ID e = dev->end_render_pass(ctx);
    timestamps.commit(e);
  }
};

class ComposePass {
  public:
  static constexpr char const *NAME = "Compose Pass";
  Pair<double, char const *>   get_duration() { return {timestamps.duration, NAME}; }

#define RESOURCE_LIST                                                                              \
  RESOURCE(signature);                                                                             \
  RESOURCE(sampler_state);                                                                         \
  RESOURCE(rt);                                                                                    \
  RESOURCE(pso);

#define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#undef RESOURCE

  u32 width  = 0;
  u32 height = 0;

  public:
  TimeStamp_Pool timestamps = {};
  struct PushConstants {};
  void init(RenderingContext rctx) {
    auto dev = rctx.factory;
    timestamps.init(dev);

    signature = [=] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
      set_info.bindings.push({rd::Binding_t::TEXTURE, 16});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = 0; // sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    sampler_state = [&] {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::LINEAR;
      info.min_filter     = rd::Filter::LINEAR;
      info.mip_mode       = rd::Filter::LINEAR;
      info.max_lod        = 1000.0f;
      info.anisotropy     = true;
      info.max_anisotropy = 16.0f;
      return dev->create_sampler(info);
    }();
    pso = [&] {
      Resource_ID cs{};
      defer(dev->release_resource(cs));
      return dev->create_compute_pso(signature,
                                     cs = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWTexture2D<float4> compose        : register(u0, space0);
[[vk::binding(1, 0)]] SamplerState        ss             : register(s1, space0);

[[vk::binding(2, 0)]] Texture2D<float4>   inputs[16] : register(t2, space0);

#define GBUFFER_NORMAL 0
#define GBUFFER_DEPTH 1
#define GIZMO_LAYER 2

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
  uint width, height;
  compose.GetDimensions(width, height);
  if (tid.x >= width || tid.y >= height)
    return;
  float2 uv = (float2(tid.xy) + float2(0.5f, 0.5f)) / float2(width, height);
  float3 normal = inputs[GBUFFER_NORMAL].Load(int3(tid.xy, 0)).xyz;
  float4 gizmo  = inputs[GIZMO_LAYER].Load(int3(tid.xy, 0)).xyzw;
  float3 color = float3_splat(max(0.0f, dot(normal, normalize(float3(1.0, 1.0, 1.0)))));
  compose[tid.xy] = float4(lerp(pow(color, 1.0f), gizmo.xyz, gizmo.w), 1.0f);
}
)"),
                                                             NULL, 0));
    }();
  }
  void update_frame_buffer(RenderingContext rctx) {
    auto dev = rctx.factory;
    if (rt.is_valid()) dev->release_resource(rt);
    rt = [=] {
      rd::Image_Create_Info rt0_info{};
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = width;
      rt0_info.height     = height;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.usage_bits =                          //
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | //
          (u32)rd::Image_Usage_Bits::USAGE_UAV;
      return dev->create_image(rt0_info);
    }();
  }
  void render(RenderingContext rctx, GBuffer gbuffer, Resource_ID gizmo_layer) {
    auto dev = rctx.factory;
    timestamps.update(dev);
    u32 width  = rctx.config->get_u32("g_buffer_width");
    u32 height = rctx.config->get_u32("g_buffer_height");
    if (this->width != width || this->height != height) {
      this->width  = width;
      this->height = height;
      update_frame_buffer(rctx);
    }

    rd::IBinding_Table *table = dev->create_binding_table(signature);
    defer(table->release());
    table->bind_texture(0, 2, 0, gbuffer.normal, rd::Image_Subresource::top_level(),
                        rd::Format::NATIVE);
    table->bind_texture(0, 2, 1, gbuffer.depth, rd::Image_Subresource::top_level(),
                        rd::Format::NATIVE);
    table->bind_texture(0, 2, 2, gizmo_layer, rd::Image_Subresource::top_level(),
                        rd::Format::NATIVE);
    table->bind_UAV_texture(0, 0, 0, rt, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    table->bind_sampler(0, 1, sampler_state);
    rd::ICtx *ctx = dev->start_compute_pass();
    ctx->image_barrier(gbuffer.normal, rd::Image_Access::SAMPLED);
    ctx->image_barrier(gbuffer.depth, rd::Image_Access::SAMPLED);
    ctx->image_barrier(gizmo_layer, rd::Image_Access::SAMPLED);
    {
      TracyVulkIINamedZone(ctx, NAME);
      timestamps.begin_range(ctx);
      ctx->bind_compute(pso);
      ctx->bind_table(table);
      ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
      timestamps.end_range(ctx);
    }
    Resource_ID e = dev->end_compute_pass(ctx);
    timestamps.commit(e);
  }
  void release(rd::IDevice *factory) {
    timestamps.release(factory);
#define RESOURCE(name)                                                                             \
  if (name.is_valid()) factory->release_resource(name);
    RESOURCE_LIST
#undef RESOURCE
  }
#undef RESOURCE_LIST
};
#if 1
class Event_Consumer : public IGUIApp {
  public:
  BakerPass            baker_pass;
  GIComputeGBufferPass gi_compute_gbuffer_pass;
  GBufferPass          gbuffer_pass;
  GizmoPass            gizmo_pass;
  ComposePass          compose_pass;
  ComputeGBufferPass   compute_gbuffer_pass;

  RenderingContext rctx{};
  void             init_traverse(List *l) {
    if (l == NULL) return;
    if (l->child) {
      init_traverse(l->child);
      init_traverse(l->next);
    } else {
      if (l->cmp_symbol("camera")) {
        rctx.gizmo_layer->get_camera().traverse(l->next);
      } else if (l->cmp_symbol("config")) {
        rctx.config->traverse(l->next);
      } else if (l->cmp_symbol("scene")) {
        rctx.scene->restore(l);
      }
    }
  }
  void on_gui() override { //
    timer.update();
    ImGui::Begin("Scene");
    {
      String_Builder sb;
      sb.init();
      defer(sb.release());
      rctx.scene->save(sb);
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(sb.get_str(), Tmp_List_Allocator());
      if (cur) {
        int id = 0;
        on_gui_traverse_nodes(cur, id);
        rctx.scene->restore(cur);
      }
    }
    ImGui::End();

    ImGui::Begin("Config");
    if (rctx.config->on_imgui()) rctx.dump();
    ImGui::Text("%s %fms", baker_pass.get_duration().second, baker_pass.get_duration().first);
    ImGui::Text("%s %fms", gi_compute_gbuffer_pass.get_duration().second,
                gi_compute_gbuffer_pass.get_duration().first);
    ImGui::Text("%s %fms", gbuffer_pass.get_duration().second, gbuffer_pass.get_duration().first);
    ImGui::Text("%s %fms", compute_gbuffer_pass.get_duration().second,
                compute_gbuffer_pass.get_duration().first);
    ImGui::Text("%s %fms", gizmo_pass.get_duration().second, gizmo_pass.get_duration().first);
    ImGui::Text("%s %fms", compose_pass.get_duration().second, compose_pass.get_duration().first);
    if (ImGui::Button("Rebuild BVH")) {
      rctx.scene->traverse([&](Node *node) {
        if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
          if (mn->getComponent<BVHSufraceComponent>()) {
            mn->getComponent<BVHSufraceComponent>()->updateBVH();
          }
        }
      });
    }
    ImGui::End();

    ImGui::Begin(gi_compute_gbuffer_pass.NAME);
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(gi_compute_gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
      { Ray ray = rctx.gizmo_layer->getMouseRay(); }
    }
    ImGui::End();
    ImGui::Begin("Sample Counter");
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(gi_compute_gbuffer_pass.prev_grid, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
      { Ray ray = rctx.gizmo_layer->getMouseRay(); }
    }
    ImGui::End();
    ImGui::Begin("Baking Pass");
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(baker_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
      { Ray ray = rctx.gizmo_layer->getMouseRay(); }
    }
    ImGui::End();
    ImGui::Begin("Gbuffer normal");
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
      { Ray ray = rctx.gizmo_layer->getMouseRay(); }
    }
    ImGui::End();
    ImGui::Begin("Compute Gbuffer normal");
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(compute_gbuffer_pass.normal_rt, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
      { Ray ray = rctx.gizmo_layer->getMouseRay(); }
    }
    ImGui::End();
    ImGui::Begin("Gbuffer depth");
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(gbuffer_pass.depth_rt, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
    }
    ImGui::End();
    ImGui::Begin(gizmo_pass.NAME);
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(gizmo_pass.rt, 0, 0, rd::Format::NATIVE), ImVec2(wsize.x, wsize.y));
    }
    ImGui::End();
    ImGui::Begin(compose_pass.NAME);
    {
      rctx.gizmo_layer->per_imgui_window();
      auto wsize = get_window_size();
      ImGui::Image(bind_texture(compose_pass.rt, 0, 0, rd::Format::NATIVE),
                   ImVec2(wsize.x, wsize.y));
    }
    ImGui::End();
  }
  void on_init() override { //

    rctx.factory = this->factory;
    TMP_STORAGE_SCOPE;

    // new XYZDragGizmo(gizmo_layer, &pos);
    rctx.scene  = Scene::create();
    rctx.config = new Config;
    rctx.config->init(stref_s(R"(
 (
  (add u32  g_buffer_width 512 (min 4) (max 2048))
  (add u32  g_buffer_height 512 (min 4) (max 2048))
  (add u32  baking.size 512 (min 4) (max 4096))
  (add bool G.I.color_triangles 0)
 )
 )"));
    rctx.scene->load_mesh(stref_s("mesh"),
                          stref_s("models/castle-ban-the-rhins-of-galloway/scene.gltf"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/norradalur-froyar/scene.gltf"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/cut.gltf"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/human_bust_sculpt/untitled.gltf"));
    // rctx.scene->load_mesh(stref_s("mesh"), stref_s("models/light/scene.gltf"));
    rctx.scene->update();
    rctx.scene->traverse([&](Node *node) {
      if (MeshNode *mn = node->dyn_cast<MeshNode>()) {
        GfxSufraceComponent::create(rctx.factory, mn);
        // MeshletSufraceComponent::create(mn, 255, 256);
        // GfxMeshletSufraceComponent::create(factory, mn);
      }
    });
    baker_pass.init(rctx);
    gi_compute_gbuffer_pass.init(rctx);
    gbuffer_pass.init(rctx);
    compute_gbuffer_pass.init(rctx);
    gizmo_pass.init(rctx);
    compose_pass.init(rctx);
    rctx.gizmo_layer = Gizmo_Layer::create(factory, gizmo_pass.pass);
    char *state      = read_file_tmp("scene_state");

    if (state != NULL) {
      TMP_STORAGE_SCOPE;
      List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
      init_traverse(cur);
    }
  }
  void on_release() override { //
    rctx.dump();
    rctx.gizmo_layer->release();
    rctx.scene->release();
    rctx.config->release();
    baker_pass.release(rctx.factory);
    gi_compute_gbuffer_pass.release(rctx.factory);
    gbuffer_pass.release(rctx.factory);
    compute_gbuffer_pass.release(rctx.factory);
    gizmo_pass.release(rctx.factory);
    compose_pass.release(rctx.factory);
    delete rctx.config;
  }
  void on_frame() override { //
    rctx.scene->get_root()->update();
    baker_pass.render(rctx);
    gi_compute_gbuffer_pass.render(rctx, baker_pass.position_rt, baker_pass.normal_rt);
    gbuffer_pass.render(rctx);
    compute_gbuffer_pass.render(rctx);
    gizmo_pass.render(rctx, gbuffer_pass.get_gbuffer().depth);
    compose_pass.render(rctx, gbuffer_pass.get_gbuffer(), gizmo_pass.rt);
  }
};
#endif
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  auto window_loop = [](rd::Impl_t impl) { IGUIApp::start<Event_Consumer>(impl); };
  // std::thread vulkan_thread = std::thread([window_loop] { window_loop(rd::Impl_t::VULKAN); });
  // std::thread dx12_thread = std::thread([window_loop] { window_loop(rd::Impl_t::DX12); });
  // vulkan_thread.join();
  // dx12_thread.join();

  // window_loop(rd::Impl_t::VULKAN);
  window_loop(rd::Impl_t::DX12);
  return 0;
}