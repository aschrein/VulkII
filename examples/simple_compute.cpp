#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC
#include "rendering.hpp"
//#include "rendering_utils.hpp"
#include "script.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  // rd::IDevice *factory = rd::create_vulkan(NULL);
  RenderDoc_CTX::start();
#if 0


  {
    Image2D *image = load_image(stref_s("images/ECurtis.png"));
    defer(if (image) image->release());
    ASSERT_ALWAYS(image);

    Resource_ID texture =
        Mip_Builder::create_image(factory, image, (u32)rd::Image_Usage_Bits::USAGE_SAMPLED);
    Resource_ID rw_texture{};
    {
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA32_FLOAT;
      info.width      = image->width;
      info.height     = image->height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV;
      rw_texture      = factory->create_image(info);
    }
    defer(factory->release_resource(texture));
    rd::Binding_Table_Create_Info set_info{};
    set_info.bindings.push({0, rd::Binding_t::TEXTURE, 1});
    set_info.bindings.push({1, rd::Binding_t::UAV_TEXTURE, 1});
    set_info.bindings.push({2, rd::Binding_t::SAMPLER, 1});
    rd::IBinding_Table *set0 = factory->create_binding_table(set_info);
    defer(set0->release());
    ctx->bind_compute(
        factory->create_compute_pso(&set0, 1,
                                    factory->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] Texture2D<float4>   rtex          : register(t0, space0);
[[vk::binding(1, 0)]] RWTexture2D<float4> rwtex         : register(u1, space0);
[[vk::binding(2, 0)]] SamplerState        ss            : register(s2, space0);

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
  uint width, height;
  rwtex.GetDimensions(width, height);
  if (tid.x >= width || tid.y >= height)
    return;
  float2 uv = (float2(tid.xy) + float2(0.5f, 0.5f)) / float2(width, height);
  rwtex[tid.xy] = rtex.SampleLevel(ss, uv, 0.0f);
}
)"),
                                                           NULL, 0)));
    ctx->image_barrier(texture, (u32)rd::Access_Bits::SHADER_READ,
                       rd::Image_Layout::READ_ONLY_OPTIMAL);
    ctx->image_barrier(rw_texture, (u32)rd::Access_Bits::SHADER_WRITE,
                       rd::Image_Layout::UAV_OPTIMAL);
    set0->bind_texture(0, 0, texture, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
    set0->bind_UAV_texture(1, 0, rw_texture, rd::Image_Subresource::top_level(),
                           rd::Format::NATIVE);
    Resource_ID sampler_state{};
    {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::NEAREST;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = false;
      info.max_anisotropy = 16.0f;
      sampler_state       = factory->create_sampler(info);
    }
    defer(factory->release_resource(sampler_state));
    set0->bind_sampler(2, sampler_state);
    ctx->bind_table(0, set0);
    ctx->dispatch(image->width / 16 + 1, image->height / 16 + 1, 1);
  }
#endif
  rd::IDevice *factory = rd::create_dx12(NULL);
  factory->start_frame();
  defer({
    factory->end_frame();
    factory->release();
  });
  rd::ICtx *ctx = factory->start_compute_pass();

  Resource_ID signature = [factory] {
    rd::Binding_Space_Create_Info set_info{};
    set_info.bindings.push({0, rd::Binding_t::UAV_BUFFER, 1});
    set_info.bindings.push({1, rd::Binding_t::UAV_BUFFER, 1});
    rd::Binding_Table_Create_Info table_info{};
    table_info.spaces.push(set_info);
    table_info.push_constants_size = 4;
    return factory->create_signature(table_info);
  }();
  defer(factory->release_resource(signature));
  rd::IBinding_Table *table = factory->create_binding_table(signature);
  defer(table->release());
  Resource_ID cs = factory->create_compute_pso(
      signature, factory->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> BufferOut : register(u0, space0);

[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    BufferOut[DTid.x] = DTid.x;
}
)"),
                                        NULL, 0));
  ctx->bind_compute(cs);

  Resource_ID buffer = [factory] {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
    buf_info.usage_bits =
        (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
    buf_info.size = sizeof(u32) * 1024;
    return factory->create_buffer(buf_info);
  }();
  defer(factory->release_resource(buffer));
  table->bind_structured_UAV_buffer(0, 0, buffer, 0, sizeof(u32) * 1024, sizeof(u32));
  ctx->bind_table(table);
  ctx->buffer_barrier(buffer, rd::Buffer_Access::UAV);
  ctx->dispatch(1, 1, 1);

  ctx->buffer_barrier(buffer, rd::Buffer_Access::UAV);

  Resource_ID cs2 = factory->create_compute_pso(
      signature, factory->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> BufferOut: register(u0, space0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> BufferIn : register(u1, space0);

struct CullPushConstants
{
  uint val;
};
[[vk::push_constant]] ConstantBuffer<CullPushConstants> pc : register(b0, space777);

[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    BufferOut[DTid.x] = BufferIn[DTid.x] * pc.val;
}
)"),
                                        NULL, 0));
  table->bind_structured_UAV_buffer(0, 1, buffer, 0, sizeof(u32) * 1024, sizeof(u32));
  ctx->bind_compute(cs2);
  u32 val = 3;
  table->push_constants(&val, 0, 4);
  ctx->dispatch(1, 1, 1);
  /*Resource_ID event_id = factory->create_event();
  ctx->insert_event(event_id);
  Resource_ID readback{};
  {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::CPU_READ_WRITE;
    buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
    buf_info.size        = sizeof(u32) * 1024;
    readback             = factory->create_buffer(buf_info);
  }
  ctx->buffer_barrier(res_id, (u32)rd::Access_Bits::TRANSFER_READ);
  ctx->copy_buffer(res_id, 0, readback, 0, sizeof(u32) * 1024);*/
  factory->end_compute_pass(ctx);
  /* while (!factory->get_event_state(event_id)) fprintf(stdout, "waiting...\n");
   factory->release_resource(event_id);
   u32 *map = (u32 *)factory->map_buffer(readback);
   ito(1024) fprintf(stdout, "%i ", map[i]);
   fflush(stdout);
   factory->unmap_buffer(readback);*/

  // factory->release_resource(res_id);
  RenderDoc_CTX::end();

  return 0;
}
