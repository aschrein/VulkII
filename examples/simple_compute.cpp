#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC
#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "scene.hpp"
#include "script.hpp"
#include "utils.hpp"

void test_buffers(rd::IDevice *factory) {
  rd::ICtx *  ctx       = factory->start_compute_pass();
  Resource_ID signature = [factory] {
    rd::Binding_Space_Create_Info set_info{};
    set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
    set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
    rd::Binding_Table_Create_Info table_info{};
    table_info.spaces.push(set_info);
    table_info.push_constants_size = 4;
    return factory->create_signature(table_info);
  }();
  defer(factory->release_resource(signature));
  Resource_ID cs = factory->create_compute_pso(
      signature, factory->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);

[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    BufferOut.Store<uint>(DTid.x * 4, DTid.x);
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

  rd::IBinding_Table *table = factory->create_binding_table(signature);
  defer(table->release());
  table->bind_UAV_buffer(0, 0, buffer, 0, sizeof(u32) * 1024);
  table->bind_UAV_buffer(0, 1, buffer, 0, sizeof(u32) * 1024);

  ctx->bind_table(table);
  ctx->buffer_barrier(buffer, rd::Buffer_Access::UAV);
  ctx->dispatch(1, 1, 1);

  ctx->buffer_barrier(buffer, rd::Buffer_Access::UAV);

  Resource_ID cs2 = factory->create_compute_pso(
      signature, factory->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut: register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer BufferIn : register(u1, space0);
  
struct CullPushConstants
{
  uint val;
};
[[vk::push_constant]] ConstantBuffer<CullPushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;
  
[numthreads(1024, 1, 1)]
  void main(uint3 DTid : SV_DispatchThreadID)
{
    BufferOut.Store<uint>(DTid.x * 4, BufferIn.Load<uint>(DTid.x * 4) * pc.val);
}
)"),
                                        NULL, 0));

  ctx->bind_compute(cs2);
  u32 val = 3;
  table->push_constants(&val, 0, 4);
  ctx->dispatch(1, 1, 1);
  /* Resource_ID event_id = factory->create_event();
   ctx->insert_event(event_id);*/
  Resource_ID readback = [factory] {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::CPU_READ_WRITE;
    buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
    buf_info.size        = sizeof(u32) * 1024;
    return factory->create_buffer(buf_info);
  }();
  ctx->buffer_barrier(buffer, rd::Buffer_Access::TRANSFER_SRC);
  ctx->copy_buffer(buffer, 0, readback, 0, sizeof(u32) * 1024);
  factory->end_compute_pass(ctx);
  factory->wait_idle();
  // while (!factory->get_event_state(event_id)) fprintf(stdout, "waiting...\n");
  // factory->release_resource(event_id);
  u32 *map = (u32 *)factory->map_buffer(readback);
  ito(1024) {
    // fprintf(stdout, "%i ", map[i]);
    ASSERT_ALWAYS(map[i] == i * 3);
  }
  fflush(stdout);
  factory->unmap_buffer(readback);
}

void test_mipmap_generation(rd::IDevice *factory) {

  Image2D *image = load_image(stref_s("images/ECurtis.png"));
  defer(if (image) image->release());
  ASSERT_ALWAYS(image);
  Resource_ID texture =
      Mip_Builder::create_image(factory, image, (u32)rd::Image_Usage_Bits::USAGE_SAMPLED);

  fprintf(stdout, "MIP levels generated\n");
  Resource_ID sampler_state = [&] {
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
    return factory->create_sampler(info);
  }();
  defer(factory->release_resource(sampler_state));

  Resource_ID rw_texture = [=] {
    rd::Image_Create_Info info;
    MEMZERO(info);
    info.format = rd::Format::RGBA32_FLOAT;
    info.width  = image->width;
    info.height = image->height;
    info.depth  = 1;
    info.layers = 1;
    info.levels = 1;
    info.usage_bits =
        (u32)rd::Image_Usage_Bits::USAGE_UAV | (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC;
    return factory->create_image(info);
  }();
  defer(factory->release_resource(texture));

  Resource_ID signature = [factory] {
    rd::Binding_Space_Create_Info set_info{};
    set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
    set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
    set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
    rd::Binding_Table_Create_Info table_info{};
    table_info.spaces.push(set_info);
    table_info.push_constants_size = 4;
    return factory->create_signature(table_info);
  }();
  defer(factory->release_resource(signature));
  rd::IBinding_Table *table = factory->create_binding_table(signature);
  defer(table->release());
  rd::ICtx *ctx = factory->start_compute_pass();
  ctx->image_barrier(texture, rd::Image_Access::SAMPLED);
  ctx->image_barrier(rw_texture, rd::Image_Access::UAV);

  table->bind_texture(0, 0, 0, texture, rd::Image_Subresource::all_levels(), rd::Format::NATIVE);
  table->bind_UAV_texture(0, 1, 0, rw_texture, rd::Image_Subresource::top_level(),
                          rd::Format::NATIVE);
  table->bind_sampler(0, 2, sampler_state);
  ctx->bind_table(table);

  ctx->bind_compute(factory->create_compute_pso(
      signature, factory->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
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
  rwtex[tid.xy] = pow(rtex.SampleLevel(ss, uv, lerp(0.0f, 10.0f, uv.x)), 0.5f);
}
)"),
                                        NULL, 0)));
  ctx->dispatch(image->width / 16 + 1, image->height / 16 + 1, 1);
  u32         pitch    = rd::IDevice::align_up(sizeof(float) * 4 * image->width,
                                    rd::IDevice::TEXTURE_DATA_PITCH_ALIGNMENT);
  Resource_ID readback = [=] {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::CPU_READ_WRITE;
    buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;

    buf_info.size = pitch * image->height;
    return factory->create_buffer(buf_info);
  }();
  factory->release_resource(readback);
  ctx->image_barrier(rw_texture, rd::Image_Access::TRANSFER_SRC);
  ctx->copy_image_to_buffer(readback, 0, rw_texture, rd::Image_Copy::top_level(pitch));
  factory->end_compute_pass(ctx);
  factory->wait_idle();
  u8 *map = (u8 *)factory->map_buffer(readback);
  {
    Image2D tmp{};
    tmp.data   = map;
    tmp.width  = image->width;
    tmp.height = image->height;
    tmp.format = rd::Format::RGBA32_FLOAT;
    write_image_rgba32_float_pfm("img.pfm", tmp.data, tmp.width, pitch, tmp.height, true);
  }

  factory->unmap_buffer(readback);
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  {
    auto launch_tests = [](rd::IDevice *factory) {
      factory->start_frame();
      RenderDoc_CTX::start();
      defer({
        factory->end_frame();
        factory->release();
      });
      test_mipmap_generation(factory);
      test_buffers(factory);
      RenderDoc_CTX::end();
    };
    // fprintf(stdout, "Testing Vulkan backend\n");
    // launch_tests(rd::create_vulkan(NULL));
    fprintf(stdout, "Testing Dx12 backend\n");
    launch_tests(rd::create_dx12(NULL));
  }

  return 0;
}
