#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC

#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "scene.hpp"
#include "script.hpp"
#include "utils.hpp"

void test_buffers(rd::IDevice *dev) {
  Resource_ID wevent_0{};
  Resource_ID wevent_1{};
  Resource_ID buffer = [dev] {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
    buf_info.usage_bits =
        (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
    buf_info.size = sizeof(u32) * 1024;
    return dev->create_buffer(buf_info);
  }();
  defer(dev->release_resource(buffer));
  // Allocate a buffer.
  Resource_ID buffer1 = [dev] {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
    buf_info.usage_bits =
        (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
    buf_info.size = sizeof(u32) * 1024;
    return dev->create_buffer(buf_info);
  }();
  defer(dev->release_resource(buffer1));
  Resource_ID readback = [dev] {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::CPU_READ_WRITE;
    buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
    buf_info.size        = sizeof(u32) * 1024;
    return dev->create_buffer(buf_info);
  }();
  Resource_ID signature = [dev] {
    rd::Binding_Space_Create_Info set_info{};
    set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
    set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
    rd::Binding_Table_Create_Info table_info{};
    table_info.spaces.push(set_info);
    table_info.push_constants_size = 4;
    return dev->create_signature(table_info);
  }();
  defer(dev->release_resource(signature));

  defer(dev->release_resource(readback));

  Resource_ID cs0 =
      dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);

[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    for (uint i = 0; i < 100000; i++)
      BufferOut.Store<uint>(DTid.x * 4, BufferOut.Load<uint>(DTid.x * 4) + 1);
}
)"),
                                                            NULL, 0));
  dev->release_resource(cs0);
  Resource_ID cs1 =
      dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer BufferIn : register(u1, space0);

[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    for (uint i = 0; i < 100000; i++)
      BufferOut.Store<uint>(DTid.x * 4, BufferIn.Load<uint>(DTid.x * 4) + DTid.x);
}
)"),
                                                            NULL, 0));
  dev->release_resource(cs1);
  Resource_ID cs2 =
      dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
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
    for (uint i = 0; i < 100000; i++)
      BufferOut.Store<uint>(DTid.x * 4, BufferIn.Load<uint>(DTid.x * 4) * pc.val);
}
)"),
                                                            NULL, 0));
  dev->release_resource(cs2);
  {
    // Start renderdoc capture
    // RenderDoc_CTX::start();
    rd::ICtx *ctx = dev->start_compute_pass();
    {
      TracyVulkIINamedZone(ctx, "Async Compute Example 1");
      ctx->bind_compute(cs0);
      // Allocate a binding table.
      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      table->bind_UAV_buffer(/* set/space= */ 0, /* binding= */ 0, buffer, /* offset= */ 0,
                             /* size= */ sizeof(u32) * 1024);
      table->bind_UAV_buffer(/* set/space= */ 0, /* binding= */ 1, buffer, /* offset= */ 0,
                             /* size= */ sizeof(u32) * 1024);
      ctx->bind_table(table);
      ctx->dispatch(1, 1, 1);
    }
    // End the pass and submit the commands to the queue.
    wevent_0 = dev->end_compute_pass(ctx);
    // dev->wait_idle();
    // RenderDoc_CTX::end();
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
      table->bind_UAV_buffer(0, 0, buffer, 0, sizeof(u32) * 1024);
      table->bind_UAV_buffer(0, 1, buffer, 0, sizeof(u32) * 1024);
      ctx->bind_table(table);
      ctx->dispatch(1, 1, 1);
    }
    wevent_1 = dev->end_async_compute_pass(ctx);
  }
  {
    rd::ICtx *ctx = dev->start_async_compute_pass();
    {
      TracyVulkIINamedZone(ctx, "Async Compute Example 3");
      ctx->wait_for_event(wevent_0);
      ctx->wait_for_event(wevent_1);

      ctx->bind_compute(cs2);

      rd::IBinding_Table *table = dev->create_binding_table(signature);
      defer(table->release());
      table->bind_UAV_buffer(0, 0, buffer, 0, sizeof(u32) * 1024);
      table->bind_UAV_buffer(0, 1, buffer, 0, sizeof(u32) * 1024);
      table->push_constants(&val, 0, 4);
      ctx->bind_table(table);
      ctx->dispatch(1, 1, 1);
      ctx->buffer_barrier(buffer, rd::Buffer_Access::TRANSFER_SRC);
      ctx->copy_buffer(buffer, 0, readback, 0, sizeof(u32) * 1024);
    }
    dev->end_async_compute_pass(ctx);
    dev->wait_idle();
    //// while (!dev->get_event_state(event_id)) fprintf(stdout, "waiting...\n");
    //// dev->release_resource(event_id);
    // u32 *map = (u32 *)dev->map_buffer(readback);
    // ito(1024) {
    //  // fprintf(stdout, "%i ", map[i]);
    //  u32 x       = i;
    //  jto(100000) x = x * val;
    //  ASSERT_ALWAYS(map[i] == x);
    //}
    // fflush(stdout);
    // dev->unmap_buffer(readback);
  }
  fprintf(stdout, "Buffer test finished.\n");
}

void test_mipmap_generation(rd::IDevice *dev) {
  Image2D *image = load_image(stref_s("images/poster-for-movie.png"));
  defer(if (image) image->release());
  ASSERT_ALWAYS(image);
  Mip_Builder *mb = Mip_Builder::create(dev);
  defer(mb->release());
  Resource_ID texture = mb->create_image(dev, image, (u32)rd::Image_Usage_Bits::USAGE_SAMPLED);

  fprintf(stdout, "MIP levels generated\n");
  // return;
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
    return dev->create_sampler(info);
  }();
  defer(dev->release_resource(sampler_state));

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
    return dev->create_image(info);
  }();
  defer(dev->release_resource(texture));

  Resource_ID signature = [dev] {
    rd::Binding_Space_Create_Info set_info{};
    set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
    set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
    set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
    rd::Binding_Table_Create_Info table_info{};
    table_info.spaces.push(set_info);
    table_info.push_constants_size = 4;
    return dev->create_signature(table_info);
  }();
  defer(dev->release_resource(signature));
  rd::IBinding_Table *table = dev->create_binding_table(signature);
  defer(table->release());
  rd::ICtx *  ctx      = dev->start_compute_pass();
  u32         pitch    = rd::IDevice::align_up(sizeof(float) * 4 * image->width,
                                    rd::IDevice::TEXTURE_DATA_PITCH_ALIGNMENT);
  Resource_ID readback = [=] {
    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.memory_type = rd::Memory_Type::CPU_READ_WRITE;
    buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;

    buf_info.size = pitch * image->height;
    return dev->create_buffer(buf_info);
  }();

  dev->release_resource(readback);
  {
    TracyVulkIINamedZone(ctx, "Example Images");
    ctx->image_barrier(texture, rd::Image_Access::SAMPLED);
    ctx->image_barrier(rw_texture, rd::Image_Access::UAV);

    table->bind_texture(0, 0, 0, texture, rd::Image_Subresource::all_levels(), rd::Format::NATIVE);
    table->bind_UAV_texture(0, 1, 0, rw_texture, rd::Image_Subresource::top_level(),
                            rd::Format::NATIVE);
    table->bind_sampler(0, 2, sampler_state);
    ctx->bind_table(table);

    ctx->bind_compute(
        dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
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

    ctx->image_barrier(rw_texture, rd::Image_Access::TRANSFER_SRC);
    ctx->copy_image_to_buffer(readback, 0, rw_texture, rd::Image_Copy::top_level(pitch));
  }
  dev->end_compute_pass(ctx);
  dev->wait_idle();

  u8 *map = (u8 *)dev->map_buffer(readback);
  {
    Image2D tmp{};
    tmp.data   = map;
    tmp.width  = image->width;
    tmp.height = image->height;
    tmp.format = rd::Format::RGBA32_FLOAT;
    write_image_rgba32_float_pfm("img.pfm", tmp.data, tmp.width, pitch, tmp.height, true);
  }

  dev->unmap_buffer(readback);
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  {
    auto launch_tests = [](rd::IDevice *dev) {
      defer({ dev->release(); });

      //do {
        RenderDoc_CTX::start();
        dev->start_frame();
        // test_buffers(dev);
        test_mipmap_generation(dev);
        dev->end_frame();
        RenderDoc_CTX::end();
      //} while (true);
    };
    // fprintf(stdout, "Testing Vulkan backend\n");
    launch_tests(rd::create_dx12(NULL));
    // launch_tests(rd::create_vulkan(NULL));
    // fprintf(stdout, "Testing Dx12 backend\n");
    // launch_tests(rd::create_dx12(NULL));
  }
  return 0;
}
