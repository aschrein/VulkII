#define UTILS_TL_IMPL
#define SCRIPT_IMPL
#define UTILS_RENDERDOC

#include "rendering.hpp"
#include "rendering_utils.hpp"
#include "scene.hpp"
#include "script.hpp"
#include "utils.hpp"

#define RETERR(msg, ...)                                                                           \
  do {                                                                                             \
    fprintf(stderr, "[dispatch_kernel.exe]: " msg, __VA_ARGS__);                                   \
    exit(1);                                                                                       \
  } while (0)

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  string_ref kernel_path{};
  u32        PARAM_OFFSET = 2;
#define I32_ARG std::atoi(argv[PARAM_OFFSET++])
#define STREF_ARG stref_s(argv[PARAM_OFFSET++])

  kernel_path = STREF_ARG;
  // if (kernel_path.len == 0) RETERR("Cannot find the kernel %s", argv[1]);
  i32 num_groups_x     = I32_ARG;
  i32 num_groups_y     = I32_ARG;
  i32 num_groups_z     = I32_ARG;
  i32 num_launches     = I32_ARG;
  i32 num_items        = I32_ARG;
  i32 uav_texture_size = I32_ARG;
  i32 print_result     = I32_ARG;
  if (uav_texture_size < 0 || uav_texture_size > 1 << 12)
    RETERR("uav_texture_size must be > 0 and <= 1 << 12");
  if (num_items < 1 || num_items > 1 << 20 || (num_items % 64 != 0))
    RETERR("num_items must be > 1 and <= 1 << 20 and %64 == 0");
  if (num_launches < 1 || num_launches > 1 << 12) RETERR("num_launches must be > 1 and <= 1 << 12");
  if (num_groups_x < 1 || num_groups_x > 1 << 12) RETERR("num_groups must be > 1 and <= 1 << 12");
  if (num_groups_y < 1 || num_groups_y > 1 << 12) RETERR("num_groups must be > 1 and <= 1 << 12");
  if (num_groups_z < 1 || num_groups_z > 1 << 12) RETERR("num_groups must be > 1 and <= 1 << 12");
  // if (num_groups_z * num_groups_x * num_groups_y > 1024)
  //  RETERR("num_groups must be > 1 and <= 1024");
  struct PushConstants {
    u32 params[16];
  } pc{};

  for (int i = PARAM_OFFSET; i < argc; i++) {
    pc.params[i - PARAM_OFFSET] = std::atoi(argv[i]);
  }

  {

    auto launch_tests = [&](rd::IDevice *dev) {
      RenderDoc_CTX::start();
      Image2D *   image = NULL;
      Resource_ID texture{};
      Resource_ID sampler_state{};
      Resource_ID rw_texture0{};
      Resource_ID rw_texture1{};
      {
        TMP_STORAGE_SCOPE;
        dev->start_frame();
        TimeStamp_Pool timestamps = {};
        timestamps.init(dev);
        if (uav_texture_size) {

          char ownPth[MAX_PATH];

          // When NULL is passed to GetModuleHandle, the handle of the exe itself is returned
          HMODULE hModule = GetModuleHandle(NULL);
          if (hModule != NULL) {
            // Use GetModuleFileName() with module handle to get the path
            GetModuleFileName(hModule, ownPth, (sizeof(ownPth)));
            char *last_delim = NULL;
            char *c          = ownPth;
            while (*c != '\0') {
              if (*c == '/' || *c == '\\') last_delim = c;
              c++;
            }
            if (last_delim) *last_delim = '\0';
          }

          image = load_image(stref_concat_tmp(stref_concat_tmp(stref_s(ownPth), stref_s("\\")),
                                              stref_s("gotg.jpg")));
          ASSERT_ALWAYS(image);
          texture = Mip_Builder::create_image(dev, image, (u32)rd::Image_Usage_Bits::USAGE_SAMPLED,
                                              false);
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

          rw_texture0 = [=] {
            rd::Image_Create_Info info;
            MEMZERO(info);
            info.format     = rd::Format::RGBA32_FLOAT;
            info.width      = uav_texture_size;
            info.height     = uav_texture_size;
            info.depth      = 1;
            info.layers     = 1;
            info.levels     = 1;
            info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                              (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC;
            return dev->create_image(info);
          }();
          rw_texture1 = [=] {
            rd::Image_Create_Info info;
            MEMZERO(info);
            info.format     = rd::Format::RGBA32_FLOAT;
            info.width      = uav_texture_size;
            info.height     = uav_texture_size;
            info.depth      = 1;
            info.layers     = 1;
            info.levels     = 1;
            info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                              (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC;
            return dev->create_image(info);
          }();
        }
        defer(if (image) image->release());

        defer(if (texture.is_valid()) dev->release_resource(texture));
        defer(if (sampler_state.is_valid()) dev->release_resource(sampler_state));
        defer(if (rw_texture0.is_valid()) dev->release_resource(rw_texture0));
        defer(if (rw_texture1.is_valid()) dev->release_resource(rw_texture1));

        u32 pitch = rd::IDevice::align_up(sizeof(float) * 4 * uav_texture_size,
                                          rd::IDevice::TEXTURE_DATA_PITCH_ALIGNMENT);

        Resource_ID buffer0 = [=] {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
          buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                                (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
          buf_info.size = sizeof(u32) * num_items;
          return dev->create_buffer(buf_info);
        }();
        defer(dev->release_resource(buffer0));
        Resource_ID buffer1 = [=] {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
          buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                                (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
          buf_info.size = sizeof(u32) * num_items;
          return dev->create_buffer(buf_info);
        }();
        defer(dev->release_resource(buffer1));
        Resource_ID readback = [=] {
          rd::Buffer_Create_Info buf_info;
          MEMZERO(buf_info);
          buf_info.memory_type = rd::Memory_Type::CPU_READ_WRITE;
          buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
          if (image)
            buf_info.size = MAX(sizeof(u32) * num_items, pitch * image->height);
          else
            buf_info.size = sizeof(u32) * num_items;
          return dev->create_buffer(buf_info);
        }();
        Resource_ID signature = [dev] {
          rd::Binding_Space_Create_Info set_info{};
          set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
          set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
          set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
          set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
          set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
          set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
          rd::Binding_Table_Create_Info table_info{};
          table_info.spaces.push(set_info);
          table_info.push_constants_size = sizeof(PushConstants);
          return dev->create_signature(table_info);
        }();
        defer(dev->release_resource(signature));

        defer(dev->release_resource(readback));
        string_ref text = read_file_tmp_stref(stref_to_tmp_cstr(kernel_path));
        if (text.len == 0) RETERR("Cannot find the kernel %.*s", STRF(kernel_path));
        Resource_ID cs0 = dev->create_compute_pso(
            signature, dev->create_shader(rd::Stage_t::COMPUTE, text, NULL, 0));
        defer(dev->release_resource(cs0));

        Resource_ID cs1 =
            dev->create_compute_pso(signature, dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
[[vk::binding(0, 0)]] RWByteAddressBuffer BufferOut : register(u0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer BufferIn : register(u1, space0);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
  BufferOut.Store<float>(DTid.x * 4, float(DTid.x));
  BufferIn.Store<float>(DTid.x * 4, float(DTid.x));
}
)"),
                                                                  NULL, 0));
        defer(dev->release_resource(cs1));
        // Init
        {
          rd::ICtx *ctx = dev->start_compute_pass();
          TracyVulkIINamedZone(ctx, "Init");
          ctx->bind_compute(cs1);
          rd::IBinding_Table *table = dev->create_binding_table(signature);
          defer(table->release());
          table->bind_UAV_buffer(0, 0, buffer0, 0, sizeof(u32) * num_items);
          table->bind_UAV_buffer(0, 1, buffer1, 0, sizeof(u32) * num_items);
          if (uav_texture_size) {
            table->bind_UAV_texture(0, 2, 0, rw_texture0, rd::Image_Subresource::all_levels(),
                                    rd::Format::NATIVE);
            table->bind_UAV_texture(0, 3, 0, rw_texture1, rd::Image_Subresource::all_levels(),
                                    rd::Format::NATIVE);
            table->bind_texture(0, 4, 0, texture, rd::Image_Subresource::all_levels(),
                                rd::Format::NATIVE);
            table->bind_sampler(0, 5, sampler_state);
          }
          ctx->bind_table(table);
          ctx->dispatch(num_items / 64, 1, 1);
          Resource_ID e = dev->end_compute_pass(ctx);
        }
        dev->end_frame();
        {
          rd::IBinding_Table *table = dev->create_binding_table(signature);
          defer(table->release());
          table->bind_UAV_buffer(0, 0, buffer0, 0, sizeof(u32) * num_items);
          table->bind_UAV_buffer(0, 1, buffer1, 0, sizeof(u32) * num_items);
          if (uav_texture_size) {
            table->bind_UAV_texture(0, 2, 0, rw_texture0, rd::Image_Subresource::all_levels(),
                                    rd::Format::NATIVE);
            table->bind_UAV_texture(0, 3, 0, rw_texture1, rd::Image_Subresource::all_levels(),
                                    rd::Format::NATIVE);
            table->bind_texture(0, 4, 0, texture, rd::Image_Subresource::all_levels(),
                                rd::Format::NATIVE);
            table->bind_sampler(0, 5, sampler_state);
          }
          table->push_constants(&pc, 0, sizeof(pc));

          u32 warmup_launches = 2;
          ito(warmup_launches) {
            dev->start_frame();
            timestamps.update(dev);
            rd::ICtx *ctx = dev->start_compute_pass();
            {
              TracyVulkIINamedZone(ctx, "Warmup");
              if (uav_texture_size) {
                ctx->image_barrier(rw_texture0, rd::Image_Access::UAV);
                ctx->image_barrier(rw_texture1, rd::Image_Access::UAV);
                ctx->image_barrier(texture, rd::Image_Access::SAMPLED);
              }
              ctx->bind_compute(cs0);
              ctx->bind_table(table);
              ctx->dispatch(num_groups_x, num_groups_y, num_groups_z);
            }
            dev->end_compute_pass(ctx);
            dev->end_frame();
          }
          ito(num_launches) {
            dev->start_frame();
            timestamps.update(dev);
            rd::ICtx *ctx = dev->start_compute_pass();
            {
              TracyVulkIINamedZone(ctx, "Compute");
              if (uav_texture_size) {
                ctx->image_barrier(rw_texture0, rd::Image_Access::UAV);
                ctx->image_barrier(rw_texture1, rd::Image_Access::UAV);
                ctx->image_barrier(texture, rd::Image_Access::SAMPLED);
              }
              ctx->bind_compute(cs0);
              ctx->bind_table(table);
              timestamps.begin_range(ctx);
              ctx->dispatch(num_groups_x, num_groups_y, num_groups_z);
              timestamps.end_range(ctx);
            }
            Resource_ID e = dev->end_compute_pass(ctx);
            timestamps.commit(e);
            dev->end_frame();
          }
          dev->wait_idle();
          timestamps.update(dev);

          // fprintf(stdout, "duration: %f us\n", (timestamps.total_duration * 1.0e3) /
          // double(timestamps.total_samples));
          fprintf(stdout, "%s min duration: %f us\n", (dev->getImplType() == rd::Impl_t::VULKAN ? "vk" : "dx"), (timestamps.min_sample * 1.0e3));
          dev->start_frame();
          if (print_result) {
            {
              rd::ICtx *ctx = dev->start_compute_pass();
              {
                ctx->buffer_barrier(buffer1, rd::Buffer_Access::TRANSFER_SRC);
                ctx->copy_buffer(buffer1, 0, readback, 0, sizeof(f32) * num_items);
              }
              dev->end_compute_pass(ctx);
              dev->wait_idle();
              f32 *map = (f32 *)dev->map_buffer(readback);
              ito(num_items) { fprintf(stdout, "%f ", map[i]); }
              fflush(stdout);
              dev->unmap_buffer(readback);
            }
          }
          if (uav_texture_size) {
            rd::ICtx *ctx = dev->start_compute_pass();
            {
              ctx->image_barrier(rw_texture1, rd::Image_Access::TRANSFER_SRC);
              ctx->copy_image_to_buffer(readback, 0, rw_texture1, rd::Image_Copy::top_level(pitch));
            }
            dev->end_compute_pass(ctx);
            dev->wait_idle();

            u8 *map = (u8 *)dev->map_buffer(readback);
            {
              Image2D tmp{};
              tmp.data   = map;
              tmp.width  = uav_texture_size;
              tmp.height = uav_texture_size;
              tmp.format = rd::Format::RGBA32_FLOAT;
              write_image_rgba32_float_pfm(
                  stref_to_tmp_cstr(stref_concat_tmp(kernel_path, stref_s(".pfm"))), tmp.data,
                  tmp.width, pitch, tmp.height, true);
              // write_image_rgba32_float_pfm("tmp.pfm", tmp.data, tmp.width, pitch, tmp.height,
              // true);
            }
            dev->unmap_buffer(readback);
          }
          dev->end_frame();
        }
        // fprintf(stdout, "Buffer test finished.\n");
      }

      RenderDoc_CTX::end();
      dev->release();
      return 0;
    };
    if (argc < 10)
      RETERR(
          "Usage: ./dispatch_kernel.exe vk/dx kernel.hlsl NUM_GROUPS_X NUM_GROUPS_Y NUM_GROUPS_Z "
          "NUM_OF_LAUNCHES NUM_ITEMS "
          "UAV_TEXTURE_SIZE "
          "PRINT_RESULT=0,1 "
          "[PARAM0 "
          "PARAM1 ... PARAM7]");
    // fprintf(stdout, "Testing Vulkan backend\n");
    if (stref_s(argv[1]) == stref_s("vk"))
      return launch_tests(rd::create_vulkan(NULL));
    else
      return launch_tests(rd::create_dx12(NULL));
    // return launch_tests(rd::create_vulkan(NULL));
    // fprintf(stdout, "Testing Dx12 backend\n");
    // launch_tests(rd::create_dx12(NULL));
  }
  return 0;
}
