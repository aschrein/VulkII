#ifndef RENDERING_UTILS_HPP
#define RENDERING_UTILS_HPP

#include "rendering.hpp"
#include "scene.hpp"
#include "simplefont.h"
#include "utils.hpp"
#include <mutex>

#ifdef __linux__
#  include <SDL2/SDL.h>
#  include <SDL2/SDL_syswm.h>
#  include <X11/Xlib-xcb.h>
#  include <xcb/xcb.h>
#else
#  include <SDL.h>
#  include <SDL_syswm.h>
#endif

#include <imgui.h>
#include <imgui/examples/imgui_impl_sdl.h>

f32         max3(float3 const &a) { return MAX3(a.x, a.y, a.z); }
static void setup_default_state(rd::Graphics_Pipeline_State &state, u32 num_rts = 1) {
  rd::Blend_State bs;
  MEMZERO(bs);
  bs.enabled          = false;
  bs.color_write_mask = (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
                        (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
  ito(num_rts) state.OM_set_blend_state(i, bs);
  state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
  rd::RS_State rs_state;
  MEMZERO(rs_state);
  rs_state.polygon_mode = rd::Polygon_Mode::FILL;
  rs_state.front_face   = rd::Front_Face::CW;
  rs_state.cull_mode    = rd::Cull_Mode::NONE;
  state.RS_set_state(rs_state);
  rd::DS_State ds_state;
  MEMZERO(ds_state);
  ds_state.cmp_op             = rd::Cmp::EQ;
  ds_state.enable_depth_test  = false;
  ds_state.enable_depth_write = false;
  state.DS_set_state(ds_state);
  rd::MS_State ms_state;
  MEMZERO(ms_state);
  ms_state.sample_mask = 0xffffffffu;
  ms_state.num_samples = 1;
  state.MS_set_state(ms_state);
}

struct ImGui_ID {
  Resource_ID     id;
  u32             base_level;
  u32             base_layer;
  float           min;
  float           max;
  rd::Format      format;
  static ImGui_ID def(Resource_ID id) {
    ImGui_ID iid;
    MEMZERO(iid);
    iid.id     = id;
    iid.format = rd::Format::NATIVE;
    return iid;
  }
};

class IGUIApp {
  protected:
  InlineArray<ImGui_ID, 0x100> image_bindings{};

  i32          last_m_x          = 0;
  i32          last_m_y          = 0;
  ImDrawData * draw_data         = NULL;
  Timer        timer             = {};
  bool         imgui_initialized = false;
  rd::IDevice *factory           = NULL;
  SDL_Window * window            = NULL;

  virtual void on_gui() {}
  virtual void on_init() {}
  virtual void on_release() {}
  virtual void on_frame() {}

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
  }
  void _start(rd::Impl_t impl) {
    static int  init   = [] { return SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS); }();
    SDL_Window *window = NULL;
    {
      static std::mutex           mutex;
      std::lock_guard<std::mutex> _locke(mutex);
      window = SDL_CreateWindow(impl == rd::Impl_t::VULKAN ? "VulkII on Vulkan" : "VulkII on DX12",
                                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 512, 512,
                                SDL_WINDOW_RESIZABLE);
    }
    void *handle = NULL;
#ifdef WIN32
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;
    handle    = (void *)hwnd;
#else
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    Window            hwnd       = wmInfo.info.x11.window;
    xcb_connection_t *connection = XGetXCBConnection(wmInfo.info.x11.display);
    Ptr2              ptrs{(void *)hwnd, (void *)connection};
    handle = (void *)&ptrs;
#endif
    if (impl == rd::Impl_t::VULKAN) {
      factory = rd::create_vulkan(handle);
    } else if (impl == rd::Impl_t::DX12) {
      factory = rd::create_dx12(handle);
    } else {
      TRAP;
    }
    if (factory == NULL) return;

    factory->start_frame();
    on_init();

    constexpr u32 MAX_FRAMES = 0x10;
    u32           NUM_FRAMES = factory->get_num_swapchain_images();
    Resource_ID   frame_buffers[MAX_FRAMES]{};
    Resource_ID   render_pass{};
    Resource_ID   pso{};
    defer({
      ito(NUM_FRAMES) {
        if (frame_buffers[i].is_valid()) factory->release_resource(frame_buffers[i]);
      }
      if (render_pass.is_valid()) factory->release_resource(render_pass);
      if (pso.is_valid()) factory->release_resource(pso);
    });
    u32  frame_id = 0;
    u32  width = 0, height = 0;
    u32  window_id    = SDL_GetWindowID(window);
    bool focus_events = false;
    struct PushConstants {
      float2 uScale;
      float2 uTranslate;
      u32    control_flags;
      float  min;
      float  max;
    };
    static string_ref imgui_shader = stref_s(R"(
struct PushConstants
{
  float2 uScale;
  float2 uTranslate;
  u32    control_flags;
  float min;
  float max;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

#define CONTROL_DEPTH_ENABLE 1
#define CONTROL_UINT         2
#define is_control(flag) (pc.control_flags & flag) != 0

[[vk::binding(0, 0)]] Texture2D<float4>   sTexture          : register(t0, space0);
[[vk::binding(1, 0)]] Texture2D<uint>     uTexture          : register(t1, space0);
[[vk::binding(2, 0)]] SamplerState        sSampler          : register(s2, space0);

struct PSInput {
  [[vk::location(0)]] float4 pos   : SV_POSITION;
  [[vk::location(1)]] float2 UV    : TEXCOORD0;
  [[vk::location(2)]] float4 Color : TEXCOORD1;
};

#ifdef VERTEX

struct VSInput {
  [[vk::location(0)]] float2 aPos    : POSITION;
  [[vk::location(1)]] float2 aUV     : TEXCOORD0;
  [[vk::location(2)]] float4 aColor  : TEXCOORD1;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.Color = input.aColor;
  output.UV    = input.aUV;
  output.pos   = float4(input.aPos * pc.uScale + pc.uTranslate, 0.0f, 1.0f);
  output.pos.y *= -1.0f;
  return output;
}
#endif
#ifdef PIXEL

float4 main(in PSInput input) : SV_TARGET0 {
  if (is_control(CONTROL_DEPTH_ENABLE)) {
    float depth = sTexture.Sample(sSampler, input.UV).r;
    depth = pow(depth * 5.0f, 1.0f / 2.0f);
    return float4_splat((depth - pc.min) / (pc.max - pc.min));
  } else if (is_control(CONTROL_UINT)) {
    u32 width, height;
    uTexture.GetDimensions(width, height);
    uint val = uTexture.Load(int3((input.UV.xy) * float2(width, height), 0));
    return float4(float3_splat(float(val) / 20.0f), 1.0f);
  } else {
    float4 color = input.Color * sTexture.Sample(sSampler, input.UV);
    color = (color - float4_splat(pc.min)) / float4_splat(pc.max - pc.min);
    return color;
  }
}
#endif
)");
    Resource_ID       signature    = [=] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return factory->create_signature(table_info);
    }();
    defer(factory->release_resource(signature));
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    Resource_ID vs = factory->create_shader(rd::Stage_t::VERTEX, imgui_shader, &defines[0], 1);
    Resource_ID ps = factory->create_shader(rd::Stage_t::PIXEL, imgui_shader, &defines[1], 1);
    defer({
      factory->release_resource(vs);
      factory->release_resource(ps);
    });

    Resource_ID sampler = [=] {
      rd::Sampler_Create_Info info;
      MEMZERO(info);
      info.address_mode_u = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_v = rd::Address_Mode::CLAMP_TO_EDGE;
      info.address_mode_w = rd::Address_Mode::CLAMP_TO_EDGE;
      info.mag_filter     = rd::Filter::NEAREST;
      info.min_filter     = rd::Filter::NEAREST;
      info.mip_mode       = rd::Filter::NEAREST;
      info.anisotropy     = false;
      return factory->create_sampler(info);
    }();
    IMGUI_CHECKVERSION();
    ImGuiContext *imgui_ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(imgui_ctx);
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    {
      static std::mutex           mutex;
      std::lock_guard<std::mutex> _locke(mutex);
      if (factory->getImplType() == rd::Impl_t::VULKAN) {
        ImGui_ImplSDL2_InitForVulkan(window);
      } else if (factory->getImplType() == rd::Impl_t::DX12) {
        ImGui_ImplSDL2_InitForD3D(window);
      } else {
        TRAP;
      }
    }
    Resource_ID font_texture = [&] {
      static std::mutex           mutex;
      std::lock_guard<std::mutex> _locke(mutex);
      unsigned char *             font_pixels = NULL;
      int                         font_width = 0, font_height = 0;
      io.Fonts->GetTexDataAsRGBA32(&font_pixels, &font_width, &font_height);

      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format = rd::Format::RGBA8_UNORM;
      info.width  = font_width;
      info.height = font_height;
      info.depth  = 1;
      info.layers = 1;
      info.levels = 1;
      info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
      Resource_ID font_texture = factory->create_image(info);
      io.Fonts->TexID          = (ImTextureID)bind_texture(font_texture, 0, 0, rd::Format::NATIVE);

      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
      u32 pitch = rd::IDevice::align_up(font_width * 4, rd::IDevice::TEXTURE_DATA_PITCH_ALIGNMENT);
      buf_info.size              = font_height * pitch;
      Resource_ID staging_buffer = factory->create_buffer(buf_info);
      factory->release_resource(staging_buffer);

      u8 *dst = (u8 *)factory->map_buffer(staging_buffer);
      yto(font_height) {
        xto(font_width) {
          memcpy(dst + pitch * y, font_pixels + font_width * 4 * y, font_width * 4);
        }
      }
      factory->unmap_buffer(staging_buffer);
      rd::ICtx *ctx = factory->start_compute_pass();
      ctx->image_barrier(font_texture, rd::Image_Access::TRANSFER_DST);
      ctx->copy_buffer_to_image(staging_buffer, 0, font_texture, rd::Image_Copy::top_level(pitch));
      factory->end_compute_pass(ctx);
      font_pixels = NULL;
      return font_texture;
    }();
    {
      rd::Graphics_Pipeline_State gfx_state{};
      setup_default_state(gfx_state);
      rd::Blend_State bs;
      MEMZERO(bs);
      bs.enabled        = true;
      bs.alpha_blend_op = rd::Blend_OP::ADD;
      bs.color_blend_op = rd::Blend_OP::ADD;
      bs.dst_alpha      = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
      bs.src_alpha      = rd::Blend_Factor::SRC_ALPHA;
      bs.dst_color      = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
      bs.src_color      = rd::Blend_Factor::SRC_ALPHA;
      bs.color_write_mask =
          (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
          (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
      gfx_state.OM_set_blend_state(0, bs);
      gfx_state.VS_set_shader(vs);
      gfx_state.PS_set_shader(ps);
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 0;
        info.offset   = 0;
        info.type     = rd::Attriute_t::POSITION;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RG32_FLOAT;
        info.location = 1;
        info.offset   = 8;
        info.type     = rd::Attriute_t::TEXCOORD0;
        gfx_state.IA_set_attribute(info);
      }
      {
        rd::Attribute_Info info;
        MEMZERO(info);
        info.binding  = 0;
        info.format   = rd::Format::RGBA8_UNORM;
        info.location = 2;
        info.offset   = 16;
        info.type     = rd::Attriute_t::TEXCOORD1;
        gfx_state.IA_set_attribute(info);
      }
      gfx_state.IA_set_vertex_binding(0, sizeof(ImDrawVert), rd::Input_Rate::VERTEX);
      rd::Render_Pass_Create_Info info{};
      rd::RT_Ref                  rt0{};
      rt0.format            = factory->get_swapchain_image_info().format;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      rt0.clear_color.g     = 0.0f;
      rt0.clear_color.b     = 0.0f;
      rt0.clear_color.a     = 0.0f;
      info.rts.push(rt0);
      render_pass = factory->create_render_pass(info);
      pso         = factory->create_graphics_pso(signature, render_pass, gfx_state);
    }
    factory->end_frame();
    // static InlineArray<SDL_Event, 0x100> sdl_event_queue;
    // static std::mutex                    sdl_event_mutex;
    // static std::atomic<int>              sdl_event_cnt;
    while (true) {
      ImGui::SetCurrentContext(imgui_ctx);

      SDL_Event event{};
      {
        // std::lock_guard<std::mutex> _locke(sdl_event_mutex);
        while (SDL_PollEvent(&event)) {
          // SDL_Event event = sdl_event_queue[i];
          if (focus_events) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
            }
          }
          if (event.type == SDL_WINDOWEVENT) {
            if (event.window.windowID != window_id) continue;
            if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
              goto exit_loop;
            } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED ||
                       event.window.event == SDL_WINDOWEVENT_ENTER) {
              focus_events = true;
            } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST ||
                       event.window.event == SDL_WINDOWEVENT_LEAVE) {
              focus_events = false;
            } else if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
            }
          }
          // sdl_event_queue.push(event);
          // sdl_event_cnt = 2;
        }
        /* if (int val = sdl_event_cnt.fetch_add(-1) > 0) {
           defer({
             if (val == 1) {
               sdl_event_queue.size = 0;
             }
           });
           ito(sdl_event_queue.size) {
             SDL_Event event = sdl_event_queue[i];
             if (focus_events) {
               ImGui_ImplSDL2_ProcessEvent(&event);
               if (event.type == SDL_QUIT) {
               }
             }
             if (event.type == SDL_WINDOWEVENT) {
               if (event.window.windowID != window_id) continue;
               if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                 return;
               } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED ||
                          event.window.event == SDL_WINDOWEVENT_ENTER) {
                 focus_events = true;
               } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST ||
                          event.window.event == SDL_WINDOWEVENT_LEAVE) {
                 focus_events = false;
               } else if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
               }
             }
           }
         }*/
      }
      factory->start_frame();
      rd::Image2D_Info scinfo = factory->get_swapchain_image_info();
      if (width != scinfo.width || height != scinfo.height) {
        width  = scinfo.width;
        height = scinfo.height;
        ito(NUM_FRAMES) {
          if (frame_buffers[i].is_valid()) {
            factory->release_resource(frame_buffers[i]);
            frame_buffers[i] = {};
          };
        }
      }
      if (frame_buffers[frame_id].is_null()) {
        rd::Frame_Buffer_Create_Info info{};
        rd::RT_View                  rt0{};
        rt0.image  = factory->get_swapchain_image();
        rt0.format = rd::Format::NATIVE;
        info.rts.push(rt0);
        frame_buffers[frame_id] = factory->create_frame_buffer(render_pass, info);
      }
      on_frame();
      rd::ICtx *ctx = factory->start_render_pass(render_pass, frame_buffers[frame_id]);
      {
        TracyVulkIINamedZone(ctx, "ImGui Rendering Pass");
        image_bindings.reset();
        io.Fonts->TexID      = (ImTextureID)bind_texture(font_texture, 0, 0, rd::Format::NATIVE);
        auto         sc_info = factory->get_swapchain_image_info();
        static float angle   = 0.0f;
        angle += 1.0e-3f;
        // ctx->set_scissor(0, 0, sc_info.width, sc_info.height);
        ctx->set_viewport(0.0f, 0.0f, (f32)sc_info.width, (f32)sc_info.height, 0.0f, 1.0f);
        // ImGui Pass
        {
          ImGui_ImplSDL2_NewFrame(window);
          ImGui::NewFrame();
          ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
          ImGuiViewport *  viewport     = ImGui::GetMainViewport();
          ImGui::SetNextWindowPos(viewport->Pos);
          ImGui::SetNextWindowSize(viewport->Size);
          ImGui::SetNextWindowViewport(viewport->ID);
          ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
          ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
          window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                          ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
          window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
          window_flags |= ImGuiWindowFlags_NoBackground;
          ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
          ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 1.0f);
          ImGui::SetNextWindowBgAlpha(-1.0f);
          ImGui::Begin("DockSpace", nullptr, window_flags);
          ImGui::PopStyleVar(4);
          ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
          ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                           ImGuiDockNodeFlags_PassthruCentralNode);
          on_gui();
          ImGui::End();

          // static bool show_demo_window = true;
          // ImGui::ShowDemoWindow(&show_demo_window);

          ImGui::Render();
        }
        ImDrawData *draw_data = ImGui::GetDrawData();
        // Render IMGUI
        {
          Resource_ID vertex_buffer = [=] {
            rd::Buffer_Create_Info buf_info;
            MEMZERO(buf_info);
            buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
            buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
            buf_info.size        = (draw_data->TotalVtxCount + 1) * sizeof(ImDrawVert);
            return factory->create_buffer(buf_info);
          }();
          factory->release_resource(vertex_buffer);
          Resource_ID index_buffer = [=] {
            rd::Buffer_Create_Info buf_info;
            MEMZERO(buf_info);
            buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
            buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
            buf_info.size        = (draw_data->TotalIdxCount + 1) * sizeof(ImDrawIdx);
            return factory->create_buffer(buf_info);
          }();
          factory->release_resource(index_buffer);
          {
            {
              ImDrawVert *vtx_dst = (ImDrawVert *)factory->map_buffer(vertex_buffer);

              ito(draw_data->CmdListsCount) {

                const ImDrawList *cmd_list = draw_data->CmdLists[i];
                memcpy(vtx_dst, cmd_list->VtxBuffer.Data,
                       cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));

                vtx_dst += cmd_list->VtxBuffer.Size;
              }
              factory->unmap_buffer(vertex_buffer);
            }
            {
              ImDrawIdx *idx_dst = (ImDrawIdx *)factory->map_buffer(index_buffer);
              ito(draw_data->CmdListsCount) {

                const ImDrawList *cmd_list = draw_data->CmdLists[i];

                memcpy(idx_dst, cmd_list->IdxBuffer.Data,
                       cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
                idx_dst += cmd_list->IdxBuffer.Size;
              }
              factory->unmap_buffer(index_buffer);
            }
          }
          ImVec2 clip_off          = draw_data->DisplayPos;
          ImVec2 clip_scale        = draw_data->FramebufferScale;
          int    global_vtx_offset = 0;
          int    global_idx_offset = 0;
          ito(image_bindings.size) {
            ctx->image_barrier(image_bindings[i].id, rd::Image_Access::SAMPLED);
          }
          ctx->start_render_pass();
          ctx->bind_graphics_pso(pso);
          if (sizeof(ImDrawIdx) == 2)
            ctx->bind_index_buffer(index_buffer, 0, rd::Index_t::UINT16);
          else
            ctx->bind_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
          ctx->bind_vertex_buffer(0, vertex_buffer, 0);
          PushConstants pc{};
          for (int n = 0; n < draw_data->CmdListsCount; n++) {
            const ImDrawList *cmd_list = draw_data->CmdLists[n];
            for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
              const ImDrawCmd *pcmd = &cmd_list->CmdBuffer[cmd_i];
              ImVec4           clip_rect;
              clip_rect.x  = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
              clip_rect.y  = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
              clip_rect.z  = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
              clip_rect.w  = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;
              ImGui_ID img = image_bindings[(size_t)pcmd->TextureId];
              if (img.format == rd::Format::D32_OR_R32_FLOAT) {
                pc.control_flags = 1;
              } else if (img.format == rd::Format::R32_UINT) {
                pc.control_flags = 2;
              } else {
                pc.control_flags = 0;
              }
              pc.min                    = img.min;
              pc.max                    = img.max;
              rd::IBinding_Table *table = factory->create_binding_table(signature);
              defer(table->release());
              {
                float2 scale;
                scale[0] = 2.0f / draw_data->DisplaySize.x;
                scale[1] = 2.0f / draw_data->DisplaySize.y;
                float2 translate;
                translate[0]  = -1.0f - draw_data->DisplayPos.x * scale[0];
                translate[1]  = -1.0f - draw_data->DisplayPos.y * scale[1];
                pc.uScale     = scale;
                pc.uTranslate = translate;
                table->push_constants(&pc, 0, sizeof(pc));
              }
              rd::Image_Subresource range{};
              range.layer      = img.base_layer;
              range.level      = img.base_level;
              range.num_layers = 1;
              range.num_levels = 1;
              table->bind_sampler(0, 2, sampler);
              if (pc.control_flags == 2)
                table->bind_texture(0, 1, 0, img.id, range, img.format);
              else
                table->bind_texture(0, 0, 0, img.id, range, img.format);
              ctx->bind_table(table);

              ctx->set_scissor(clip_rect.x, clip_rect.y, clip_rect.z - clip_rect.x,
                               clip_rect.w - clip_rect.y);
              ctx->draw_indexed(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, 0,
                                pcmd->VtxOffset + global_vtx_offset);
            }
            global_idx_offset += cmd_list->IdxBuffer.Size;
            global_vtx_offset += cmd_list->VtxBuffer.Size;
          }
          ctx->end_render_pass();
        }
      }
      factory->end_render_pass(ctx);
      factory->end_frame();
      // factory->wait_idle();
      frame_id = (frame_id + 1) % NUM_FRAMES;
    }
  exit_loop:
    on_release();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext(imgui_ctx);
  }
  ImTextureID bind_texture(Resource_ID id, u32 layer, u32 level, rd::Format format,
                           float min = 0.0f, float max = 1.0f) {
    ImGui_ID iid;
    MEMZERO(iid);
    iid.id         = id;
    iid.base_layer = layer;
    iid.base_level = level;
    iid.min        = min;
    iid.max        = max;
    if (format == rd::Format::NATIVE) format = factory->get_image_info(id).format;
    iid.format = format;
    image_bindings.push(iid);
    return (ImTextureID)(size_t)(image_bindings.size - 1);
  }
  ~IGUIApp() = default;
  IGUIApp()  = default;

  public:
  template <typename T> static void start(rd::Impl_t impl_t) {
    T t;
    t._start(impl_t);
  }
};

static ImVec2 get_window_size() {
  auto  wsize       = ImGui::GetWindowSize();
  float height_diff = 42;
  if (wsize.y < height_diff + 2) {
    wsize.y = 2;
  } else {
    wsize.y = wsize.y - height_diff;
  }
  return wsize;
}

static void init_buffer(rd::IDevice *factory, Resource_ID buf, void const *src, size_t size) {
  rd::ICtx *             ctx = factory->start_compute_pass();
  rd::Buffer_Create_Info info;
  MEMZERO(info);
  info.memory_type    = rd::Memory_Type::CPU_WRITE_GPU_READ;
  info.usage_bits     = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
  info.size           = size;
  Resource_ID staging = factory->create_buffer(info);
  void *      dst     = factory->map_buffer(staging);
  memcpy(dst, src, size);
  factory->unmap_buffer(staging);
  ctx->copy_buffer(staging, 0, buf, 0, size);
  factory->end_compute_pass(ctx);
  factory->release_resource(staging);
}

#if 1

template <typename T> static Resource_ID create_uniform(rd::IDevice *factory, T const &src) {

  rd::Buffer_Create_Info info;
  MEMZERO(info);
  info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
  info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
  info.size        = sizeof(T);
  Resource_ID out  = factory->create_buffer(info);
  void *      dst  = factory->map_buffer(out);
  memcpy(dst, &src, sizeof(src));
  factory->unmap_buffer(out);
  return out;
}

static Resource_ID get_or_create_image(rd::IDevice *factory, rd::Image_Create_Info const &ci,
                                       Resource_ID old) {
  bool create = false;
  if (old.is_null()) create = true;
  if (!create) {
    auto img_info = factory->get_image_info(old);
    create        = img_info.width != ci.width || img_info.height != ci.height;
  }
  if (create) {
    if (old.is_null() == false) factory->release_resource(old);
    return factory->create_image(ci);
  } else {
    return old;
  }
}

struct TimeStamp_Pool {
  static constexpr u32 NUM_TIMESTAMPS = 0x10;

  u64 head = 0;
  u64 tail = 0;
  struct Query {
    bool        in_fly = false;
    Resource_ID t0;
    Resource_ID t1;
    Resource_ID e;
  };
  i32    cur_query_id = -1;
  Query  timestamps[NUM_TIMESTAMPS]{};
  double duration       = 0.0;
  double total_duration = 0.0;
  u64    total_samples  = 0;
  double min_sample     = 1.0e6;
  double max_sample     = 0.0;

  void init(rd::IDevice *factory) {
    ito(NUM_TIMESTAMPS) {
      timestamps[i].in_fly = false;
      timestamps[i].t0     = factory->create_timestamp();
      timestamps[i].t1     = factory->create_timestamp();
    }
  }
  void begin_range(rd::ICtx *ctx) {
    ito(NUM_TIMESTAMPS) {
      if (timestamps[i].in_fly == false) {
        cur_query_id = i;
        ctx->insert_timestamp(timestamps[cur_query_id].t0);
        return;
      }
    }
    cur_query_id = -1;
  }
  void end_range(rd::ICtx *ctx) {
    if (cur_query_id == -1) return;
    ctx->insert_timestamp(timestamps[cur_query_id].t1);
  }
  void commit(Resource_ID e) {
    if (cur_query_id == -1) return;
    timestamps[cur_query_id].e      = e;
    timestamps[cur_query_id].in_fly = true;
  }
  void update(rd::IDevice *factory) {
    ito(NUM_TIMESTAMPS) {
      if (timestamps[i].in_fly) {
        bool ready = factory->get_event_state(timestamps[i].e);
        if (ready) {
          timestamps[i].in_fly = false;
          double cur_dur       = factory->get_timestamp_ms(timestamps[i].t0, timestamps[i].t1);
          total_duration += cur_dur;
          total_samples += 1;
          min_sample = MIN(min_sample, cur_dur);
          max_sample = MAX(max_sample, cur_dur);
          duration += (cur_dur - duration) * 0.05;
        }
      }
    }
    // if (tail + 1 >= head) return;
    // u32 cnt = 0;
    // while (tail + 1 <= head) {
    //  bool ready = factory->get_event_state(timestamps[(tail) % NUM_TIMESTAMPS].e);
    //  if (ready) {
    //    tail++;
    //    cnt++;
    //    double cur_dur = factory->get_timestamp_ms(timestamps[(tail) % NUM_TIMESTAMPS].t0,
    //                                               timestamps[(tail) % NUM_TIMESTAMPS].t1);
    //    fprintf(stdout, "cur duration : %f\n", cur_dur);
    //    duration += (cur_dur - duration) * 0.05;

    //  } else {
    //    fprintf(stdout, "Not Ready!\n");
    //    break;
    //  }
    //  tail += 1;
    //}
  }
  void release(rd::IDevice *factory) {
    ito(NUM_TIMESTAMPS) {
      factory->release_resource(timestamps[i].t0);
      factory->release_resource(timestamps[i].t1);
    }
  }
};
#endif // 0
#if 1
class Mip_Builder {
  ~Mip_Builder() = default;

  rd::IDevice *dev = NULL;
  // TimeStamp_Pool timestamp{};

#  define RESOURCE_LIST                                                                            \
    RESOURCE(signature);                                                                           \
    RESOURCE(mip_shader);                                                                          \
    RESOURCE(reset_cnt_shader);                                                                    \
    RESOURCE(reset_cnt_pso);                                                                       \
    RESOURCE(mip_pso);

#  define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#  undef RESOURCE
  struct PushConstants {
    u32 gamma_correct;
    u32 levels;
    u32 op;
    u32 total_cnt;
    u32 last_mip;
    u32 last_width;
    u32 last_height;
  };
  void init(rd::IDevice *dev) {
    this->dev = dev;
    // timestamp.init(dev);
    TMP_STORAGE_SCOPE;
    signature = [=] {
      rd::Binding_Space_Create_Info set_info{};

      set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
      // set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
      set_info.bindings.push({rd::Binding_t::UAV_BUFFER, 1});
      set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::UAV_TEXTURE, 16});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    reset_cnt_shader = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(

[[vk::binding(1, 0)]] globallycoherent RWByteAddressBuffer counter : register(u1, space0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<uint>   counter_image : register(u2, space0);

[numthreads(32, 32, 1)] void main(uint3 tid : SV_DispatchThreadID) {
  if (all(tid == uint3(0, 0, 0)))
    counter.Store<u32>(0, 0);
  int2 size;
  counter_image.GetDimensions(size.x, size.y);
  if (all(tid.xy < size))
    counter_image[tid.xy] = 0;
}
)"),
                                          NULL, 0);
    reset_cnt_pso    = dev->create_compute_pso(signature, reset_cnt_shader);
    mip_shader       = dev->create_shader(rd::Stage_t::COMPUTE, stref_s(R"(
struct PushConstants {
  u32 gamma_correct;
  u32 levels;
  u32 op;
  u32 total_cnt;
  u32 last_mip;
  u32 last_width;
  u32 last_height;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

#define RGBA8_SRGBA 0
#define RGBA8_UNORM 1
#define RGB32_FLOAT 2
#define R32_FLOAT 3
#define RGBA32_FLOAT 4

#define OP_AVG 0
#define OP_MAX 1
#define OP_MIN 2
#define OP_SUM 3


[[vk::binding(0, 0)]] Texture2D<float4>                    src_tex : register(t0, space0);
[[vk::binding(1, 0)]] globallycoherent RWByteAddressBuffer counter : register(u1, space0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<uint>   counter_image : register(u2, space0);
[[vk::binding(3, 0)]] globallycoherent RWTexture2D<float4> tex[16] : register(u3, space0);

groupshared u32                                            group_cnt;

void downsample(int mip_i, int2 dst_xy, int2 src_mip_size) {
  int2   xy  = int2(dst_xy * 2);
  float  N   = 0.0f;
  float4 acc = float4_splat(0.0f);

  if (xy.x == src_mip_size.x - 3) {
    acc += tex[mip_i - 1].Load(int3(xy + int2(2, 0), 0));
    N += 1.0f;
  }
  if (xy.y == src_mip_size.y - 3) {
    acc += tex[mip_i - 1].Load(int3(xy + int2(0, 2), 0));
    N += 1.0f;
  }
  if (xy.x == src_mip_size.x - 3 && xy.y == src_mip_size.y - 3) {
    acc += tex[mip_i - 1].Load(int3(xy + int2(2, 2), 0));
    N += 1.0f;
  }
  acc += tex[mip_i - 1].Load(int3(xy + int2(0, 0), 0));
  acc += tex[mip_i - 1].Load(int3(xy + int2(1, 0), 0));
  acc += tex[mip_i - 1].Load(int3(xy + int2(0, 1), 0));
  acc += tex[mip_i - 1].Load(int3(xy + int2(1, 1), 0));
  N += 4.0f;
  tex[mip_i][dst_xy] = acc / N;
}

[numthreads(32, 32, 1)] void main(uint3 tid
                                  : SV_DispatchThreadID, uint3 gtid
                                  : SV_GroupThreadID, uint3    gid
                                  : SV_GroupID) {
  int2 size;
  tex[0].GetDimensions(size.x, size.y);
  {
    if (all(tid.xy * 2 < size)) {
      // Copy top level
      // Gamma correct?
      if (pc.gamma_correct) {
        tex[0][tid.xy * 2 + int2(0, 0)] = pow(src_tex.Load(int3(tid.xy * 2 + int2(0, 0), 0)), 0.5f);
        tex[0][tid.xy * 2 + int2(1, 0)] = pow(src_tex.Load(int3(tid.xy * 2 + int2(1, 0), 0)), 0.5f);
        tex[0][tid.xy * 2 + int2(0, 1)] = pow(src_tex.Load(int3(tid.xy * 2 + int2(0, 1), 0)), 0.5f);
        tex[0][tid.xy * 2 + int2(1, 1)] = pow(src_tex.Load(int3(tid.xy * 2 + int2(1, 1), 0)), 0.5f);
      } else {
        tex[0][tid.xy * 2 + int2(0, 0)] = src_tex.Load(int3(tid.xy * 2 + int2(0, 0), 0));
        tex[0][tid.xy * 2 + int2(1, 0)] = src_tex.Load(int3(tid.xy * 2 + int2(1, 0), 0));
        tex[0][tid.xy * 2 + int2(0, 1)] = src_tex.Load(int3(tid.xy * 2 + int2(0, 1), 0));
        tex[0][tid.xy * 2 + int2(1, 1)] = src_tex.Load(int3(tid.xy * 2 + int2(1, 1), 0));
      }
    }
    GroupMemoryBarrierWithGroupSync();
    int2 src_mip_size         = size;
    int2 dst_mip_size         = max(int2(1, 1), size / 2);
    u32  pixels_per_group = 32;
    u32  mip_i            = 1;
    int2 group_offset     = gid.xy;

    for (; mip_i < pc.levels; mip_i++) {
      GroupMemoryBarrierWithGroupSync();
      int2 xy = int2(group_offset.xy * pixels_per_group + gtid.xy);
      if (all(xy < dst_mip_size) && all(gtid < pixels_per_group))
        downsample(mip_i, xy, src_mip_size);
      // Scalar break
      if (all(dst_mip_size == int2(1, 1))) break;
      pixels_per_group = pixels_per_group / 2;
      if (pixels_per_group == 0) break;
      src_mip_size = dst_mip_size;
      dst_mip_size = max(int2(1, 1), src_mip_size / 2);
    }
  }
  //group_cnt = 0;
  GroupMemoryBarrierWithGroupSync();
  if (gtid.x == 0 && gtid.y == 0) {
    u32 cnt;
    counter.InterlockedAdd(0, 1, cnt);
    group_cnt = cnt;
    //tex[0][tid.xy] = float4(float(group_cnt), 1.0f, 1.0f, 1.0f);
  }
  GroupMemoryBarrierWithGroupSync();
  // The last group
  if (group_cnt == pc.total_cnt - 1) {
    //tex[0][gtid.xy] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    u32 pixels_per_group = 32;
    u32  mip_i            = pc.last_mip + 1;
    int2 mip_size = int2(pc.last_width, pc.last_height);
    int2 src_mip_size         = mip_size;
    int2 dst_mip_size         = max(int2(1, 1), mip_size / 2);
    for (; mip_i < pc.levels; mip_i++) {
      GroupMemoryBarrierWithGroupSync();
      for (u32 y = 0; y < pc.last_height; y += pixels_per_group) {
        for (u32 x = 0; x < pc.last_width; x += pixels_per_group) {
          int2 xy = int2(gtid.xy + int2(x, y));
          if (all(xy < dst_mip_size))
            downsample(mip_i, xy, src_mip_size);
        }
      }
      // Scalar break
      if (all(dst_mip_size == int2(1, 1))) break;
      pixels_per_group = pixels_per_group / 2;
      if (pixels_per_group == 0) break;
      src_mip_size = dst_mip_size;
      dst_mip_size = max(int2(1, 1), src_mip_size / 2);
    }
  }
}

)"),
                                    NULL, 0);
    mip_pso          = dev->create_compute_pso(signature, mip_shader);
  }

  public:
  static Mip_Builder *create(rd::IDevice *dev) {
    Mip_Builder *o = new Mip_Builder;
    o->init(dev);
    return o;
  }

  void release() {
#  define RESOURCE(name)                                                                           \
    if (name.is_valid()) dev->release_resource(name);
    RESOURCE_LIST
#  undef RESOURCE
    // timestamp.release(dev);
    delete this;
  }
#  undef RESOURCE_LIST

  enum Filter { FILTER_BOX, FILTER_MIN, FILTER_MAX };
  Resource_ID create_image(rd::IDevice *factory, Image2D const *image,
                           u32  _usage    = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED,
                           bool build_mip = true, Filter filter = FILTER_BOX) {
    Resource_ID output_image = [=] {
      rd::Image_Create_Info info{};
      u32                   usage = _usage | (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                  (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST |
                  (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC;
      info.format = image->format;
      info.width  = image->width;
      info.height = image->height;
      info.depth  = 1;
      info.layers = 1;
      if (build_mip)
        info.levels = image->get_num_mip_levels();
      else
        info.levels = 1;
      info.usage_bits = usage;
      return factory->create_image(info);
    }();
    size_t      pitch = rd::IDevice::align_up(image->width * Image2D::get_bpp(image->format),
                                         rd::IDevice::TEXTURE_DATA_PITCH_ALIGNMENT);
    Resource_ID staging_buffer = [&] {
      rd::Buffer_Create_Info buf_info{};
      buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = pitch * image->height;
      return factory->create_buffer(buf_info);
    }();
    defer(factory->release_resource(staging_buffer));
    u8 *ptr = (u8 *)factory->map_buffer(staging_buffer);
    ito(image->height) {
      memcpy(ptr + pitch * i,
             image->data + (size_t)image->width * Image2D::get_bpp(image->format) * i,
             (size_t)image->width * Image2D::get_bpp(image->format));
    }
    // memcpy(ptr, image->data, image->get_size_in_bytes());
    factory->unmap_buffer(staging_buffer);

    Resource_ID image_staging_buffer = [&] {
      rd::Buffer_Create_Info buf_info{};
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = pitch * image->height;
      return factory->create_buffer(buf_info);
    }();
    defer(factory->release_resource(image_staging_buffer));

    {
      rd::ICtx *ctx = factory->start_compute_pass();
      ctx->buffer_barrier(image_staging_buffer, rd::Buffer_Access::TRANSFER_DST);
      ctx->copy_buffer(staging_buffer, 0, image_staging_buffer, 0, pitch * image->height);
      ctx->buffer_barrier(image_staging_buffer, rd::Buffer_Access::TRANSFER_SRC);
      ctx->image_barrier(output_image, rd::Image_Access::TRANSFER_DST);
      ctx->copy_buffer_to_image(image_staging_buffer, 0, output_image,
                                rd::Image_Copy::top_level(pitch));
      factory->end_compute_pass(ctx);
    }
    if (build_mip) return build_mips(factory, output_image, filter);
    return output_image;
  }

  Resource_ID build_mips(rd::IDevice *factory, Resource_ID output_image,
                         Filter filter = FILTER_BOX) {
    auto        desc      = factory->get_image_info(output_image);
    Resource_ID tmp_image = [=] {
      rd::Image_Create_Info info{};
      u32 usage = (u32)rd::Image_Usage_Bits::USAGE_UAV | (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                  (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST |
                  (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_SRC;
      info.format     = rd::Format::RGBA32_FLOAT;
      info.width      = desc.width;
      info.height     = desc.height;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = desc.levels;
      info.usage_bits = usage;
      return factory->create_image(info);
    }();
    defer(factory->release_resource(output_image));
    // defer(factory->release_resource(tmp_image));
    Resource_ID cnt_buffer = [&] {
      rd::Buffer_Create_Info buf_info{};
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
      buf_info.size        = 4;
      return factory->create_buffer(buf_info);
    }();
    defer(factory->release_resource(cnt_buffer));
    /*((u32 *)factory->map_buffer(cnt_buffer))[0] = 0;
    factory->unmap_buffer(cnt_buffer);*/
    rd::ICtx *ctx = factory->start_compute_pass();
    {
      TracyVulkIINamedZone(ctx, "Build Mips");
      // timestamp.begin_range(ctx);
      rd::IBinding_Table *table = factory->create_binding_table(signature);
      defer(table->release());
      ctx->image_barrier(tmp_image, rd::Image_Access::UAV);
      ctx->image_barrier(output_image, rd::Image_Access::SAMPLED);
      ito(desc.levels) table->bind_UAV_texture(
          0, 3, i, tmp_image, rd::Image_Subresource::top_level(i), rd::Format::NATIVE);
      table->bind_texture(0, 0, 0, output_image, rd::Image_Subresource::top_level(0),
                          rd::Format::NATIVE);
      table->bind_UAV_buffer(0, 1, cnt_buffer, 0, 0);

      ctx->bind_table(table);
      ctx->bind_compute(reset_cnt_pso);
      ctx->dispatch(1, 1, 1);
      ctx->buffer_barrier(cnt_buffer, rd::Buffer_Access::UAV);
      ctx->bind_compute(mip_pso);
      PushConstants pc{};
      pc.levels            = desc.levels;
      pc.op                = 0;
      u32 pixels_per_group = 64;
      pc.last_width        = ((desc.width) + pixels_per_group - 1) / pixels_per_group;
      pc.last_height       = ((desc.height) + pixels_per_group - 1) / pixels_per_group;
      pc.last_mip          = 6;
      pc.gamma_correct     = rd::is_srgb(desc.format);
      //{
      //  u32 width  = desc.width;
      //  u32 height = desc.height;
      //  while (true) {
      //    if (width == pc.last_width && height == pc.last_height) break;
      //    width  = MAX(1, width / 2);
      //    height = MAX(1, height / 2);
      //    pc.last_mip++;
      //  }
      //  // pc.last_width  = MAX(1, width / 2);
      //  // pc.last_height = MAX(1, height / 2);
      //  // pc.last_mip++;
      //}
      pc.total_cnt = pc.last_height * pc.last_width;
      table->push_constants(&pc, 0, sizeof(pc));
      ctx->dispatch(pc.last_width, pc.last_height, 1);
      // ctx->dispatch(1 << 7, 1 << 7, 1);
      // timestamp.end_range(ctx);
    }
    Resource_ID e = factory->end_compute_pass(ctx);

    // timestamp.commit(e);
    // factory->wait_idle();
    // timestamp.update(factory);
    // fprintf(stdout, "%f ms\n", timestamp.total_duration / timestamp.total_samples);
    return tmp_image;
  }
};
#endif
#if 1
struct Raw_Mesh_3p16i_Wrapper {
  Resource_ID vertex_buffer;
  Resource_ID index_buffer;
  u32         num_indices;
  u32         num_vertices;

  void release(rd::IDevice *dev) {
    dev->release_resource(vertex_buffer);
    dev->release_resource(index_buffer);
    MEMZERO(*this);
  }
  void init(rd::IDevice *dev, float3 const *positions, u16 const *indices, u32 num_indices,
            u32 num_vertices) {
    this->num_indices  = num_indices;
    this->num_vertices = num_vertices;
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size        = u32(sizeof(float3) * num_vertices);
      vertex_buffer        = dev->create_buffer(buf_info);
      init_buffer(dev, vertex_buffer, &positions[0], sizeof(float3) * num_vertices);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size        = u32(sizeof(u16) * num_indices);
      index_buffer         = dev->create_buffer(buf_info);
      init_buffer(dev, index_buffer, &indices[0], sizeof(u16) * num_indices);
    }
  }
  void init(rd::IDevice *dev, Raw_Mesh_3p16i const &model) {
    num_indices  = model.indices.size * 3;
    num_vertices = model.positions.size;
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size        = u32(sizeof(float3) * model.positions.size);
      vertex_buffer        = dev->create_buffer(buf_info);
      init_buffer(dev, vertex_buffer, &model.positions[0], sizeof(float3) * model.positions.size);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::GPU_LOCAL;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size        = u32(sizeof(u16_face) * model.indices.size);
      index_buffer         = dev->create_buffer(buf_info);
      init_buffer(dev, index_buffer, &model.indices[0], sizeof(u16_face) * model.indices.size);
    }
  }
  void draw(rd::ICtx *ctx, u32 instances = 1, u32 first_instance = 0) {
    ctx->bind_vertex_buffer(0, vertex_buffer, 0);
    ctx->bind_index_buffer(index_buffer, 0, rd::Index_t::UINT16);
    ctx->draw_indexed(num_indices, instances, 0, first_instance, 0);
  }
};
#endif
#if 1
class Gizmo_Layer;

class Gizmo_Layer {
  public:
  class Gizmo : public Typed {
public:
    DECLARE_TYPE(Gizmo, Typed)

    Gizmo(Gizmo_Layer *layer) : layer(layer) { layer->add_gizmo(this); }

    virtual AABB getAABB() = 0;
    virtual void release() {
      layer->remove_gizmo(this);
      delete this;
    }
    virtual void update() {}
    virtual void paint() {}

    virtual bool isSelectable() { return true; }
    virtual bool isHoverable() { return true; }

    // IO callbacks
    virtual void on_select() {}
    virtual void on_unselect() {}
    virtual void on_mouse_enter() {}
    virtual void on_mouse_hover() {}
    virtual void on_mouse_leave() {}
    virtual void on_mouse_down(int mb) {}
    virtual void on_mouse_drag() {}
    virtual void on_mouse_up(int mb) {}
    virtual void on_mouse_wheel(int z) {}

    virtual ~Gizmo() {}

    bool isScheduledForRemoval() { return to_release; }
    void releaseLater() { to_release = true; }

protected:
    bool         to_release = false;
    Gizmo_Layer *layer      = NULL;
  };
  struct Gizmo_Vertex {
    afloat4 position;
  };
  struct Gizmo_Line_Vertex {
    float3 position;
    float3 color;
  };
  static_assert(sizeof(Gizmo_Line_Vertex) == 24, "");
  struct Gizmo_Instance_Data_CPU {
    afloat4x4 transform;
    afloat3   color;
  };
  static_assert(sizeof(Gizmo_Instance_Data_CPU) == 80, "");
  struct Gizmo_Push_Constants {
    afloat4x4 viewproj;
  };

  private:
  AutoArray<Gizmo_Instance_Data_CPU, 0x1000> cylinder_draw_cmds{};
  AutoArray<Gizmo_Instance_Data_CPU, 0x1000> sphere_draw_cmds{};
  AutoArray<Gizmo_Instance_Data_CPU, 0x1000> cone_draw_cmds{};
  AutoArray<Gizmo_Line_Vertex, 1000000>      line_segments{};
  Pool<char>                                 char_storage{};
  struct _String2D {
    char *   c_str;
    uint32_t len;
    float    x, y, z;
    float3   color;
  };
  AutoArray<_String2D, 0x1000> strings{};

  Raw_Mesh_3p16i_Wrapper icosahedron_wrapper = {};
  Raw_Mesh_3p16i_Wrapper cylinder_wrapper    = {};
  Raw_Mesh_3p16i_Wrapper cone_wrapper        = {};
  Raw_Mesh_3p16i_Wrapper glyph_wrapper       = {};
  struct Glyph_Instance {
    float x, y, z;
    float u, v;
    float r, g, b;
  };

#  define RESOURCE_LIST                                                                            \
    RESOURCE(signature);                                                                           \
    RESOURCE(gizmo_pso);                                                                           \
    RESOURCE(gizmo_lines_pso);                                                                     \
    RESOURCE(gizmo_font_pso);                                                                      \
    RESOURCE(font_sampler);                                                                        \
    RESOURCE(font_texture);

#  define RESOURCE(name) Resource_ID name{};
  RESOURCE_LIST
#  undef RESOURCE

  void release_resources() {
#  define RESOURCE(name) dev->release_resource(name);
    RESOURCE_LIST
#  undef RESOURCE
  }
#  undef RESOURCE_LIST

  rd::IDevice *dev = NULL;
  Camera       g_camera;
  bool         update_mouse_ray = true;
  Ray          mouse_ray{};
  float2       mouse_cursor{};
  float2       resolution{};
  enum Mode {
    NONE = 0,
    CAMERA_DRAG,
    GIZMO_DRAG,
  };
  Mode               mode = NONE;
  AutoArray<Gizmo *> gizmos{};
  AutoArray<Gizmo *> selected_gizmos{};
  Gizmo *            hovered_gizmo = NULL;
  bool               mb[3]         = {};
  bool               last_mb[3]    = {};
  int2               mpos          = {};
  int2               last_mpos     = {};
  Timer              timer{};
  bool               keys[0x100]      = {};
  bool               last_keys[0x100] = {};
  struct PushConstants {
    float4x4 viewproj;
  };
  void init(rd::IDevice *dev, Resource_ID pass) {
    this->dev = dev;
    timer.init();
    g_camera.init();
    char_storage = Pool<char>::create(1 * (1 << 20));
    {
      auto mesh = subdivide_cone(8, 1.0f, 1.0f);
      cone_wrapper.init(dev, mesh);
      mesh.release();
    }
    {
      auto mesh = subdivide_icosahedron(2);
      icosahedron_wrapper.init(dev, mesh);
      mesh.release();
    }
    {
      auto mesh = subdivide_cylinder(8, 1.0f, 1.0f);
      cylinder_wrapper.init(dev, mesh);
      mesh.release();
    }
    {
      float pos[] = {
          0.0f, 0.0f, 0.0f, //
          1.0f, 0.0f, 0.0f, //
          1.0f, 1.0f, 0.0f, //
          0.0f, 0.0f, 0.0f, //
          1.0f, 1.0f, 0.0f, //
          0.0f, 1.0f, 0.0f, //
      };
      u16 indices[] = {0, 1, 2, 3, 4, 5, 6};
      glyph_wrapper.init(dev, (float3 *)pos, indices, 6, 6);
    }
    signature = [dev] {
      rd::Binding_Space_Create_Info set_info{};
      set_info.bindings.push({rd::Binding_t::TEXTURE, 1});
      set_info.bindings.push({rd::Binding_t::SAMPLER, 1});
      rd::Binding_Table_Create_Info table_info{};
      table_info.spaces.push(set_info);
      table_info.push_constants_size = sizeof(PushConstants);
      return dev->create_signature(table_info);
    }();
    static string_ref            shader    = stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos            : SV_POSITION;
  [[vk::location(1)]] float4 pixel_color    : TEXCOORD0;
};

#ifdef VERTEX

struct VSInput {
  [[vk::location(0)]] float3 in_position    : POSITION;
  [[vk::location(1)]] float4 in_model_0     : TEXCOORD0;
  [[vk::location(2)]] float4 in_model_1     : TEXCOORD1;
  [[vk::location(3)]] float4 in_model_2     : TEXCOORD2;
  [[vk::location(4)]] float4 in_model_3     : TEXCOORD3;
  [[vk::location(5)]] float4 in_color       : TEXCOORD4;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.pixel_color = input.in_color;
  output.pos =
      mul(pc.viewproj,
        mul(
          float4(input.in_position, 1.0f),
          float4x4(
            input.in_model_0,
            input.in_model_1,
            input.in_model_2,
            input.in_model_3
          )
        )
      );
  return output;
}
#endif
#ifdef PIXEL

float4 main(in PSInput input) : SV_TARGET0 {
  return
    float4(input.pixel_color.xyz, 1.0f);
}
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    static string_ref shader_lines = stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos            : SV_POSITION;
  [[vk::location(1)]] float3 pixel_color    : TEXCOORD0;
};

#ifdef VERTEX

struct VSInput {
  [[vk::location(0)]] float3 in_position    : POSITION;
  [[vk::location(1)]] float3 in_color       : TEXCOORD0;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.pixel_color = input.in_color;
  output.pos =
      mul(
        pc.viewproj,
        float4(input.in_position, 1.0f)
      );
  return output;
}
#endif
#ifdef PIXEL

float4 main(in PSInput input) : SV_TARGET0 {
  return
    float4(input.pixel_color.xyz, 1.0f);
}
#endif
)");
    gizmo_pso                      = [&] {
      Resource_ID gizmo_vs = dev->create_shader(rd::Stage_t::VERTEX, shader, &defines[0], 1);
      Resource_ID gizmo_ps = dev->create_shader(rd::Stage_t::PIXEL, shader, &defines[1], 1);
      dev->release_resource(gizmo_vs);
      dev->release_resource(gizmo_ps);
      rd::Graphics_Pipeline_State gfx_state{};
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
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
      gfx_state.VS_set_shader(gizmo_vs);
      gfx_state.PS_set_shader(gizmo_ps);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      gfx_state.IA_set_attribute(info);
      ito(5) {
        MEMZERO(info);
        info.binding  = 1;
        info.format   = rd::Format::RGBA32_FLOAT;
        info.location = 1 + i;
        info.offset   = 16 * i;
        info.type     = (rd::Attriute_t)((u32)rd::Attriute_t::TEXCOORD0 + i);
        gfx_state.IA_set_attribute(info);
      }
      gfx_state.IA_set_vertex_binding(0, sizeof(float3), rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(1, sizeof(Gizmo_Instance_Data_CPU), rd::Input_Rate::INSTANCE);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();

    gizmo_lines_pso = [&] {
      Resource_ID gizmo_lines_vs =
          dev->create_shader(rd::Stage_t::VERTEX, shader_lines, &defines[0], 1);
      Resource_ID gizmo_lines_ps =
          dev->create_shader(rd::Stage_t::PIXEL, shader_lines, &defines[1], 1);
      dev->release_resource(gizmo_lines_vs);
      dev->release_resource(gizmo_lines_ps);
      rd::Graphics_Pipeline_State gfx_state{};
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
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
      gfx_state.VS_set_shader(gizmo_lines_vs);
      gfx_state.PS_set_shader(gizmo_lines_ps);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      gfx_state.IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 1;
      info.offset   = 12;
      info.type     = rd::Attriute_t::TEXCOORD0;
      gfx_state.IA_set_attribute(info);
      gfx_state.IA_set_vertex_binding(0, sizeof(Gizmo_Line_Vertex), rd::Input_Rate::VERTEX);
      gfx_state.IA_set_topology(rd::Primitive::LINE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();
    {
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.width  = simplefont_bitmap_width;
      info.height = simplefont_bitmap_height;
      info.depth  = 1;
      info.layers = 1;
      info.levels = 1;
      info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      info.format  = rd::Format::R8_UNORM;
      font_texture = dev->create_image(info);

      Resource_ID staging_buf{};
      defer(dev->release_resource(staging_buf));
      // RenderDoc_CTX::get().start();
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
        buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
        u32 pitch            = rd::IDevice::align_up(simplefont_bitmap_width,
                                          rd::IDevice::TEXTURE_DATA_PITCH_ALIGNMENT);
        buf_info.size        = pitch * simplefont_bitmap_height;
        staging_buf          = dev->create_buffer(buf_info);
        u8 *ptr              = (u8 *)dev->map_buffer(staging_buf);
        yto(simplefont_bitmap_height) {
          xto(simplefont_bitmap_width) {
            ptr[pitch * y + x] = simplefont_bitmap[y][x] == ' ' ? 0 : 0xffu;
          }
        }
        dev->unmap_buffer(staging_buf);
        auto ctx = dev->start_compute_pass();
        ctx->image_barrier(font_texture, rd::Image_Access::TRANSFER_DST);
        ctx->copy_buffer_to_image(staging_buf, 0, font_texture, rd::Image_Copy::top_level(pitch));
        dev->end_compute_pass(ctx);
      }
      // RenderDoc_CTX::get().end();
    }
    static const char *font_shader = R"(
struct PushConstants
{
  float2 glyph_uv_size;
  float2 glyph_size; 
  float2 viewport_size;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : DX12_PUSH_CONSTANTS_REGISTER;

struct PSInput {
  [[vk::location(0)]] float4 pos   : SV_POSITION;
  [[vk::location(1)]] float2 uv    : TEXCOORD0;
  [[vk::location(2)]] float3 color : TEXCOORD1;
};

#ifdef VERTEX

struct VSInput {
  [[vk::location(0)]] float3 vertex_position     : POSITION;
  [[vk::location(1)]] float3 instance_offset     : TEXCOORD0;
  [[vk::location(2)]] float2 instance_uv_offset  : TEXCOORD1;
  [[vk::location(3)]] float3 instance_color      : TEXCOORD2;
};

PSInput main(in VSInput input) {
  PSInput output;
  output.color = input.instance_color;
  output.uv = input.instance_uv_offset + (input.vertex_position.xy * float2(1.0, 1.0) + float2(0.0, 0.0)) * pc.glyph_uv_size;
  float4 sspos =  float4(input.vertex_position.xy * pc.glyph_size + input.instance_offset.xy, 0.0, 1.0);
  int pixel_x = int(pc.viewport_size.x * (sspos.x * 0.5 + 0.5));
  int pixel_y = int(pc.viewport_size.y * (sspos.y * 0.5 + 0.5));
  sspos.x = 2.0 * (float(pixel_x)) / pc.viewport_size.x - 1.0;
  sspos.y = 2.0 * (float(pixel_y)) / pc.viewport_size.y - 1.0;
  sspos.z = input.instance_offset.z;
  output.pos = sspos;
  return output;
}
#endif
#ifdef PIXEL
[[vk::binding(0, 0)]] Texture2D<float4>   font_texture          : register(t0, space0);
[[vk::binding(1, 0)]] SamplerState        my_sampler            : register(s1, space0);

float4 main(in PSInput input) : SV_TARGET0 {
  if (font_texture.Sample(my_sampler, input.uv).x < 0.5f)
    discard;
  return
    float4(input.color.xyz, 1.0f);
}
#endif
)";
    gizmo_font_pso                 = [&] {
      Resource_ID font_vs =
          dev->create_shader(rd::Stage_t::VERTEX, stref_s(font_shader), &defines[0], 1);
      Resource_ID font_ps =
          dev->create_shader(rd::Stage_t::PIXEL, stref_s(font_shader), &defines[1], 1);
      dev->release_resource(font_vs);
      dev->release_resource(font_ps);
      rd::Graphics_Pipeline_State gfx_state{};
      setup_default_state(gfx_state);
      rd::DS_State ds_state{};
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
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
      gfx_state.VS_set_shader(font_vs);
      gfx_state.PS_set_shader(font_ps);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      gfx_state.IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 1;
      info.offset   = 0;
      info.type     = rd::Attriute_t::TEXCOORD0;
      gfx_state.IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RG32_FLOAT;
      info.location = 2;
      info.offset   = 12;
      info.type     = rd::Attriute_t::TEXCOORD1;
      gfx_state.IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 3;
      info.offset   = 20;
      info.type     = rd::Attriute_t::TEXCOORD2;
      gfx_state.IA_set_attribute(info);
      gfx_state.IA_set_vertex_binding(0, sizeof(float3), rd::Input_Rate::VERTEX);
      gfx_state.IA_set_vertex_binding(1, sizeof(Glyph_Instance), rd::Input_Rate::INSTANCE);
      gfx_state.IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      return dev->create_graphics_pso(signature, pass, gfx_state);
    }();

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
    font_sampler        = dev->create_sampler(info);
  }
  Gizmo *pick(Ray const &ray) {
    float  t       = 1.0e6;
    Gizmo *closest = NULL;
    ito(gizmos.size) {
      float curt;
      if (gizmos[i] && gizmos[i]->isHoverable() &&
          gizmos[i]->getAABB().collide(ray.o, ray.d, curt, t)) {
        if (curt < t) {
          closest = gizmos[i];
          t       = curt;
        }
      }
    }
    return closest;
  }

  public:
  static Gizmo_Layer *create(rd::IDevice *dev, Resource_ID pass) {
    Gizmo_Layer *gl = new Gizmo_Layer;
    gl->init(dev, pass);
    return gl;
  }
  void remove_gizmo(Gizmo *g) {
    if (selected_gizmos.contains(g)) {
      selected_gizmos.replace(g, NULL);
      g->on_unselect();
    }
    ito(gizmos.size) {
      if (gizmos[i] == g) gizmos[i] = NULL;
    }
    if (hovered_gizmo == g) {
      hovered_gizmo = g;
      g->on_mouse_leave();
    }
  }
  void add_gizmo(Gizmo *g) {
    ito(gizmos.size) {
      if (gizmos[i] == NULL) {
        gizmos[i] = g;
        return;
      }
    }
    gizmos.push(g);
  }
  void release() {
    ito(gizmos.size) {
      if (gizmos[i]) gizmos[i]->release();
    }
    g_camera.release();
    cylinder_draw_cmds.release();
    char_storage.release();
    strings.release();
    cone_draw_cmds.release();
    sphere_draw_cmds.release();
    line_segments.release();
    cylinder_wrapper.release(dev);
    icosahedron_wrapper.release(dev);
    cylinder_wrapper.release(dev);
    release_resources();
    delete this;
  }
#  undef RESOURCE_LIST
  Camera &get_camera() { return g_camera; }
  void    draw_cylinder(float3 start, float3 end, float radius, float3 color) {
    float3 dr      = end - start;
    float  length  = glm::length(dr);
    float3 dir     = glm::normalize(dr);
    float3 tangent = glm::cross(dir, float3{0.0f, 1.0f, 0.0f});
    if (length2(tangent) < 1.0e-3f) tangent = glm::cross(dir, float3{0.0f, 0.0f, 1.0f});
    tangent           = glm::normalize(tangent);
    float3   binormal = -glm::cross(dir, tangent);
    float4x4 tranform =
        // clang-format off
        float4x4(tangent.x,  tangent.y,  tangent.z,  0.0f,
                 binormal.x, binormal.y, binormal.z, 0.0f,
                 dir.x,      dir.y,      dir.z,      0.0f,
                 start.x,    start.y,    start.z,    1.0f);
    // clang-format on
    Gizmo_Instance_Data_CPU cmd;
    MEMZERO(cmd);
    cmd.color     = color;
    cmd.transform = tranform * glm::scale(float4x4(1.0f), float3(radius, radius, length));
    cylinder_draw_cmds.push(cmd);
  }
  void draw_string(float3 position, float3 color, char const *fmt, ...) {
    char    buf[0x100];
    va_list args;
    va_start(args, fmt);
    i32 len = vsprintf(buf, fmt, args);
    va_end(args);
    draw_string(stref_s(buf), position, color);
  }
  void draw_string(string_ref str, float3 position, float3 color) {
    if (str.len == 0) return;
    char *dst = char_storage.alloc(str.len + 1);
    memcpy(dst, str.ptr, str.len);
    dst[str.len] = '\0';
    _String2D internal_string;
    internal_string.color = color;
    internal_string.c_str = dst;
    internal_string.len   = (uint32_t)str.len;
    internal_string.x     = position.x;
    internal_string.y     = position.y;
    internal_string.z     = position.z;
    strings.push(internal_string);
  }
  void draw_ss_circle(float3 o, float radius, float3 color) {
    int    N         = 16;
    float3 last_pos  = o + g_camera.right * radius;
    float  delta_phi = 2.0f * PI / N;
    for (int i = 1; i <= N; i++) {
      float  s       = sinf(delta_phi * i);
      float  c       = cosf(delta_phi * i);
      float3 new_pos = o + (s * g_camera.up + c * g_camera.right) * radius;
      draw_line(last_pos, new_pos, color);
      last_pos = new_pos;
    }
  }
  void draw_sphere(float3 start, float radius, float3 color) {

    Gizmo_Instance_Data_CPU cmd;
    MEMZERO(cmd);
    cmd.color = color;
    float4x4 tranform =
        // clang-format off
        float4x4(radius,     0.0f,    0.0f,    0.0f,
                 0.0f,       radius,  0.0f,    0.0f,
                 0.0f,       0.0f,    radius,  0.0,
                 start.x,    start.y, start.z, 1.0f);
    cmd.transform = tranform;
    sphere_draw_cmds.push(cmd);
  }
  void draw_cone(float3 start, float3 dir, float radius, float3 color) {
    float3 normal = normalize(dir);
    float3 up =
        fabs(normal.z) > 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 1.0f);
    float3   tangent  = normalize(cross(normal, up));
    float3   binormal = -cross(normal, tangent);
    float4x4 tranform = float4x4(
        // clang-format off
      tangent.x,  tangent.y,  tangent.z,  0.0f,
      binormal.x, binormal.y, binormal.z, 0.0f,
      dir.x,      dir.y,      dir.z,      0.0f,
      start.x,    start.y,    start.z,    1.0f);
    // clang-format on
    Gizmo_Instance_Data_CPU cmd;
    MEMZERO(cmd);
    cmd.color     = color;
    cmd.transform = tranform * glm::scale(float4x4(1.0f), float3(radius, radius, 1.0f));
    cone_draw_cmds.push(cmd);
  }
  void draw_line(float3 p0, float3 p1, float3 color) {
    line_segments.push({p0, color});
    line_segments.push({p1, color});
  }

  void on_mouse_down(int mb) {
    if (mb == 0) {
      if (!hovered_gizmo) {
        clearSelection();
        mode = CAMERA_DRAG;
      } else {
        if (hovered_gizmo->isSelectable()) {
          if (!selected_gizmos.contains(hovered_gizmo)) {
            clearSelection();
            hovered_gizmo->on_select();
            selected_gizmos.push(hovered_gizmo);
          }
          ito(selected_gizmos.size) if (selected_gizmos[i]) selected_gizmos[i]->on_mouse_down(mb);
          mode = GIZMO_DRAG;
        }
      }
    }
  }
  void on_mouse_up(int mb) {
    if (mb == 0) mode = NONE;
  }
  void on_mouse_wheel(int z) {}
  Ray  getMouseRay() { return mouse_ray; }
  void on_mouse_move() {
    float2 uv = float2(mpos.x, mpos.y);
    uv /= resolution;
    uv           = 2.0f * uv - float2(1.0f, 1.0f);
    uv.y         = -uv.y;
    mouse_cursor = uv;
    if (update_mouse_ray) mouse_ray = g_camera.gen_ray(uv);
    Gizmo *new_hover = pick(mouse_ray);
    if (hovered_gizmo && new_hover != hovered_gizmo) {
      hovered_gizmo->on_mouse_leave();
    }
    if (new_hover && new_hover != hovered_gizmo) {
      new_hover->on_mouse_enter();
    }
    hovered_gizmo = new_hover;
    if (hovered_gizmo) {
      hovered_gizmo->on_mouse_hover();
    }
    if (mode == GIZMO_DRAG) {
      ito(selected_gizmos.size) {
        if (selected_gizmos[i]) selected_gizmos[i]->on_mouse_drag();
      }
    } else if (mode == CAMERA_DRAG) {
      i32 dx = mpos.x - last_mpos.x;
      i32 dy = mpos.y - last_mpos.y;
      g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
      g_camera.theta -= (float)(dy)*5.0e-3f;
    }
  }
  void clearSelection() {
    ito(selected_gizmos.size) {
      if (selected_gizmos[i]) selected_gizmos[i]->on_unselect();
    }
    selected_gizmos.release();
  }
  void setFocus(Gizmo *g) {
    clearSelection();
    clearHover();
    selected_gizmos.push(g);
    hovered_gizmo = g;
    g->on_mouse_enter();
    g->on_select();
  }
  // Called multiple times per frame per imgui window
  void per_imgui_window() {

    ImGuiIO &io = ImGui::GetIO();
    if (ImGui::IsWindowHovered()) {
      ImVec2 imguires  = get_window_size();
      this->resolution = float2(imguires.x, imguires.y);
      auto scroll_y    = ImGui::GetIO().MouseWheel;
      if (scroll_y) {
        g_camera.distance += g_camera.distance * 2.e-1 * scroll_y;
        g_camera.distance = clamp(g_camera.distance, 1.0e-3f, 1000.0f);
      }
      f32 camera_speed = 2.0f * g_camera.distance;
      if (ImGui::GetIO().KeysDown[SDL_SCANCODE_LSHIFT]) {
        camera_speed = 10.0f * g_camera.distance;
      }
      float3 camera_diff = float3(0.0f, 0.0f, 0.0f);
      if (ImGui::GetIO().KeysDown[SDL_SCANCODE_W]) {
        camera_diff += g_camera.look;
      }
      if (ImGui::GetIO().KeysDown[SDL_SCANCODE_S]) {
        camera_diff -= g_camera.look;
      }
      if (ImGui::GetIO().KeysDown[SDL_SCANCODE_A]) {
        camera_diff -= g_camera.right;
      }
      if (ImGui::GetIO().KeysDown[SDL_SCANCODE_D]) {
        camera_diff += g_camera.right;
      }
      if (dot(camera_diff, camera_diff) > 1.0e-3f) {
        g_camera.look_at += glm::normalize(camera_diff) * camera_speed * (float)timer.dt;
      }
      ImVec2 imguimpos = ImGui::GetMousePos();
      auto   wpos      = ImGui::GetCursorScreenPos();
      auto   wsize     = ImGui::GetWindowSize();
      g_camera.aspect  = float(wsize.x) / wsize.y;
      timer.update();
      g_camera.update();
      imguimpos.x -= wpos.x;
      imguimpos.y -= wpos.y;
      ito(3) {
        last_mb[i] = mb[i];
        mb[i]      = io.MouseDown[i];
        if (mb[i] && !last_mb[i]) on_mouse_down(i);
        if (last_mb[i] && !mb[i]) on_mouse_up(i);
      }
      ito(0x100) {
        last_keys[i] = keys[i];
        keys[i]      = ImGui::GetIO().KeysDown[i];
      }
      if (keys[SDL_SCANCODE_SPACE] && !last_keys[SDL_SCANCODE_SPACE])
        update_mouse_ray = !update_mouse_ray;
      last_mpos = mpos;
      mpos      = int2(imguimpos.x, imguimpos.y);
      if (mpos != last_mpos) {
        on_mouse_move();
      }
    }
  }
  void clearHover() {
    if (hovered_gizmo) {
      hovered_gizmo->on_mouse_leave();
      hovered_gizmo = NULL;
    }
  }
  float2 getMouse() const { return mouse_cursor; }
  Ray    getMouseRay() const { return mouse_ray; }
  void   reserveLines(size_t cnt) { line_segments.reserve(cnt); }
  void   render_linebox(float3 min, float3 max, float3 color) {
    float coordsx[6] = {
        min.x,
        max.x,
    };
    float coordsy[6] = {
        min.y,
        max.y,
    };
    float coordsz[6] = {
        min.z,
        max.z,
    };
    ito(8) {
      int x = (i >> 0) & 1;
      int y = (i >> 1) & 1;
      int z = (i >> 2) & 1;
      if (x == 0) {
        draw_line(float3(coordsx[0], coordsy[y], coordsz[z]),
                  float3(coordsx[1], coordsy[y], coordsz[z]), color);
      }
      if (y == 0) {
        draw_line(float3(coordsx[x], coordsy[0], coordsz[z]),
                  float3(coordsx[x], coordsy[1], coordsz[z]), color);
      }
      if (z == 0) {
        draw_line(float3(coordsx[x], coordsy[y], coordsz[0]),
                  float3(coordsx[x], coordsy[y], coordsz[1]), color);
      }
    }
  }
  void reset() {
    cylinder_draw_cmds.reset();
    cone_draw_cmds.reset();
    sphere_draw_cmds.reset();
    line_segments.reset();
  }
  void render(rd::ICtx *ctx, u32 width, u32 height) {
    rd::IBinding_Table *table = dev->create_binding_table(signature);
    defer(table->release());
    table->bind_sampler(0, 1, font_sampler);
    table->bind_texture(0, 0, 0, font_texture, rd::Image_Subresource::top_level(),
                        rd::Format::NATIVE);
    ctx->bind_table(table);
    PushConstants pc{};
    pc.viewproj = g_camera.viewproj();
    defer(char_storage.reset());
    float4x4 viewproj = g_camera.viewproj();
    if (hovered_gizmo) {
      auto   aabb  = hovered_gizmo->getAABB();
      float3 color = float3(1.0f, 1.0f, 1.0f);
      render_linebox(aabb.min, aabb.max, color);
    }
    ito(gizmos.size) if (gizmos[i]) { gizmos[i]->update(); }
    ito(gizmos.size) if (gizmos[i] && gizmos[i]->isScheduledForRemoval()) { gizmos[i]->release(); }
    ito(gizmos.size) if (gizmos[i]) { gizmos[i]->paint(); }
    if (strings.size == 0 && cylinder_draw_cmds.size == 0 && sphere_draw_cmds.size == 0 &&
        cone_draw_cmds.size == 0 && line_segments.size == 0)
      return;
    if (cylinder_draw_cmds.size != 0 || sphere_draw_cmds.size != 0 || cone_draw_cmds.size != 0) {
      u32                    cylinder_offset = 0;
      u32                    num_cylinders   = cylinder_draw_cmds.size;
      u32                    sphere_offset   = num_cylinders;
      u32                    num_spheres     = sphere_draw_cmds.size;
      u32                    cone_offset     = num_cylinders + num_spheres;
      u32                    num_cones       = cone_draw_cmds.size;
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
      buf_info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size = (cylinder_draw_cmds.size + sphere_draw_cmds.size + cone_draw_cmds.size) *
                      sizeof(Gizmo_Instance_Data_CPU);
      Resource_ID gizmo_instance_buffer = dev->create_buffer(buf_info);
      dev->release_resource(gizmo_instance_buffer);
      void *ptr = dev->map_buffer(gizmo_instance_buffer);
      if (cylinder_draw_cmds.size > 0)
        memcpy((u8 *)ptr + cylinder_offset * sizeof(Gizmo_Instance_Data_CPU),
               &cylinder_draw_cmds[0], num_cylinders * sizeof(Gizmo_Instance_Data_CPU));
      if (sphere_draw_cmds.size > 0)
        memcpy((u8 *)ptr + sphere_offset * sizeof(Gizmo_Instance_Data_CPU), &sphere_draw_cmds[0],
               num_spheres * sizeof(Gizmo_Instance_Data_CPU));
      if (cone_draw_cmds.size > 0)
        memcpy((u8 *)ptr + cone_offset * sizeof(Gizmo_Instance_Data_CPU), &cone_draw_cmds[0],
               num_cones * sizeof(Gizmo_Instance_Data_CPU));

      num_cylinders = cylinder_draw_cmds.size;
      dev->unmap_buffer(gizmo_instance_buffer);
      cylinder_draw_cmds.reset();
      cone_draw_cmds.reset();
      sphere_draw_cmds.reset();
      table->push_constants(&pc, 0, sizeof(pc));
      ctx->bind_graphics_pso(gizmo_pso);
      ctx->bind_vertex_buffer(1, gizmo_instance_buffer, 0);
      cylinder_wrapper.draw(ctx, num_cylinders, cylinder_offset);
      icosahedron_wrapper.draw(ctx, num_spheres, sphere_offset);
      cone_wrapper.draw(ctx, num_cones, cone_offset);
    }
    if (strings.size != 0) {
      defer(strings.reset());
      float    glyph_uv_width  = (float)simplefont_bitmap_glyphs_width / simplefont_bitmap_width;
      float    glyph_uv_height = (float)simplefont_bitmap_glyphs_height / simplefont_bitmap_height;
      float4x4 viewproj        = g_camera.viewproj();
      float    glyph_pad_ss    = 2.0f / width;
      uint32_t max_num_glyphs  = 0;
      uint32_t num_strings     = strings.size;
      kto(num_strings) { max_num_glyphs += (uint32_t)strings[k].len; }
      TMP_STORAGE_SCOPE;
      Glyph_Instance *glyphs =
          (Glyph_Instance *)tl_alloc_tmp(sizeof(Glyph_Instance) * max_num_glyphs);
      uint32_t num_glyphs          = 0;
      f32      glyphs_screen_width = 2.0f * 1.0f * (float)(simplefont_bitmap_glyphs_width) / width;
      f32 glyphs_screen_height = 2.0f * 1.0f * (float)(simplefont_bitmap_glyphs_height) / height;
      kto(num_strings) {
        _String2D string = strings[k];
        if (string.len == 0) continue;

        float3 p = {string.x, string.y, string.z};
        // float  x0 = glm::dot(viewproj[0], float4{p.x, p.y, p.z, 1.0f});
        // float  x1 = glm::dot(viewproj[1], float4{p.x, p.y, p.z, 1.0f});
        // float    x2       = glm::dot(viewproj[2], float4{p.x, p.y, p.z, 1.0f});
        // float  x3       = glm::dot(viewproj[3], float4{p.x, p.y, p.z, 1.0f});
        // float2 ss       = float2{x0, x1} / x3;
        float4 pp       = viewproj * float4{p.x, p.y, p.z, 1.0f};
        float  z        = pp.z / pp.w;
        float2 ss       = pp.xy / pp.ww;
        float  min_ss_x = ss.x;
        float  min_ss_y = ss.y;
        float  max_ss_x = ss.x + (glyphs_screen_width + glyph_pad_ss) * string.len;
        float  max_ss_y = ss.y + glyphs_screen_height;
        if (z < 0.0f || z > 1.0f || min_ss_x > 1.0f || min_ss_y > 1.0f || max_ss_x < -1.0f ||
            max_ss_y < -1.0f)
          continue;

        ito(string.len) {
          uint32_t c = (uint32_t)string.c_str[i];

          // Printable characters only
          c            = clamp(c, 0x20u, 0x7eu);
          uint32_t row = (c - 0x20) / simplefont_bitmap_glyphs_per_row;
          uint32_t col = (c - 0x20) % simplefont_bitmap_glyphs_per_row;
          float    v0 =
              ((float)row * (simplefont_bitmap_glyphs_height + simplefont_bitmap_glyphs_pad_y * 2) +
               simplefont_bitmap_glyphs_pad_y) /
              simplefont_bitmap_height;
          float u0 =
              ((float)col * (simplefont_bitmap_glyphs_width + simplefont_bitmap_glyphs_pad_x * 2) +
               simplefont_bitmap_glyphs_pad_x) /
              simplefont_bitmap_width;
          Glyph_Instance glyph;
          glyph.u              = u0;
          glyph.v              = v0;
          glyph.x              = ss.x + (glyphs_screen_width + glyph_pad_ss) * i;
          glyph.y              = ss.y;
          glyph.z              = z;
          glyph.r              = string.color.r;
          glyph.g              = string.color.g;
          glyph.b              = string.color.b;
          glyphs[num_glyphs++] = glyph;
        }
      }
      if (num_glyphs != 0) {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.memory_type              = rd::Memory_Type::CPU_WRITE_GPU_READ;
        buf_info.usage_bits               = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
        buf_info.size                     = num_glyphs * sizeof(Glyph_Instance);
        Resource_ID gizmo_instance_buffer = dev->create_buffer(buf_info);
        dev->release_resource(gizmo_instance_buffer);
        Glyph_Instance *ptr = (Glyph_Instance *)dev->map_buffer(gizmo_instance_buffer);
        memcpy(ptr, glyphs, buf_info.size);
        dev->unmap_buffer(gizmo_instance_buffer);
        ctx->bind_graphics_pso(gizmo_font_pso);
        ctx->bind_vertex_buffer(1, gizmo_instance_buffer, 0);
        struct PC {
          float2 glyph_uv_size;
          float2 glyph_size;
          float2 viewport_size;
        } pc;
        pc.glyph_uv_size = {glyph_uv_width, glyph_uv_height};
        pc.glyph_size    = {glyphs_screen_width, -glyphs_screen_height};
        pc.viewport_size = {(f32)width, (f32)height};
        table->push_constants(&pc, 0, sizeof(pc));
        glyph_wrapper.draw(ctx, num_glyphs, 0);
      }
    }
    if (line_segments.size != 0) {

      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.memory_type              = rd::Memory_Type::CPU_WRITE_GPU_READ;
      buf_info.usage_bits               = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size                     = (line_segments.size) * sizeof(Gizmo_Line_Vertex);
      Resource_ID gizmo_instance_buffer = dev->create_buffer(buf_info);
      void *      ptr                   = dev->map_buffer(gizmo_instance_buffer);
      memcpy((u8 *)ptr, &line_segments[0], line_segments.size * sizeof(Gizmo_Line_Vertex));
      dev->unmap_buffer(gizmo_instance_buffer);
      dev->release_resource(gizmo_instance_buffer);
      ctx->bind_graphics_pso(gizmo_lines_pso);
      ctx->bind_vertex_buffer(0, gizmo_instance_buffer, 0);
      table->push_constants(&pc, 0, sizeof(pc));
      ctx->draw(line_segments.size, 1, 0, 0);
      line_segments.reset();
    }
  }
};

template <typename T> class GPUBuffer {
  private:
  rd::IDevice *factory;
  Array<T>     cpu_array;
  Resource_ID  gpu_buffer;
  Resource_ID  cpu_buffer;
  size_t       gpu_buffer_size;

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
      if (cpu_buffer.is_valid()) factory->release_resource(cpu_buffer);
      if (gpu_buffer.is_valid()) factory->release_resource(gpu_buffer);
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
        info.memory_type = rd::Memory_Type::CPU_WRITE_GPU_READ;
        info.usage_bits  = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
        info.size        = cpu_array.size * sizeof(T);
        cpu_array        = factory->create_buffer(info);
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
    if (cpu_buffer.is_valid()) factory->release_resource(cpu_buffer);
    if (gpu_buffer.is_valid()) factory->release_resource(gpu_buffer);
    cpu_array.release();
  }
};

class DragGizmo : public Gizmo_Layer::Gizmo {
  public:
  DECLARE_TYPE(DragGizmo, Gizmo_Layer::Gizmo)

  DragGizmo(Gizmo_Layer *layer, float3 axis, float3 *pos, float offset, float3 color)
      : Gizmo(layer), offset(offset), axis(normalize(axis)), position(pos), color(color) {
    // if (axis.z > 0.9)
    // tangent = normalize(cross(axis, float3(0.0f, 1.0f, 0.0f)));
    // else
    // tangent = normalize(cross(axis, float3(0.0f, 0.0f, 1.0f)));
    // binormal = cross(tangent, axis);
  }
  ~DragGizmo() override {}
  AABB getAABB() override { return aabb; }
  void release() override { Gizmo::release(); }
  void update() override {
    aabb.min = axis * offset + *position - float3(1.0f, 1.0f, 1.0f) * 0.4f;
    aabb.max = axis * offset + *position + float3(1.0f, 1.0f, 1.0f) * 0.4f;
  }
  void paint() override {
    // float3 color = float3(1.0f, 1.0f, 1.0f);
    // if (selected) color = float3(1.0f, 0.0f, 0.0f);
    layer->draw_cone(axis * offset + *position, axis * 0.4f, 0.2f, color);
  }
  float3 get_cpa(float3 const &ray_origin, float3 const &ray_dir) {
    float  b  = dot(ray_dir, axis);
    float3 w0 = ray_origin - *position;
    float  d  = dot(ray_dir, w0);
    float  e  = dot(axis, w0);
    float  t  = (b * e - d) / (1.0f - b * b);
    return ray_origin + ray_dir * t;
  }
  void setAxis(float3 a) { axis = normalize(a); }
  // IO callbacks
  void on_select() override { selected = true; }
  void on_unselect() override { selected = false; }
  void on_mouse_enter() override { hovered = true; }
  void on_mouse_hover() override {}
  void on_mouse_leave() override { hovered = false; }
  void on_mouse_down(int mb) override {
    float3 cpa = get_cpa(layer->getMouseRay().o, layer->getMouseRay().d);
    old_cpa    = cpa;
  }
  void on_mouse_drag() override {
    float3 cpa = get_cpa(layer->getMouseRay().o, layer->getMouseRay().d);
    *position += axis * dot(cpa - old_cpa, axis);
    old_cpa = cpa;
  }
  void on_mouse_up(int mb) override {}
  void on_mouse_wheel(int z) override {}
  bool isSelected() const { return selected; }

  protected:
  bool    hovered  = false;
  bool    selected = false;
  float   offset;
  float3  old_cpa{};
  float3  axis;
  float3  color;
  float3 *position = NULL;
  AABB    aabb;
};

class XYZDragGizmo : public Gizmo_Layer::Gizmo {
  public:
  DECLARE_TYPE(XYZDragGizmo, Gizmo_Layer::Gizmo)

  XYZDragGizmo(Gizmo_Layer *layer, float3 *pos) : Gizmo(layer), position(pos) {
    xdrag = new DragGizmo(layer, float3(1.0f, 0.0f, 0.0f), pos, 1.0f, float3(1.0f, 0.0f, 0.0f));
    ydrag = new DragGizmo(layer, float3(0.0f, 1.0f, 0.0f), pos, 1.0f, float3(0.0f, 1.0f, 0.0f));
    zdrag = new DragGizmo(layer, float3(0.0f, 0.0f, 1.0f), pos, 1.0f, float3(0.0f, 0.0f, 1.0f));
  }
  ~XYZDragGizmo() override {}

  AABB getAABB() override { return aabb; }
  void release() override { Gizmo::release(); }
  void update() override {
    aabb.min = *position;
    aabb.max = *position;
  }
  void paint() override {}
  // IO callbacks
  void on_select() override { selected = true; }
  void on_unselect() override { selected = false; }
  void on_mouse_enter() override { hovered = true; }
  void on_mouse_hover() override {}
  void on_mouse_leave() override { hovered = false; }
  void on_mouse_down(int mb) override {}
  void on_mouse_drag() override {}
  void on_mouse_up(int mb) override {}
  void on_mouse_wheel(int z) override {}

  DragGizmo *getX() { return xdrag; }
  DragGizmo *getY() { return ydrag; }
  DragGizmo *getZ() { return zdrag; }

  protected:
  bool       hovered  = false;
  bool       selected = false;
  DragGizmo *xdrag, *ydrag, *zdrag;
  float3 *   position = NULL;
  AABB       aabb;
};

class MeshGizmo : public Gizmo_Layer::Gizmo {
  public:
  DECLARE_TYPE(MeshGizmo, Gizmo_Layer::Gizmo)

  MeshGizmo(Gizmo_Layer *layer, MeshNode *mn) : Gizmo(layer), mn(mn) {}
  ~MeshGizmo() override {}

  AABB getAABB() override { return aabb; }
  void release() override { Gizmo::release(); }
  void update() override {
    aabb = mn->getAABB();
    if (gizmo) {
      gizmo->getX()->setAxis((mn->get_transform() * float4(1.0f, 0.0f, 0.0f, 0.0f)).xyz);
      gizmo->getY()->setAxis((mn->get_transform() * float4(0.0f, 1.0f, 0.0f, 0.0f)).xyz);
      gizmo->getZ()->setAxis((mn->get_transform() * float4(0.0f, 0.0f, 1.0f, 0.0f)).xyz);
    }
  }
  bool isSelectable() override { return gizmo == NULL; }
  bool isHoverable() override { return gizmo == NULL; }
  void paint() override {}
  // IO callbacks
  void on_select() override {
    selected = true;
    gizmo    = new XYZDragGizmo(layer, &mn->offset);
    layer->setFocus(gizmo);
  }
  void on_unselect() override { selected = false; }
  void on_mouse_enter() override { hovered = true; }
  void on_mouse_hover() override {}
  void on_mouse_leave() override { hovered = false; }
  void on_mouse_down(int mb) override {}
  void on_mouse_drag() override {}
  void on_mouse_up(int mb) override {}
  void on_mouse_wheel(int z) override {}

  protected:
  bool          hovered  = false;
  bool          selected = false;
  MeshNode *    mn;
  AABB          aabb;
  XYZDragGizmo *gizmo = NULL;
};

class GizmoComponent : public Node::Component {
  public:
  DECLARE_TYPE(GizmoComponent, Component)

  GizmoComponent(Gizmo_Layer *layer, MeshNode *n) : Component(n) {
    position = n->offset;
    gizmo    = new MeshGizmo(layer, n);
  }
  static GizmoComponent *create(Gizmo_Layer *layer, MeshNode *n) {
    return new GizmoComponent(layer, n);
  }
  void release() override {
    gizmo->releaseLater();
    Component::release();
  }
  ~GizmoComponent() override {}
  void update() override {}

  protected:
  float3     position;
  MeshGizmo *gizmo = NULL;
};
static float3 transform(float4x4 const &t, float3 const &v) {
  float4 r = t * float4(v.xyz, 1.0f);
  return r.xyz;
}
static void render_bvh(float4x4 const &t, bvh::Node *bvh, Gizmo_Layer *gl) {
  ASSERT_DEBUG(bvh);
  bvh->traverse([&](bvh::Node *node) {
    if (node->is_leaf()) return;
    gl->render_linebox(transform(t, node->aabb.min), transform(t, node->aabb.max),
                       float3(1.0f, 0.0f, 0.0f));
  });
}
#endif

class IPass;
class IPassMng {
  public:
  virtual IPass *getPass(char const *name) = 0;
};

struct RenderingContext {
  rd::IDevice *factory     = NULL;
  Config *     config      = NULL;
  Scene *      scene       = NULL;
  Gizmo_Layer *gizmo_layer = NULL;
  IPassMng *   pass_mng    = NULL;
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

class IPass {
  public:
  virtual char const *getName()            = 0;
  virtual u32         getNumBuffers()      = 0;
  virtual char const *getBufferName(u32 i) = 0;
  Resource_ID         getBuffer(char const *name) {
    ito(getNumBuffers()) if (strcmp(getBufferName(i), name) == 0) return getBuffer(i);
    return {};
  }
  virtual Resource_ID getBuffer(u32 i) = 0;
  virtual void        render()         = 0;
  // auto deletes itself
  virtual void   release()             = 0;
  virtual double getLastDurationInMs() = 0;
};

#endif // RENDERING_UTILS_HPP
