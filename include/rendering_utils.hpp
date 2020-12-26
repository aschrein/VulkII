#ifndef RENDERING_UTILS_HPP
#define RENDERING_UTILS_HPP

#include "rendering.hpp"
#include "scene.hpp"
#include "simplefont.h"
#include "utils.hpp"

#ifdef __linux__
#  include <SDL2/SDL.h>
#else
#  include <SDL.h>
#  include <SDL_syswm.h>
#endif

#include <imgui.h>
#include <imgui/examples/imgui_impl_sdl.h>

#endif

#include "3rdparty/renderdoc_app.h"

class RenderDoc_CTX {
  private:
  RENDERDOC_API_1_4_1 * renderdoc_api = NULL;
  bool                  dll_not_found = false;
  static RenderDoc_CTX &get() {
    static RenderDoc_CTX ctx;
    return ctx;
  }
  void init() {
    if (dll_not_found) return;
    if (renderdoc_api == NULL) {
      HMODULE mod = GetModuleHandle("renderdoc.dll");
      if (mod) {
        pRENDERDOC_GetAPI getApi = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        if (getApi) {
          if (getApi(eRENDERDOC_API_Version_1_4_1, (void **)&renderdoc_api) == 1) {

          } else {
            fprintf(stderr, "[RenderDoc] failed to retieve API functions!\n");
          }
        } else {
          fprintf(stderr, "[RenderDoc] GetAPI not found!\n");
        }
      } else {
        dll_not_found = true;
        fprintf(stderr, "[RenderDoc] module not found!\n");
      }
    }
  }
  void _start() {
    if (renderdoc_api && renderdoc_api->IsFrameCapturing()) return;
#ifdef WIN32
    if (renderdoc_api == NULL) {
      if (dll_not_found) return;
      init();
    }
#endif
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

f32 max3(float3 const &a) { return MAX3(a.x, a.y, a.z); }

static void setup_default_state(rd::Imm_Ctx *ctx, u32 num_rts = 1) {
  rd::Blend_State bs;
  MEMZERO(bs);
  bs.enabled          = false;
  bs.color_write_mask = (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
                        (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
  ito(num_rts) ctx->OM_set_blend_state(i, bs);
  ctx->IA_set_topology(rd::Primitive::TRIANGLE_LIST);
  rd::RS_State rs_state;
  MEMZERO(rs_state);
  rs_state.polygon_mode = rd::Polygon_Mode::FILL;
  rs_state.front_face   = rd::Front_Face::CW;
  rs_state.cull_mode    = rd::Cull_Mode::NONE;
  ctx->RS_set_state(rs_state);
  rd::DS_State ds_state;
  MEMZERO(ds_state);
  ds_state.cmp_op             = rd::Cmp::EQ;
  ds_state.enable_depth_test  = false;
  ds_state.enable_depth_write = false;
  ctx->DS_set_state(ds_state);
  rd::MS_State ms_state;
  MEMZERO(ms_state);
  ms_state.sample_mask = 0xffffffffu;
  ms_state.num_samples = 1;
  ctx->MS_set_state(ms_state);
}

struct ImGui_ID {
  Resource_ID     id;
  u32             base_level;
  u32             base_layer;
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
  Resource_ID vs{};
  Resource_ID ps{};
  u32         width{0}, height{0};
  Resource_ID sampler{};
  Resource_ID vertex_buffer{};
  Resource_ID index_buffer{};

  Resource_ID font_texture{};
  Resource_ID staging_buffer{};

  InlineArray<ImGui_ID, 0x100> image_bindings{};

  unsigned char *font_pixels = NULL;
  int            font_width = 0, font_height = 0;

  i32           last_m_x          = 0;
  i32           last_m_y          = 0;
  ImDrawData *  draw_data         = NULL;
  Timer         timer             = {};
  bool          imgui_initialized = false;
  rd::IFactory *factory           = NULL;
  SDL_Window *  window            = NULL;

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
  void _start(rd::Impl_t impl_t) {
    static int init = [] { return SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS); }();
    window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 512, 512,
                              SDL_WINDOW_RESIZABLE);
    defer({
      release();
      // SDL_Quit();
    });

    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);
    HWND hwnd = wmInfo.info.win.window;

    if (impl_t == rd::Impl_t::VULKAN) {
      factory = rd::create_vulkan(hwnd);
    } else if (impl_t == rd::Impl_t::DX12) {
      TRAP;
      // factory = rd::create_vulkan(NULL);
    } else {
      TRAP;
    }
    factory->start_frame();
    imgui_initialized = true;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    if (factory->getImplType() == rd::Impl_t::VULKAN) {
      ImGui_ImplSDL2_InitForVulkan(window);
    } else if (factory->getImplType() == rd::Impl_t::DX12) {
      ImGui_ImplSDL2_InitForD3D(window);
    } else {
      TRAP;
    }

    io.Fonts->GetTexDataAsRGBA32(&font_pixels, &font_width, &font_height);

    rd::Image_Create_Info info;
    MEMZERO(info);
    info.format   = rd::Format::RGBA8_UNORM;
    info.width    = font_width;
    info.height   = font_height;
    info.depth    = 1;
    info.layers   = 1;
    info.levels   = 1;
    info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
    info.usage_bits =
        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
    font_texture    = factory->create_image(info);
    io.Fonts->TexID = bind_texture(font_texture, 0, 0, rd::Format::RGBA8_UNORM);

    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
    buf_info.size       = font_width * font_height * 4;
    staging_buffer      = factory->create_buffer(buf_info);

    on_init();
    factory->end_frame();
    while (true) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT) {
          exit(0);
        }
        switch (event.type) {
        case SDL_WINDOWEVENT:
          if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
          }
          break;
        }
      }
      factory->start_frame();
      frame();
      factory->end_frame();
    }
  }
  ImTextureID bind_texture(Resource_ID id, u32 layer, u32 level, rd::Format format) {
    ImGui_ID iid;
    MEMZERO(iid);
    iid.id         = id;
    iid.base_layer = layer;
    iid.base_level = level;
    iid.format     = format;
    image_bindings.push(iid);
    return (ImTextureID)(size_t)(image_bindings.size - 1);
  }
  void frame() {
    on_frame();
    rd::Image2D_Info scinfo = factory->get_swapchain_image_info();
    width                   = scinfo.width;
    height                  = scinfo.height;
    timer.update();
    rd::Clear_Color cl;
    MEMZERO(cl);
    cl.clear = true;

    rd::Render_Pass_Create_Info info;
    MEMZERO(info);
    rd::RT_View rt0;
    MEMZERO(rt0);
    rt0.image             = factory->get_swapchain_image();
    rt0.format            = rd::Format::NATIVE;
    rt0.clear_color.clear = true;
    info.rts.push(rt0);
    rd::Imm_Ctx *ctx = factory->start_render_pass(info);

    static string_ref            shader    = stref_s(R"(
struct PushConstants
{
  float2 uScale;
  float2 uTranslate; 
  u32    control_flags;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : register(b0, space0);

#define CONTROL_DEPTH_ENABLE 1
#define CONTROL_AMPLIFY 2
#define is_control(flag) (pc.control_flags & flag) != 0

[[vk::binding(0, 0)]] Texture2D<float4>   sTexture          : register(t0, space0);
[[vk::binding(1, 0)]] SamplerState        sSampler          : register(s2, space0);

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
  return output;
}
#endif
#ifdef PIXEL

float4 main(in PSInput input) : SV_TARGET0 {
  if (is_control(CONTROL_DEPTH_ENABLE)) {
    float depth = sTexture.Sample(sSampler, input.UV).r;
    depth = pow(depth * 500.0f, 1.0f / 2.0f);
    return float4_splat(depth);
  } else {
    float4 color = input.Color * sTexture.Sample(sSampler, input.UV);
    if (is_control(CONTROL_AMPLIFY)) {
      color *= 10.0f;
      color.a = 1.0f;
    }
    return color;
  }
}
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    if (vs.is_null()) vs = factory->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null()) ps = factory->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);

    if (sampler.is_null()) {
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
      sampler             = factory->create_sampler(info);
    }

    if (!imgui_initialized) {
    }

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
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    static bool show_demo_window = true;
    ImGui::ShowDemoWindow(&show_demo_window);
    on_gui();
    ImGui::Render();

    draw_data = ImGui::GetDrawData();
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size       = (draw_data->TotalVtxCount + 1) * sizeof(ImDrawVert);
      vertex_buffer       = factory->create_buffer(buf_info);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size       = (draw_data->TotalIdxCount + 1) * sizeof(ImDrawIdx);
      index_buffer        = factory->create_buffer(buf_info);
    }
    {
      {
        ImDrawVert *vtx_dst = (ImDrawVert *)factory->map_buffer(vertex_buffer);

        ito(draw_data->CmdListsCount) {

          const ImDrawList *cmd_list = draw_data->CmdLists[i];
          memcpy(vtx_dst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));

          vtx_dst += cmd_list->VtxBuffer.Size;
        }
        factory->unmap_buffer(vertex_buffer);
      }
      {
        ImDrawIdx *idx_dst = (ImDrawIdx *)factory->map_buffer(index_buffer);
        ito(draw_data->CmdListsCount) {

          const ImDrawList *cmd_list = draw_data->CmdLists[i];

          memcpy(idx_dst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
          idx_dst += cmd_list->IdxBuffer.Size;
        }
        factory->unmap_buffer(index_buffer);
      }
    }
    if (font_pixels != NULL) {
      void *dst = factory->map_buffer(staging_buffer);
      memcpy(dst, font_pixels, font_width * font_height * 4);
      factory->unmap_buffer(staging_buffer);
      ctx->copy_buffer_to_image(staging_buffer, 0, font_texture, rd::Image_Copy::top_level());
      font_pixels = NULL;
    }
    setup_default_state(ctx);
    rd::Blend_State bs;
    MEMZERO(bs);
    bs.enabled          = true;
    bs.alpha_blend_op   = rd::Blend_OP::ADD;
    bs.color_blend_op   = rd::Blend_OP::ADD;
    bs.dst_alpha        = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
    bs.src_alpha        = rd::Blend_Factor::SRC_ALPHA;
    bs.dst_color        = rd::Blend_Factor::ONE_MINUS_SRC_ALPHA;
    bs.src_color        = rd::Blend_Factor::SRC_ALPHA;
    bs.color_write_mask = (u32)rd::Color_Component_Bit::R_BIT |
                          (u32)rd::Color_Component_Bit::G_BIT |
                          (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
    ctx->OM_set_blend_state(0, bs);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->bind_sampler(0, 1, sampler);
    ImVec2 clip_off          = draw_data->DisplayPos;
    ImVec2 clip_scale        = draw_data->FramebufferScale;
    int    global_vtx_offset = 0;
    int    global_idx_offset = 0;
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RG32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      ctx->IA_set_attribute(info);
    }
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RG32_FLOAT;
      info.location = 1;
      info.offset   = 8;
      info.type     = rd::Attriute_t::TEXCOORD0;
      ctx->IA_set_attribute(info);
    }
    {
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGBA8_UNORM;
      info.location = 2;
      info.offset   = 16;
      info.type     = rd::Attriute_t::TEXCOORD1;
      ctx->IA_set_attribute(info);
    }
    {
      float scale[2];
      scale[0] = 2.0f / draw_data->DisplaySize.x;
      scale[1] = 2.0f / draw_data->DisplaySize.y;
      float translate[2];
      translate[0] = -1.0f - draw_data->DisplayPos.x * scale[0];
      translate[1] = -1.0f - draw_data->DisplayPos.y * scale[1];
      ctx->push_constants(scale, 0, 8);
      ctx->push_constants(translate, 8, 8);
      u32 control = 0;
      ctx->push_constants(&control, 16, 4);
    }
    if (sizeof(ImDrawIdx) == 2)
      ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT16);
    else
      ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT32);
    ctx->IA_set_vertex_buffer(0, vertex_buffer, 0, sizeof(ImDrawVert), rd::Input_Rate::VERTEX);
    u32 control = 0;
    ctx->push_constants(&control, 16, 4);
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
        if (img.format == rd::Format::D32_FLOAT) {
          control = 1;
        }
        if (img.format == rd::Format::R32_UINT) {
          control    = 2;
          img.format = rd::Format::RGBA8_UNORM;
        }
        if (control != 0) {
          ctx->push_constants(&control, 16, 4);
        }
        rd::Image_Subresource range;
        range.layer      = img.base_layer;
        range.level      = img.base_level;
        range.num_layers = 1;
        range.num_levels = 1;
        ctx->bind_image(0, 0, 0, img.id, range, img.format);
        ctx->set_scissor(clip_rect.x, clip_rect.y, clip_rect.z - clip_rect.x,
                         clip_rect.w - clip_rect.y);
        ctx->draw_indexed(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, 0,
                          pcmd->VtxOffset + global_vtx_offset);
        if (control != 0) {
          control = 0;
          ctx->push_constants(&control, 16, 4);
        }
      }
      global_idx_offset += cmd_list->IdxBuffer.Size;
      global_vtx_offset += cmd_list->VtxBuffer.Size;
    }

    ImGuiIO &io = ImGui::GetIO();
    factory->release_resource(vertex_buffer);
    factory->release_resource(index_buffer);
    if (staging_buffer.is_null() == false) {
      factory->release_resource(staging_buffer);
    }
    vertex_buffer.reset();
    index_buffer.reset();
    staging_buffer.reset();
    image_bindings.size = 1;
    factory->end_render_pass(ctx);
  }
  void release() {
    on_release();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    factory->release_resource(vs);
    factory->release_resource(ps);
    factory->release();
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

static void init_buffer(rd::IFactory *factory, Resource_ID buf, void const *src, size_t size) {
  rd::Imm_Ctx *          ctx = factory->start_compute_pass();
  rd::Buffer_Create_Info info;
  MEMZERO(info);
  info.mem_bits       = (u32)rd::Memory_Bits::HOST_VISIBLE;
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

template <typename T> static Resource_ID create_uniform(rd::IFactory *factory, T const &src) {

  rd::Buffer_Create_Info info;
  MEMZERO(info);
  info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
  info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UNIFORM_BUFFER;
  info.size       = sizeof(T);
  Resource_ID out = factory->create_buffer(info);
  void *      dst = factory->map_buffer(out);
  memcpy(dst, &src, sizeof(src));
  factory->unmap_buffer(out);
  return out;
}

static Resource_ID get_or_create_image(rd::IFactory *factory, rd::Image_Create_Info const &ci,
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
  InlineArray<Resource_ID, 0x10> timestamps;
  double                         duration;
  void                           init() { timestamps.init(); }
  void                           insert(rd::IFactory *factory, rd::Imm_Ctx *ctx) {
    if (timestamps.size == 0x10) {
      ito(0x10) { factory->release_resource(timestamps[i]); }
      timestamps[0]   = ctx->insert_timestamp();
      timestamps.size = 1;
    } else {
      timestamps.push(ctx->insert_timestamp());
    }
  }
  void update(rd::IFactory *factory) {
    if (timestamps.size < 2) return;
    u32 cnt = 0;
    ito(timestamps.size / 2) {
      bool ready = factory->get_timestamp_state(timestamps[i * 2]) &&
                   factory->get_timestamp_state(timestamps[i * 2 + 1]);
      if (ready) {
        cnt++;
        duration +=
            (factory->get_timestamp_ms(timestamps[i * 2], timestamps[i * 2 + 1]) - duration) * 0.05;
      } else {
        break;
      }
    }
    ito(cnt) {
      factory->release_resource(timestamps[i * 2]);
      factory->release_resource(timestamps[i * 2 + 1]);
    }
    for (u32 i = cnt * 2; i < timestamps.size; i++) {
      timestamps[i - cnt * 2] = timestamps[i];
    }
    timestamps.size -= cnt * 2;
  }
  void release(rd::IFactory *factory) {
    ito(0x10) { factory->release_resource(timestamps[i]); }
  }
};

#if 1
struct Mip_Builder {
  static Resource_ID create_image(rd::IFactory *factory, Image2D const *image,
                                  u32 usage = (u32)rd::Image_Usage_Bits::USAGE_SAMPLED) {
    Resource_ID           output_image;
    rd::Image_Create_Info info;
    usage |= (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST;
    MEMZERO(info);
    info.format     = image->format;
    info.width      = image->width;
    info.height     = image->height;
    info.depth      = 1;
    info.layers     = 1;
    info.levels     = image->get_num_mip_levels();
    info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
    info.usage_bits = usage;
    output_image    = factory->create_image(info);
    Resource_ID cs;
    Resource_ID staging_buffer;

    rd::Buffer_Create_Info buf_info;
    MEMZERO(buf_info);
    buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
    buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC |
                          (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST |
                          (u32)rd::Buffer_Usage_Bits::USAGE_UAV;
    buf_info.size  = image->get_size_in_bytes() * 2;
    staging_buffer = factory->create_buffer(buf_info);

    TMP_STORAGE_SCOPE;
    cs = factory->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
struct PushConstants
{
  u32 src_offset;
  u32 src_width;
  u32 src_height;
  u32 dst_offset;
  u32 dst_width;
  u32 dst_height;
  u32 format;
  u32 op; 
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : register(b0, space0);

#define RGBA8_SRGBA    0
#define RGBA8_UNORM    1
#define RGB32_FLOAT    2
#define R32_FLOAT      3
#define RGBA32_FLOAT   4

#define OP_AVG 0
#define OP_MAX 1
#define OP_MIN 2
#define OP_SUM 3

[[vk::binding(0, 0)]] RWStructuredBuffer<float> data_f32 : register(u0, space0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  data_u32 : register(u1, space0);

float4 load(int2 coord) {
  if (coord.x >= pc.src_width) coord.x = int(pc.src_width) - 1;
  if (coord.y >= pc.src_height) coord.y = int(pc.src_height) - 1;
  if (coord.x < 0) coord.x = 0;
  if (coord.y < 0) coord.y = 0;
  if (pc.format == RGBA8_SRGBA) {
    uint pixel = data_u32[coord.x + coord.y * pc.src_width + pc.src_offset];
    float4 o =
        float4(
              float((pixel >> 0u ) & 0xffu) / 255.0,
              float((pixel >> 8u ) & 0xffu) / 255.0,
              float((pixel >> 16u) & 0xffu) / 255.0,
              float((pixel >> 24u) & 0xffu) / 255.0);
    return pow(o, float4_splat(2.2));
  } else if (pc.format == RGBA8_UNORM) {
    uint pixel = data_u32[coord.x + coord.y * pc.src_width + pc.src_offset];
    float4 o =
        float4(
              float((pixel >> 0u ) & 0xffu) / 255.0,
              float((pixel >> 8u ) & 0xffu) / 255.0,
              float((pixel >> 16u) & 0xffu) / 255.0,
              float((pixel >> 24u) & 0xffu) / 255.0);
    return o;
  } else if (pc.format == RGB32_FLOAT) {
    float v_0 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset) * 3 + 0];
    float v_1 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset) * 3 + 1];
    float v_2 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset) * 3 + 2];
    return float4(v_0, v_1, v_2, 1.0f);
  } else if (pc.format == RGBA32_FLOAT) {
    float v_0 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset) * 4 + 0];
    float v_1 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset) * 4 + 1];
    float v_2 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset) * 4 + 2];
    float v_3 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset) * 4 + 3];
    return float4(v_0, v_1, v_2, v_3);
  } else if (pc.format == R32_FLOAT) {
    float v_0 = data_f32[(coord.x + coord.y * pc.src_width + pc.src_offset)];
    return float4(v_0, 0.0f, 0.0f, 0.0f);
  }
  return float4(1.0, 0.0, 0.0, 1.0);
}

void store(int2 coord, float4 val) {
  if (pc.format == RGBA8_SRGBA) {
    val = pow(val, float4_splat(1.0/2.2));
    uint r = uint(clamp(val.x * 255.0f, 0.0f, 255.0f));
    uint g = uint(clamp(val.y * 255.0f, 0.0f, 255.0f));
    uint b = uint(clamp(val.z * 255.0f, 0.0f, 255.0f));
    uint a = uint(clamp(val.w * 255.0f, 0.0f, 255.0f));
    data_u32[coord.x + coord.y * pc.dst_width + pc.dst_offset] = ((r&0xffu)  |
                                                   ((g&0xffu)  << 8u)  |
                                                   ((b&0xffu)  << 16u) |
                                                   ((a&0xffu)  << 24u));
  } else if (pc.format == RGBA8_UNORM) {
    uint r = uint(clamp(val.x * 255.0f, 0.0f, 255.0f));
    uint g = uint(clamp(val.y * 255.0f, 0.0f, 255.0f));
    uint b = uint(clamp(val.z * 255.0f, 0.0f, 255.0f));
    uint a = uint(clamp(val.w * 255.0f, 0.0f, 255.0f));
    data_u32[coord.x + coord.y * pc.dst_width + pc.dst_offset] = ((r & 0xffu)  |
                                                   ((g & 0xffu)  << 8u)  |
                                                   ((b & 0xffu)  << 16u) |
                                                   ((a & 0xffu)  << 24u));
  } else if (pc.format == RGB32_FLOAT) {
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 3] = val.x;
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 3 + 1] = val.y;
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 3 + 2] = val.z;
  } else if (pc.format == RGBA32_FLOAT) {
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 4] = val.x;
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 4 + 1] = val.y;
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 4 + 2] = val.z;
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 4 + 3] = val.w;
  } else if (pc.format == R32_FLOAT) {
    data_f32[(coord.x + coord.y * pc.dst_width + pc.dst_offset)] = val.x;
  }
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.dst_width || tid.y >= pc.dst_height)
      return;

    int2 xy = int2(tid.xy);

    float4 val_0 = load(xy * 2);
    float4 val_1 = load(xy * 2 + int2(1, 0));
    float4 val_2 = load(xy * 2 + int2(0, 1));
    float4 val_3 = load(xy * 2 + int2(1, 1));
    float4 result = float4_splat(0.0);

    if (pc.op == OP_AVG)
      result = (val_0 + val_1 + val_2 + val_3) / 4.0;
    else if (pc.op == OP_MAX)
      result = max(val_0, max(val_1, max(val_2, val_3)));
    else if (pc.op == OP_MIN)
      result = min(val_0, min(val_1, min(val_2, val_3)));
    else if (pc.op == OP_SUM)
      result = val_0 + val_1 + val_2 + val_3;
    store(int2(tid.xy), result);
}

)"),
                                    NULL, 0);

    void *ptr = factory->map_buffer(staging_buffer);
    memcpy(ptr, image->data, image->get_size_in_bytes());
    factory->unmap_buffer(staging_buffer);
    struct Push_Constants {
      u32 src_offset;
      u32 src_width;
      u32 src_height;
      u32 dst_offset;
      u32 dst_width;
      u32 dst_height;
      u32 format;
      u32 op;
    } pc;
    MEMZERO(pc);
    pc.op = 0;
    switch (image->format) {
      // clang-format off
      case rd::Format::RGBA8_SRGBA:  {  pc.format = 0; } break;
      case rd::Format::RGBA8_UNORM:  {  pc.format = 1; } break;
      case rd::Format::RGB32_FLOAT:  {  pc.format = 2; } break;
      case rd::Format::R32_FLOAT:    {  pc.format = 3; } break;
      case rd::Format::RGBA32_FLOAT: {  pc.format = 4; } break;
      // clang-format on
    default: TRAP;
    }
    rd::Imm_Ctx *ctx = factory->start_compute_pass();
    ctx->bind_storage_buffer(0, 0, staging_buffer, 0, 0);
    ctx->bind_storage_buffer(0, 1, staging_buffer, 0, 0);
    ctx->CS_set_shader(cs);

    InlineArray<u32, 0x10>  mip_offsets;
    InlineArray<int2, 0x10> mip_sizes;
    mip_offsets.init();
    mip_sizes.init();
    u32 w          = image->width;
    u32 h          = image->height;
    u32 mip_offset = 0;
    while (w || h) {
      mip_offsets.push(mip_offset);
      w = MAX(1, w);
      h = MAX(1, h);
      mip_sizes.push({w, h});
      mip_offset += w * h * image->get_bpp();
      w = w >> 1;
      h = h >> 1;
    }

    for (u32 i = 0; i < mip_offsets.size - 1; i++) {
      pc.src_offset = mip_offsets[i] / 4;
      pc.src_width  = mip_sizes[i].x;
      pc.src_height = mip_sizes[i].y;
      pc.dst_offset = mip_offsets[i + 1] / 4;
      pc.dst_width  = mip_sizes[i + 1].x;
      pc.dst_height = mip_sizes[i + 1].y;
      ctx->push_constants(&pc, 0, sizeof(pc));
      ctx->buffer_barrier(staging_buffer,
                          (u32)rd::Access_Bits::SHADER_READ | (u32)rd::Access_Bits::SHADER_WRITE);
      ctx->dispatch((mip_sizes[i + 1].x + 15) / 16, (mip_sizes[i + 1].y + 15) / 16, 1);
    }
    ito(mip_offsets.size) {
      rd::Image_Copy dst_info;
      MEMZERO(dst_info);
      dst_info.level      = i;
      dst_info.num_layers = 1;
      dst_info.size_x     = mip_sizes[i].x;
      dst_info.size_y     = mip_sizes[i].y;
      dst_info.size_z     = 1;
      ctx->image_barrier(output_image, (u32)rd::Access_Bits::MEMORY_WRITE,
                         rd::Image_Layout::TRANSFER_DST_OPTIMAL);
      ctx->copy_buffer_to_image(staging_buffer, mip_offsets[i], output_image, dst_info);
    }
    factory->end_compute_pass(ctx);
    factory->release_resource(cs);
    factory->release_resource(staging_buffer);
    return output_image;
  }
};

struct Raw_Mesh_3p16i_Wrapper {
  Resource_ID vertex_buffer;
  Resource_ID index_buffer;
  u32         num_indices;
  u32         num_vertices;

  void release(rd::IFactory *rm) {
    rm->release_resource(vertex_buffer);
    rm->release_resource(index_buffer);
    MEMZERO(*this);
  }
  void init(rd::IFactory *rm, float3 const *positions, u16 const *indices, u32 num_indices,
            u32 num_vertices) {
    this->num_indices  = num_indices;
    this->num_vertices = num_vertices;
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size       = u32(sizeof(float3) * num_vertices);
      vertex_buffer       = rm->create_buffer(buf_info);
      memcpy(rm->map_buffer(vertex_buffer), &positions[0], sizeof(float3) * num_vertices);
      rm->unmap_buffer(vertex_buffer);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size       = u32(sizeof(u16) * num_indices);
      index_buffer        = rm->create_buffer(buf_info);
      memcpy(rm->map_buffer(index_buffer), &indices[0], sizeof(u16) * num_indices);
      rm->unmap_buffer(index_buffer);
    }
  }
  void init(rd::IFactory *rm, Raw_Mesh_3p16i const &model) {
    num_indices  = model.indices.size * 3;
    num_vertices = model.positions.size;
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size       = u32(sizeof(float3) * model.positions.size);
      vertex_buffer       = rm->create_buffer(buf_info);
      memcpy(rm->map_buffer(vertex_buffer), &model.positions[0],
             sizeof(float3) * model.positions.size);
      rm->unmap_buffer(vertex_buffer);
    }
    {
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER;
      buf_info.size       = u32(sizeof(u16_face) * model.indices.size);
      index_buffer        = rm->create_buffer(buf_info);
      memcpy(rm->map_buffer(index_buffer), &model.indices[0],
             sizeof(u16_face) * model.indices.size);
      rm->unmap_buffer(index_buffer);
    }
  }
  void draw(rd::Imm_Ctx *ctx, u32 instances = 1, u32 first_instance = 0) {
    ctx->IA_set_vertex_buffer(0, vertex_buffer, 0, 12, rd::Input_Rate::VERTEX);
    ctx->IA_set_index_buffer(index_buffer, 0, rd::Index_t::UINT16);
    ctx->draw_indexed(num_indices, instances, 0, first_instance, 0);
  }
};

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
  Array<Gizmo_Instance_Data_CPU, 0x1000> cylinder_draw_cmds;
  Array<Gizmo_Instance_Data_CPU, 0x1000> sphere_draw_cmds;
  Array<Gizmo_Instance_Data_CPU, 0x1000> cone_draw_cmds;
  Array<Gizmo_Line_Vertex, 0x1000>       line_segments;
  Pool<char>                             char_storage = {};
  struct _String2D {
    char *   c_str;
    uint32_t len;
    float    x, y, z;
    float3   color;
  };
  Array<_String2D, 0x1000> strings = {};

  Raw_Mesh_3p16i_Wrapper icosahedron_wrapper = {};
  Raw_Mesh_3p16i_Wrapper cylinder_wrapper    = {};
  Raw_Mesh_3p16i_Wrapper cone_wrapper        = {};
  Raw_Mesh_3p16i_Wrapper glyph_wrapper       = {};

  Resource_ID gizmo_vs;
  Resource_ID gizmo_ps;
  Resource_ID gizmo_lines_vs;
  Resource_ID gizmo_lines_ps;
  Resource_ID font_ps      = {};
  Resource_ID font_vs      = {};
  Resource_ID font_sampler = {};
  Resource_ID font_texture = {};

  rd::IFactory *rm = NULL;
  Camera        g_camera;
  bool          update_mouse_ray = true;
  Ray           mouse_ray;
  float2        mouse_cursor;
  float2        resolution;
  enum Mode {
    NONE = 0,
    CAMERA_DRAG,
    GIZMO_DRAG,
  };
  Mode               mode = NONE;
  AutoArray<Gizmo *> gizmos;
  AutoArray<Gizmo *> selected_gizmos;
  Gizmo *            hovered_gizmo = NULL;
  bool               mb[3]         = {};
  bool               last_mb[3]    = {};
  int2               mpos          = {};
  int2               last_mpos     = {};
  Timer              timer;
  bool               keys[0x100]      = {};
  bool               last_keys[0x100] = {};

  void init(rd::IFactory *rm) {
    this->rm = rm;
    timer.init();
    g_camera.init();
    cylinder_draw_cmds.init();
    char_storage = Pool<char>::create(1 * (1 << 20));
    sphere_draw_cmds.init();
    cone_draw_cmds.init();
    line_segments.init();
    strings.init();
    {
      auto mesh = subdivide_cone(8, 1.0f, 1.0f);
      cone_wrapper.init(rm, mesh);
      mesh.release();
    }
    {
      auto mesh = subdivide_icosahedron(2);
      icosahedron_wrapper.init(rm, mesh);
      mesh.release();
    }
    {
      auto mesh = subdivide_cylinder(8, 1.0f, 1.0f);
      cylinder_wrapper.init(rm, mesh);
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
      glyph_wrapper.init(rm, (float3 *)pos, indices, 6, 6);
    }
    static string_ref            shader    = stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : register(b0, space0);

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
    gizmo_vs = rm->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    gizmo_ps = rm->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);
    static string_ref shader_lines = stref_s(R"(
struct PushConstants
{
  float4x4 viewproj;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : register(b0, space0);

struct PSInput {
  [[vk::location(0)]] float4 pos            : SV_POSITION;
  [[vk::location(1)]] float3 pixel_color    : TEXCOORD0;
};

#ifdef VERTEX

struct VSInput {
  [[vk::location(0)]] float3 in_position    : POSITION;
  [[vk::location(1)]] float3 in_color       : TEXCOORD4;
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
    gizmo_lines_vs = rm->create_shader_raw(rd::Stage_t::VERTEX, shader_lines, &defines[0], 1);
    gizmo_lines_ps = rm->create_shader_raw(rd::Stage_t::PIXEL, shader_lines, &defines[1], 1);
    {
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.width    = simplefont_bitmap_width;
      info.height   = simplefont_bitmap_height;
      info.depth    = 1;
      info.layers   = 1;
      info.levels   = 1;
      info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits =
          (u32)rd::Image_Usage_Bits::USAGE_SAMPLED | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      info.format  = rd::Format::R8_UNORM;
      font_texture = rm->create_image(info);

      Resource_ID staging_buf{};
      defer(rm->release_resource(staging_buf));
      // RenderDoc_CTX::get().start();
      {
        rd::Buffer_Create_Info buf_info;
        MEMZERO(buf_info);
        buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
        buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_SRC;
        buf_info.size       = simplefont_bitmap_width * simplefont_bitmap_height;
        staging_buf         = rm->create_buffer(buf_info);
        u8 *ptr             = (u8 *)rm->map_buffer(staging_buf);
        // memcpy(ptr, &simplefont_bitmap[0], buf_info.size);
        yto(simplefont_bitmap_height) {
          xto(simplefont_bitmap_width) {
            ptr[simplefont_bitmap_width * y + x] = simplefont_bitmap[y][x] == ' ' ? 0 : 0xffu;
          }
        }
        rm->unmap_buffer(staging_buf);
        auto ctx = rm->start_compute_pass();
        ctx->image_barrier(font_texture, (u32)rd::Access_Bits::MEMORY_WRITE,
                           rd::Image_Layout::TRANSFER_DST_OPTIMAL);
        ctx->copy_buffer_to_image(staging_buf, 0, font_texture, rd::Image_Copy::top_level());
        rm->end_compute_pass(ctx);
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
[[vk::push_constant]] ConstantBuffer<PushConstants> pc : register(b0, space0);

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
  output.uv = input.instance_uv_offset + (input.vertex_position.xy * float2(1.0, -1.0) + float2(0.0, 1.0)) * pc.glyph_uv_size;
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
[[vk::binding(1, 0)]] SamplerState        my_sampler            : register(s2, space0);

float4 main(in PSInput input) : SV_TARGET0 {
  if (font_texture.Sample(my_sampler, input.uv).x < 0.5f)
    discard;
  return
    float4(input.color.xyz, 1.0f);
}
#endif
)";
    font_vs = rm->create_shader_raw(rd::Stage_t::VERTEX, stref_s(font_shader), &defines[0], 1);
    font_ps = rm->create_shader_raw(rd::Stage_t::PIXEL, stref_s(font_shader), &defines[1], 1);

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
    font_sampler        = rm->create_sampler(info);
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
  static Gizmo_Layer *create(rd::IFactory *rm) {
    Gizmo_Layer *gl = new Gizmo_Layer;
    gl->init(rm);
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
    cylinder_wrapper.release(rm);
    icosahedron_wrapper.release(rm);
    cylinder_wrapper.release(rm);
    rm->release_resource(gizmo_ps);
    rm->release_resource(gizmo_vs);
    delete this;
  }
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
  void render(rd::IFactory *f, rd::Imm_Ctx *ctx, u32 width, u32 height) {
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
      ctx->push_state();
      defer(ctx->pop_state());
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
      ctx->RS_set_state(rs_state);
      u32                    cylinder_offset = 0;
      u32                    num_cylinders   = cylinder_draw_cmds.size;
      u32                    sphere_offset   = num_cylinders;
      u32                    num_spheres     = sphere_draw_cmds.size;
      u32                    cone_offset     = num_cylinders + num_spheres;
      u32                    num_cones       = cone_draw_cmds.size;
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size = (cylinder_draw_cmds.size + sphere_draw_cmds.size + cone_draw_cmds.size) *
                      sizeof(Gizmo_Instance_Data_CPU);
      Resource_ID gizmo_instance_buffer = f->create_buffer(buf_info);
      void *      ptr                   = f->map_buffer(gizmo_instance_buffer);
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
      f->unmap_buffer(gizmo_instance_buffer);
      cylinder_draw_cmds.reset();
      cone_draw_cmds.reset();
      sphere_draw_cmds.reset();
      defer(f->release_resource(gizmo_instance_buffer));
      ctx->PS_set_shader(gizmo_ps);
      ctx->VS_set_shader(gizmo_vs);
      ctx->push_constants(&viewproj, 0, sizeof(viewproj));
      ctx->IA_set_vertex_buffer(1, gizmo_instance_buffer, 0, sizeof(Gizmo_Instance_Data_CPU),
                                rd::Input_Rate::INSTANCE);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      ctx->IA_set_attribute(info);
      ito(5) {
        MEMZERO(info);
        info.binding  = 1;
        info.format   = rd::Format::RGBA32_FLOAT;
        info.location = 1 + i;
        info.offset   = 16 * i;
        info.type     = (rd::Attriute_t)((u32)rd::Attriute_t::TEXCOORD0 + i);
        ctx->IA_set_attribute(info);
      }
      ctx->IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      cylinder_wrapper.draw(ctx, num_cylinders, cylinder_offset);
      icosahedron_wrapper.draw(ctx, num_spheres, sphere_offset);
      cone_wrapper.draw(ctx, num_cones, cone_offset);
    }
    if (strings.size != 0) {
      ctx->push_state();
      defer(ctx->pop_state());
      defer(strings.reset());
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
      ctx->RS_set_state(rs_state);
      struct Glyph_Instance {
        float x, y, z;
        float u, v;
        float r, g, b;
      };
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
      if (num_glyphs == 0) goto skip_strings;
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits                 = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits               = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size                     = num_glyphs * sizeof(Glyph_Instance);
      Resource_ID gizmo_instance_buffer = f->create_buffer(buf_info);
      rm->release_resource(gizmo_instance_buffer);
      Glyph_Instance *ptr = (Glyph_Instance *)f->map_buffer(gizmo_instance_buffer);
      memcpy(ptr, glyphs, buf_info.size);
      f->unmap_buffer(gizmo_instance_buffer);
      ctx->IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      ctx->IA_set_vertex_buffer(1, gizmo_instance_buffer, 0, sizeof(Glyph_Instance),
                                rd::Input_Rate::INSTANCE);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      ctx->IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 1;
      info.offset   = 0;
      info.type     = rd::Attriute_t::TEXCOORD0;
      ctx->IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RG32_FLOAT;
      info.location = 2;
      info.offset   = 12;
      info.type     = rd::Attriute_t::TEXCOORD1;
      ctx->IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 1;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 3;
      info.offset   = 20;
      info.type     = rd::Attriute_t::TEXCOORD2;
      ctx->IA_set_attribute(info);
      struct PC {
        float2 glyph_uv_size;
        float2 glyph_size;
        float2 viewport_size;
      } pc;
      pc.glyph_uv_size = {glyph_uv_width, glyph_uv_height};
      pc.glyph_size    = {glyphs_screen_width, -glyphs_screen_height};
      pc.viewport_size = {(f32)width, (f32)height};
      ctx->VS_set_shader(font_vs);
      ctx->PS_set_shader(font_ps);
      ctx->push_constants(&pc, 0, sizeof(pc));
      ctx->bind_sampler(0, 1, font_sampler);
      ctx->bind_image(0, 0, 0, font_texture, rd::Image_Subresource::top_level(),
                      rd::Format::NATIVE);
      glyph_wrapper.draw(ctx, num_glyphs, 0);
    skip_strings:;
    }
    if (line_segments.size != 0) {
      ctx->push_state();
      defer(ctx->pop_state());
      ctx->RS_set_line_width(2.2f);
      ctx->RS_set_depth_bias(-30.0f);

      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits                 = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits               = (u32)rd::Buffer_Usage_Bits::USAGE_VERTEX_BUFFER;
      buf_info.size                     = (line_segments.size) * sizeof(Gizmo_Line_Vertex);
      Resource_ID gizmo_instance_buffer = f->create_buffer(buf_info);
      void *      ptr                   = f->map_buffer(gizmo_instance_buffer);
      memcpy((u8 *)ptr, &line_segments[0], line_segments.size * sizeof(Gizmo_Line_Vertex));
      f->unmap_buffer(gizmo_instance_buffer);
      defer(f->release_resource(gizmo_instance_buffer));
      ctx->IA_set_topology(rd::Primitive::LINE_LIST);
      ctx->IA_set_vertex_buffer(0, gizmo_instance_buffer, 0, sizeof(Gizmo_Line_Vertex),
                                rd::Input_Rate::VERTEX);
      rd::Attribute_Info info;
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 0;
      info.offset   = 0;
      info.type     = rd::Attriute_t::POSITION;
      ctx->IA_set_attribute(info);
      MEMZERO(info);
      info.binding  = 0;
      info.format   = rd::Format::RGB32_FLOAT;
      info.location = 1;
      info.offset   = 12;
      info.type     = rd::Attriute_t::TEXCOORD0;
      ctx->IA_set_attribute(info);
      ctx->PS_set_shader(gizmo_lines_ps);
      ctx->VS_set_shader(gizmo_lines_vs);
      ctx->push_constants(&viewproj, 0, sizeof(viewproj));
      ctx->draw(line_segments.size, 1, 0, 0);
      line_segments.reset();
    }
  }
};

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

template <typename T> static void render_bvh(float4x4 const &t, BVH<T> *bvh, Gizmo_Layer *gl) {
  ASSERT_DEBUG(bvh);
  bvh->traverse([&](BVH_Node *node) {
    gl->render_linebox(transform(t, node->min), transform(t, node->max), float3(1.0f, 0.0f, 0.0f));
  });
}
#endif // RENDERING_UTILS_HPP
