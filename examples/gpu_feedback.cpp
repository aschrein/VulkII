#include "rendering.hpp"
#include "script.hpp"

#include "rendering_utils.hpp"
#include "scene.hpp"

#ifdef __linux__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

Config g_config;
Camera g_camera;
Scene  g_scene;

static void init_traverse(List *l) {
  if (l == NULL) return;
  if (l->child) {
    init_traverse(l->child);
    init_traverse(l->next);
  } else {
    if (l->cmp_symbol("camera")) {
      g_camera.traverse(l->next);
    } else if (l->cmp_symbol("config")) {
      g_config.traverse(l->next);
    }
  }
}

static int g_init = []() {
  TMP_STORAGE_SCOPE;
  g_camera.init();
  g_config.init(stref_s(R"(
(config
 (add bool enable_rasterization_pass 1)
 (add bool enable_compute_depth 1)
 (add bool enable_compute_render_pass 1)
 (add bool enable_meshlets_render_pass 1)
 (add u32 g_buffer_width 512 (min 4) (max 1024))
 (add u32 g_buffer_height 512 (min 4) (max 1024))
)
)"));

  char *state = read_file_tmp("scene_state");

  if (state != NULL) {
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
    init_traverse(cur);
  }
  g_scene.init();
  g_scene.load_mesh(stref_s("HIGH"),
                    stref_s("models/human_skull_and_neck/scene.gltf"));
  return 0;
}();

static_defer({
  FILE *scene_dump = fopen("scene_state", "wb");
  fprintf(scene_dump, "(\n");
  defer(fclose(scene_dump));
  g_camera.dump(scene_dump);
  g_config.dump(scene_dump);
  fprintf(scene_dump, ")\n");
});

class Feedback_Pass : public rd::IPass {
  Resource_ID          vs;
  Resource_ID          ps;
  Resource_ID          output_image;
  u32                  width, height;
  static constexpr u32 NUM_BUFFERS = 8;
  struct Feedback_Buffer {
    Resource_ID buffer;
    Resource_ID fence;
    bool        in_fly;
    void        init(rd::IResource_Manager *rm) {
      MEMZERO(*this);
      in_fly = false;
      rd::Buffer_Create_Info buf_info;
      MEMZERO(buf_info);
      buf_info.mem_bits   = (u32)rd::Memory_Bits::HOST_VISIBLE;
      buf_info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_UAV |
                            (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      buf_info.size = 16;
      buffer        = rm->create_buffer(buf_info);

      fence = rm->get_fence(rd::Fence_Position::PASS_FINISED);
    }
    void release(rd::IResource_Manager *rm) {
      if (buffer.is_null() == false) rm->release_resource(buffer);
      MEMZERO(*this);
    }
  };
  InlineArray<Feedback_Buffer, NUM_BUFFERS> feedback_buffers;

  public:
  Feedback_Pass() {
    feedback_buffers.init();
    output_image.reset();
    vs.reset();
    ps.reset();
    width  = 0;
    height = 0;
  }
  void on_end(rd::IResource_Manager *rm) override { g_scene.on_pass_end(rm); }
  void on_begin(rd::IResource_Manager *pc) override {
    g_camera.update();
    g_scene.on_pass_begin(pc);
    rd::Image2D_Info info = pc->get_swapchain_image_info();
    if (output_image.is_null()) {

      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::RGBA32_FLOAT;
      info.width      = 1024;
      info.height     = 1024;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_UAV |
                        (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST |
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      output_image = pc->create_image(info);
      pc->assign_name(output_image, stref_s("feedback_pass/img0"));
    }
    width  = 1024;
    height = 1024;
    {
      rd::Clear_Depth cl;
      cl.clear = true;
      cl.d     = 0.0f;
      rd::Image_Create_Info info;
      MEMZERO(info);
      info.format     = rd::Format::D32_FLOAT;
      info.width      = 1024;
      info.height     = 1024;
      info.depth      = 1;
      info.layers     = 1;
      info.levels     = 1;
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT |
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
      pc->add_depth_target(stref_s("feedback_pass/ds"), info, 0, 0, cl);
    }
    {
      rd::Clear_Color cl;
      cl.clear = true;
      cl.r     = 0.0f;
      cl.g     = 0.0f;
      cl.b     = 0.0f;
      cl.a     = 1.0f;
      rd::Image_Create_Info rt0_info;
      MEMZERO(rt0_info);
      rt0_info.format     = rd::Format::RGBA32_FLOAT;
      rt0_info.width      = 1024;
      rt0_info.height     = 1024;
      rt0_info.depth      = 1;
      rt0_info.layers     = 1;
      rt0_info.levels     = 1;
      rt0_info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      rt0_info.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_RT |
                            (u32)rd::Image_Usage_Bits::USAGE_SAMPLED |
                            (u32)rd::Image_Usage_Bits::USAGE_UAV;
      pc->add_render_target(stref_s("feedback_pass/rt0"), rt0_info, 0, 0, cl);
    }
    static string_ref            shader    = stref_s(R"(
@(DECLARE_PUSH_CONSTANTS
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float4x4)  (name world_transform))
)

@(DECLARE_BUFFER
  (type WRITE_ONLY)
  (set 0)
  (binding 0)
  (type uint)
  (name out_cnt)
)

@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 1)
  (format RGBA32_FLOAT)
  (name out_image)
)

#ifdef VERTEX

@(DECLARE_INPUT (location 0) (type float3) (name POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name NORMAL))
@(DECLARE_INPUT (location 4) (type float2) (name TEXCOORD0))

@(DECLARE_OUTPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_OUTPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_OUTPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(ENTRY)
  PIXEL_POSITION  = POSITION;
  PIXEL_TEXCOORD0 = TEXCOORD0;
  PIXEL_NORMAL = NORMAL;
  float3 position = POSITION;
  // @(EXPORT_POSITION float4(TEXCOORD0, 0.0, 1.0));
  @(EXPORT_POSITION viewproj * float4(position, 1.0));
@(END)
#endif
#ifdef PIXEL
@(DECLARE_INPUT (location 0) (type float3) (name PIXEL_POSITION))
@(DECLARE_INPUT (location 1) (type float3) (name PIXEL_NORMAL))
@(DECLARE_INPUT (location 2) (type float2) (name PIXEL_TEXCOORD0))

@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  int2 dim = imageSize(out_image);
  i32 x = i32(0.5 + dim.x * PIXEL_TEXCOORD0.x);
  i32 y = i32(0.5 + dim.y * PIXEL_TEXCOORD0.y + 1.0);
  image_store(out_image, int2(x, y), float4(1.0, 0.0, 0.0, 1.0));
  
  buffer_atomic_add(out_cnt, 0, 1);

  float4 color = float4_splat(1.0) * (0.5 + 0.5 * dot(PIXEL_NORMAL.rgb, normalize(float3(1.0, 1.0, 1.0))));
  @(EXPORT_COLOR 0 color);
@(END)
#endif
)");
    Pair<string_ref, string_ref> defines[] = {
        {stref_s("VERTEX"), {}},
        {stref_s("PIXEL"), {}},
    };
    if (vs.is_null())
      vs = pc->create_shader_raw(rd::Stage_t::VERTEX, shader, &defines[0], 1);
    if (ps.is_null())
      ps = pc->create_shader_raw(rd::Stage_t::PIXEL, shader, &defines[1], 1);
    bool found_free = false;
    ito(feedback_buffers.size) {
      if (feedback_buffers[i].in_fly) continue;
      found_free = true;
    }
    if (!found_free && feedback_buffers.isfull() == false) {
      feedback_buffers.push({});
      ASSERT_ALWAYS(feedback_buffers.size != 0);
      feedback_buffers[feedback_buffers.size - 1].init(pc);
    }
  }
  void exec(rd::Imm_Ctx *ctx) override {
    setup_default_state(ctx, 1);
    rd::DS_State ds_state;
    MEMZERO(ds_state);
    ds_state.cmp_op             = rd::Cmp::GE;
    ds_state.enable_depth_test  = true;
    ds_state.enable_depth_write = true;
    ctx->DS_set_state(ds_state);
    ctx->VS_set_shader(vs);
    ctx->PS_set_shader(ps);
    ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
    ctx->set_scissor(0, 0, width, height);
    rd::RS_State rs_state;
    MEMZERO(rs_state);
    rs_state.polygon_mode = rd::Polygon_Mode::FILL;
    rs_state.front_face   = rd::Front_Face::CW;
    rs_state.cull_mode    = rd::Cull_Mode::BACK;
    rs_state.line_width   = 1.0f;
    rs_state.depth_bias   = 0.0f;
    ctx->RS_set_state(rs_state);
    g_scene.gfx_bind(ctx);
    float4x4 viewproj = g_camera.viewproj();
    ctx->push_constants(&viewproj, 0, sizeof(float4x4));
    {
      rd::Clear_Value cv;
      MEMZERO(cv);
      cv.v_f32[0] = 0.0f;
      cv.v_f32[1] = 0.0f;
      cv.v_f32[2] = 0.0f;
      cv.v_f32[3] = 1.0f;
      ctx->clear_image(output_image, 0, 1, 0, 1, cv);
      ctx->bind_rw_image(0, 1, 0, output_image, 0, 1, 0, 1);
    }
    {

      bool found_free = false;
      while (!found_free) {
        ito(feedback_buffers.size) {
          Feedback_Buffer &feedback_buffer = feedback_buffers[i];
          if (feedback_buffer.in_fly &&
              ctx->get_fence_state(feedback_buffer.fence)) {
            u32 *ptr = (u32 *)ctx->map_buffer(feedback_buffer.buffer);
            fprintf(stdout, "feedback buffer is finished: %i ... \n", ptr[0]);
            ctx->unmap_buffer(feedback_buffer.buffer);
            feedback_buffer.in_fly = false;
          }
        }
        ito(feedback_buffers.size) {
          Feedback_Buffer &feedback_buffer = feedback_buffers[i];
          if (feedback_buffer.in_fly == false) {
            ctx->fill_buffer(feedback_buffer.buffer, 0, 4, 0);
            ctx->bind_storage_buffer(0, 0, feedback_buffer.buffer, 0, 4);
            feedback_buffer.in_fly = true;
            found_free             = true;
            break;
          }
        }
      }
    }
    g_scene.traverse([&](Node *node) {
      if (isa<MeshNode>(node)) {
        GfxMesh *gfxmesh = ((GfxMesh *)((MeshNode *)node)->get_mesh());
        ctx->push_constants(&node->get_transform(), 64, sizeof(float4x4));
        gfxmesh->draw(ctx, g_scene.vertex_buffer,
                      g_scene.mesh_offsets.get(gfxmesh));
      }
    });
  }
  string_ref get_name() override { return stref_s("feedback_pass"); }
  void       release(rd::IResource_Manager *rm) override {
    rm->release_resource(vs);
    rm->release_resource(ps);
    ito(feedback_buffers.size) feedback_buffers[i].release(rm);
    feedback_buffers.release();
    delete this;
  }
};

class Event_Consumer : public rd::IEvent_Consumer {
  i32 last_m_x = -1;
  i32 last_m_y = -1;

  public:
  void consume(void *_event) override {

    SDL_Event *event = (SDL_Event *)_event;
    if (event->type == SDL_MOUSEMOTION) {
      SDL_MouseMotionEvent *m       = (SDL_MouseMotionEvent *)event;
      i32                   cur_m_x = m->x;
      i32                   cur_m_y = m->y;
      if ((m->state & 1) != 0 && last_m_x > 0) {
        i32 dx = cur_m_x - last_m_x;
        i32 dy = cur_m_y - last_m_y;
        g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
        g_camera.theta += (float)(dy)*5.0e-3f;
      }
      last_m_x = cur_m_x;
      last_m_y = cur_m_y;
    }
  }
};

class GUI_Pass : public IGUI_Pass {
  public:
  void on_gui(rd::IResource_Manager *pc) override {
    {
      ImGui::Begin("Config");
      ImGui::End();
    }
    {
      ImGui::Begin("postprocess_pass");
      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("feedback_pass/rt0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
        auto  wpos        = ImGui::GetCursorScreenPos();
        float height_diff = 24;
        if (wsize.y < height_diff + 2) {
          wsize.y = 2;
        } else {
          wsize.y = wsize.y - height_diff;
        }
        ImGuiIO &io = ImGui::GetIO();
        // g_config.get_u32("g_buffer_width")  = wsize.x;
        // g_config.get_u32("g_buffer_height") = wsize.y;
        if (ImGui::IsWindowHovered()) {
          auto scroll_y = ImGui::GetIO().MouseWheel;
          if (scroll_y) {
            g_camera.distance += g_camera.distance * 2.e-1 * scroll_y;
            g_camera.distance = clamp(g_camera.distance, 1.0e-3f, 1000.0f);
          }
          f32 camera_speed = 2.0f;
          if (ImGui::GetIO().KeysDown[SDL_SCANCODE_LSHIFT]) {
            camera_speed = 20.0f;
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
            g_camera.look_at +=
                glm::normalize(camera_diff) * camera_speed * (float)timer.dt;
          }
          ImVec2 mpos    = ImGui::GetMousePos();
          i32    cur_m_x = mpos.x;
          i32    cur_m_y = mpos.y;
          if (io.MouseDown[0] && last_m_x > 0) {
            i32 dx = cur_m_x - last_m_x;
            i32 dy = cur_m_y - last_m_y;
            g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
            g_camera.theta += (float)(dy)*5.0e-3f;
          }
          last_m_x = cur_m_x;
          last_m_y = cur_m_y;
        }
        ImGui::End();
      }
      ImGui::Begin("uv feedback");

      {
        auto        wsize = ImGui::GetWindowSize();
        Resource_ID img   = pc->get_resource(stref_s("feedback_pass/img0"));
        ImGui::Image((ImTextureID)(intptr_t)img.data, ImVec2(wsize.x, wsize.y));
      }

      ImGui::End();
    }
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;
  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  GUI_Pass *    gui  = new GUI_Pass;
  pmng->set_event_consumer(gui);
  pmng->add_pass(rd::Pass_t::RENDER, new Feedback_Pass);
  pmng->add_pass(rd::Pass_t::RENDER, gui);
  pmng->loop();
  return 0;
}
