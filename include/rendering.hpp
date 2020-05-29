#ifndef RENDERING_HPP
#define RENDERING_HPP

#include "script.hpp"
#include "utils.hpp"

struct ID {
  u32  _id = 0;
  u32  index() { return _id - 1; }
  bool is_null() { return _id == 0; }
};

struct Resource_ID {
  ID   id;
  u32  type;
  u32  index() { return id.index(); }
  bool is_null() { return id.is_null(); }
};

namespace rd {
enum class RT_t { Color, Depth };
enum class Type { RT, Image, Buffer, Dummy };
enum class Cmp_t { LT, LE, GT, GE, EQ };
enum class Primitive_t { TRIANGLE_LIST, LINE_LIST };
enum class Front_Face { CW, CCW };
enum class Cull_Mode { NONE, FRONT, BACK };
enum class Index_t { UINT32, UINT16 };
enum class Format {
  UNKNOWN = 0,
  RGBA8_UNORM,
  RGBA8_SNORM,
  RGBA8_SRGBA,
  RGBA8_UINT,
  RGB8_UNORM,
  RGB8_SNORM,
  RGB8_SRGBA,
  RGB8_UINT,
  RGBA32_FLOAT,
  RGB32_FLOAT,
  RG32_FLOAT,
  R32_FLOAT,
};

enum class Buffer_Usage_Bits : uint32_t {
  USAGE_VERTEX_BUFFER  = 1,
  USAGE_UNIFORM_BUFFER = 2,
  USAGE_INDEX_BUFFER   = 4,
  USAGE_UAV            = 8,
  USAGE_TRANSIENT      = 16,
};

enum class Image_Usage_Bits : uint32_t {
  USAGE_SAMPLED = 1,
  USAGE_UAV     = 2,
};

enum class Memory_Bits : uint32_t {
  MAPPABLE = 1,
  DEVICE   = 2,
};

struct RT {
  Format format;
  RT_t   type;
};

struct Image {
  Format format;
  u32    usage_bits;
  u32    mem_bits;
  u32    width, height, depth, levels, layers;
};

struct Buffer {
  u32 usage_bits;
  u32 mem_bits;
  u32 size;
};

struct Resource {
  string_ref name;
  Type       type;
  union {
    Buffer buffer_info;
    Image  image_info;
    RT     rt_info;
  };
};

struct Buffer_Ref {
  ID  buf_id;
  u32 offset;
};

struct Image_View {
  u32 base_level;
  u32 levels;
  u32 base_layer;
  u32 layers;
};

struct Binding {
  string_ref name;
  u32        slot;
};

struct Texture2D_Create_Info {
  void * data;
  Format format;
  u32    width;
  u32    height;
};

// struct Imm_Ctx;
// typedef void (*Pass_Callback_t)(Imm_Ctx *, void *);

// struct Imm_Ctx {
//  void *pImpl;

//  Resource_ID create_texture2D(Texture2D_Create_Info info, bool build_mip = true);
//  Resource_ID create_uav_image(u32 width, u32 height, Format format, u32 levels, u32 layers);
//  Resource_ID create_buffer(Buffer info, void const *initial_data = nullptr);
//  Resource_ID create_vertex_shader(List *shader_node);
//  Resource_ID create_pixel_shader(List *shader_node);
//  void        release_resource(Resource_ID id);
//  ////////////////////////////
//  // Immediate mode context //
//  ////////////////////////////
//  void IA_set_topology(Primitive_t topology);
//  void IA_set_index_buffer(Resource_ID id, u32 offset, Index_t format);
//  void IA_set_vertex_buffers(Buffer_Ref *infos, u32 num_buffers, u32 start_id);
//  void IA_set_cull_mode(Front_Face front_face, Cull_Mode cull_mode);

//  void VS_set_shader(Resource_ID id);
//  void PS_set_shader(Resource_ID id);
//  void CS_set_shader(Resource_ID id);
//  void RS_set_depth_stencil_state(bool enable_depth_test, Cmp_t cmp_op, bool enable_depth_write,
//                                  float max_depth, float depth_bias = 0.0f);
//  void RS_set_line_width(float line_width);
//  void bind_resource(string_ref name, u32 id, u32 index);
//  void bind_resource(string_ref name, string_ref id, u32 index);
//  void bind_image(string_ref name, string_ref res_name, u32 index, Image_View view);

//  void *map_buffer(Resource_ID id);
//  void  unmap_buffer(Resource_ID id);
//  void  push_constants(void *data, size_t size);

//  void clear_color(float r, float g, float b, float a);
//  void clear_depth(float value);

//  void draw_indexed(u32 indices, u32 instances, u32 first_index, u32 first_instance,
//                    i32 vertex_offset);
//  void draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance);
//  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z);
//};

// struct Pass_Mng {
//  void *           pImpl;
//  static Pass_Mng *get();

//  Resource_ID create_render_pass(string_ref name, void *payload, string_ref *dep_names,
//                                 u32 num_deps, Resource *products, u32 num_products, u32 width,
//                                 u32 height, Pass_Callback_t callback);
//  Resource_ID create_compute_pass(string_ref name, void *payload, string_ref *dep_names,
//                                  u32 num_deps, Resource *products, u32 num_products,
//                                  Pass_Callback_t callback);
//};

} // namespace rd

#endif // RENDERING_HPP
