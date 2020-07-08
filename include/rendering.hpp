#ifndef RENDERING_HPP
#define RENDERING_HPP

#include "utils.hpp"

struct ID {
  u32  _id = 0;
  u32  index() { return _id - 1; }
  bool is_null() { return _id == 0; }
  bool operator==(ID const &that) const { return _id == that._id; }
  bool operator!=(ID const &that) const { return _id != that._id; }
};

static_assert(sizeof(ID) == 4, "blimey!");

struct Resource_ID {
  ID   id;
  u32  type;
  u32  index() { return id.index(); }
  bool is_null() { return id.is_null(); }
  void reset() { memset(this, 0, sizeof(*this)); }
};

static_assert(sizeof(Resource_ID) == 8, "blimey!");

namespace rd {
enum class Impl_t { VULKAN, Null };
enum class Cmp { LT, LE, GT, GE, EQ };
enum class Primitive { TRIANGLE_LIST, LINE_LIST };
enum class Front_Face { CW, CCW };
enum class Cull_Mode { NONE, FRONT, BACK };
enum class Index_t { UINT32, UINT16 };
enum class Format {
  UNKNOWN = 0,
  BGRA8_UNORM,
  BGR8_UNORM,
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
  D32_FLOAT,
};

enum class Buffer_Usage_Bits : uint32_t {
  USAGE_VERTEX_BUFFER  = 1,
  USAGE_UNIFORM_BUFFER = 2,
  USAGE_INDEX_BUFFER   = 4,
  USAGE_UAV            = 8,
  USAGE_TRANSFER_DST   = 16,
  USAGE_TRANSFER_SRC   = 32,
};

enum class Image_Usage_Bits : uint32_t {
  USAGE_SAMPLED      = 1,
  USAGE_UAV          = 2,
  USAGE_RT           = 4,
  USAGE_DT           = 8,
  USAGE_TRANSFER_DST = 16,
  USAGE_TRANSFER_SRC = 32,
};

enum class Memory_Bits : uint32_t {
  HOST_VISIBLE = 1,
  HOST_CACHED  = 2,
  DEVICE_LOCAL = 4,
  COHERENT     = 8,
};

struct Image_Create_Info {
  Format format;
  u32    usage_bits;
  u32    mem_bits;
  u32    width, height, depth, levels, layers;
};

struct Buffer_Create_Info {
  u32 usage_bits;
  u32 mem_bits;
  u32 size;
};

struct Binding {
  string_ref name;
  u32        slot;
};

struct Image2D_Info {
  Format format;
  u32    width, height, levels, layers;
};

enum class Filter {
  NEAREST = 0,
  LINEAR  = 1,
};

enum class Address_Mode {
  REPEAT = 0,
  MIRRORED_REPEAT,
  CLAMP_TO_EDGE,
};

struct Sampler_Create_Info {
  Filter       mag_filter;
  Filter       min_filter;
  Filter       mip_mode;
  Address_Mode address_mode_u;
  Address_Mode address_mode_v;
  Address_Mode address_mode_w;
  float        mip_lod_bias;
  bool         anisotropy;
  float        max_anisotropy;
  bool         cmp;
  Cmp          cmp_op;
  float        min_lod;
  float        max_lod;
  bool         unnormalized_coordiantes;
};

enum class Polygon_Mode {
  FILL,
  LINE,
};

struct RS_State {
  Polygon_Mode polygon_mode;
  Front_Face   front_face;
  Cull_Mode    cull_mode;
  float        line_width;
  float        depth_bias;
};

struct DS_State {
  bool  enable_depth_test;
  Cmp   cmp_op;
  bool  enable_depth_write;
  float max_depth;
};

enum class Blend_Factor {
  ZERO                     = 0,
  ONE                      = 1,
  SRC_COLOR                = 2,
  ONE_MINUS_SRC_COLOR      = 3,
  DST_COLOR                = 4,
  ONE_MINUS_DST_COLOR      = 5,
  SRC_ALPHA                = 6,
  ONE_MINUS_SRC_ALPHA      = 7,
  DST_ALPHA                = 8,
  ONE_MINUS_DST_ALPHA      = 9,
  CONSTANT_COLOR           = 10,
  ONE_MINUS_CONSTANT_COLOR = 11,
  CONSTANT_ALPHA           = 12,
  ONE_MINUS_CONSTANT_ALPHA = 13,
};

enum class Blend_OP {
  ADD              = 0,
  SUBTRACT         = 1,
  REVERSE_SUBTRACT = 2,
  MIN              = 3,
  MAX              = 4,
};

enum class Color_Component_Bit { R_BIT = 1, G_BIT = 2, B_BIT = 4, A_BIT = 8 };

struct Blend_State {
  bool         enabled;
  Blend_OP     color_blend_op;
  Blend_OP     alpha_blend_op;
  Blend_Factor src_color;
  Blend_Factor dst_color;
  Blend_Factor src_alpha;
  Blend_Factor dst_alpha;
  u32          color_write_mask;
};

enum class Stage_t { UNKNOWN = 0, VERTEX, PIXEL, COMPUTE };

enum class Attriute_t {
  UNKNOWN = 0,
  POSITION,
  NORMAL,
  BINORMAL,
  TANGENT,
  TEXCOORD0,
  TEXCOORD1,
  TEXCOORD2,
  TEXCOORD3,
};

struct Attribute_Info {
  Attriute_t type;
  u32        location;
  u32        binding;
  Format     format;
  size_t     offset;
};

struct MS_State {
  u32   num_samples;
  bool  sample_shading;
  float min_sample_shading;
  u32   sample_mask;
  bool  alpha_to_coverage;
  bool  alpha_to_one;
};

enum class Input_Rate { VERTEX, INSTANCE };

class Imm_Ctx {
  public:
  ////////////////////////////
  // Immediate mode context //
  ////////////////////////////
  virtual void clear_state()                                           = 0;
  virtual void push_state()                                            = 0;
  virtual void pop_state()                                             = 0;
  virtual void IA_set_topology(Primitive topology)                     = 0;
  virtual void IA_set_index_buffer(Resource_ID id, u32 offset,
                                   Index_t format)                     = 0;
  virtual void IA_set_vertex_buffer(u32 index, Resource_ID buffer,
                                    size_t offset, size_t stride,
                                    Input_Rate rate)                   = 0;
  virtual void IA_set_attribute(Attribute_Info const &info)            = 0;
  virtual void VS_set_shader(Resource_ID id)                           = 0;
  virtual void PS_set_shader(Resource_ID id)                           = 0;
  virtual void CS_set_shader(Resource_ID id)                           = 0;
  virtual void RS_set_state(RS_State const &rs_state)                  = 0;
  virtual void DS_set_state(DS_State const &ds_state)                  = 0;
  virtual void MS_set_state(MS_State const &ds_state)                  = 0;
  virtual void OM_set_blend_state(u32 rt_index, Blend_State const &bl) = 0;

  virtual void  bind_uniform_buffer(Stage_t stage, u32 set, u32 binding,
                                    Resource_ID buf_id, size_t offset,
                                    size_t size)                       = 0;
  virtual void  bind_sampler(Stage_t stage, u32 set, u32 binding,
                             Resource_ID sampler_id)                   = 0;
  virtual void  bind_storage_buffer(Stage_t stage, u32 set, u32 binding,
                                    Resource_ID buf_id, size_t offset) = 0;
  virtual void  bind_image(Stage_t stage, u32 set, u32 binding, u32 index,
                           Resource_ID image_id, u32 layer, u32 num_layers,
                           u32 level, u32 num_levels)                  = 0;
  virtual void  bind_rw_image(Stage_t stage, u32 set, u32 binding, u32 index,
                              Resource_ID image_id, u32 layer, u32 num_layers,
                              u32 level, u32 num_levels)               = 0;
  virtual void  flush_bindings()                                       = 0;
  virtual void *map_buffer(Resource_ID id)                             = 0;
  virtual void  unmap_buffer(Resource_ID id)                           = 0;
  virtual void  push_constants(void const *data, size_t size)          = 0;
  virtual void  draw_indexed(u32 indices, u32 instances, u32 first_index,
                             u32 first_instance, i32 vertex_offset)    = 0;
  virtual void  draw(u32 vertices, u32 instances, u32 first_vertex,
                     u32 first_instance)                               = 0;
  virtual void  dispatch(u32 dim_x, u32 dim_y, u32 dim_z)              = 0;
  virtual void  set_viewport(float x, float y, float width, float height,
                             float mindepth, float maxdepth)           = 0;
  virtual void  set_scissor(u32 x, u32 y, u32 width, u32 height)       = 0;
};

struct Clear_Color {
  float r, g, b, a;
  bool  clear;
};

struct Clear_Depth {
  float d;
  bool  clear;
};

class IResource_Manager {
  public:
  virtual Resource_ID create_image(Image_Create_Info info)              = 0;
  virtual Resource_ID create_buffer(Buffer_Create_Info info)            = 0;
  virtual Resource_ID create_shader_raw(Stage_t type, string_ref text,
                                        Pair<string_ref, string_ref> *defines,
                                        size_t num_defines)             = 0;
  virtual Resource_ID create_sampler(Sampler_Create_Info const &info)   = 0;
  virtual void        release_resource(Resource_ID id)                  = 0;
  virtual void add_render_target(string_ref name, Image_Create_Info const &info,
                                 u32 layer, u32 level,
                                 Clear_Color const &cl)                 = 0;
  virtual void add_render_target(Resource_ID id, u32 layer, u32 level,
                                 Clear_Color const &cl)                 = 0;
  virtual void add_depth_target(string_ref name, Image_Create_Info const &info,
                                u32 layer, u32 level,
                                Clear_Depth const &cl)                  = 0;
  virtual void add_depth_target(Resource_ID id, u32 layer, u32 level,
                                Clear_Depth const &cl)                  = 0;
  virtual Resource_ID  get_resource(string_ref res_name)                = 0;
  virtual void         assign_name(Resource_ID res_id, string_ref name) = 0;
  virtual Resource_ID  get_swapchain_image()                            = 0;
  virtual Image2D_Info get_swapchain_image_info()                       = 0;
};

class IPass {
  public:
  virtual void       on_begin(IResource_Manager *rm) = 0;
  virtual void       exec(rd::Imm_Ctx *ctx)          = 0;
  virtual void       on_end(IResource_Manager *rm)   = 0;
  virtual void       release(IResource_Manager *rm)  = 0;
  virtual string_ref get_name()                      = 0;
};

struct Pass_Mng {
  static Pass_Mng *create(Impl_t type);
  virtual void     loop()                = 0;
  virtual void     add_pass(IPass *pass) = 0;
  virtual void     release()             = 0;
};

} // namespace rd

#endif // RENDERING_HPP
