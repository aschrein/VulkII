#ifndef RENDERING_HPP
#define RENDERING_HPP

#include "utils.hpp"

struct ID {
  u32  _id;
  u32  index() { return _id - 1; }
  bool is_null() { return _id == 0; }
  bool operator==(ID const &that) const { return _id == that._id; }
  bool operator!=(ID const &that) const { return _id != that._id; }
};

static_assert(sizeof(ID) == 4, "blimey!");

struct Resource_ID {
  union {
    struct {
      ID  id;
      u32 type;
    };
    u64 data;
  };
  static Resource_ID null() { return {0, 0}; }
  u32                index() { return id.index(); }
  bool               is_null() { return id.is_null(); }
  void               reset() { memset(this, 0, sizeof(*this)); }
};

static_assert(sizeof(Resource_ID) == 8, "blimey!");

static inline u64 hash_of(ID id) { return hash_of(id._id); }

static inline u64 hash_of(Resource_ID res) {
  return hash_of(res.id._id) ^ hash_of(res.type);
}

namespace rd {
enum class Impl_t { VULKAN, Null };
enum class Cmp { LT, LE, GT, GE, EQ };
enum class Primitive { TRIANGLE_LIST, LINE_LIST };
enum class Front_Face { CW, CCW };
enum class Cull_Mode { NONE, FRONT, BACK };
enum class Index_t { UINT32, UINT16 };
enum class Format : u32 {
  UNKNOWN = 0,
  NATIVE  = 1,
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
  R32_UINT,
  R16_UINT,
  D32_FLOAT,
};

static inline char const *format_to_cstr(Format format) {
  switch (format) {
    // clang-format off
      case Format::UNKNOWN     : return "UNKNOWN";
      case Format::NATIVE      : return "NATIVE";
      case Format::BGRA8_UNORM : return "BGRA8_UNORM";
      case Format::BGR8_UNORM  : return "BGR8_UNORM";
      case Format::RGBA8_UNORM : return "RGBA8_UNORM";
      case Format::RGBA8_SNORM : return "RGBA8_SNORM";
      case Format::RGBA8_SRGBA : return "RGBA8_SRGBA";
      case Format::RGBA8_UINT  : return "RGBA8_UINT";
      case Format::RGB8_UNORM  : return "RGB8_UNORM";
      case Format::RGB8_SNORM  : return "RGB8_SNORM";
      case Format::RGB8_SRGBA  : return "RGB8_SRGBA";
      case Format::RGB8_UINT   : return "RGB8_UINT";
      case Format::RGBA32_FLOAT: return "RGBA32_FLOAT";
      case Format::RGB32_FLOAT : return "RGB32_FLOAT";
      case Format::RG32_FLOAT  : return "RG32_FLOAT";
      case Format::R32_FLOAT   : return "R32_FLOAT";
      case Format::R32_UINT    : return "R32_UINT";
      case Format::R16_UINT    : return "R16_UINT";
      case Format::D32_FLOAT   : return "D32_FLOAT";
  // clang-format on
  default: TRAP;
  }
}

enum class Buffer_Usage_Bits : uint32_t {
  USAGE_VERTEX_BUFFER      = 1,
  USAGE_UNIFORM_BUFFER     = 2,
  USAGE_INDEX_BUFFER       = 4,
  USAGE_UAV                = 8,
  USAGE_TRANSFER_DST       = 16,
  USAGE_TRANSFER_SRC       = 32,
  USAGE_INDIRECT_ARGUMENTS = 64,
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

struct Image_Info {
  Format format;
  bool   is_depth;
  u32    width, height, depth, levels, layers;
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

enum class Stage_t { UNKNOWN = 0, VERTEX, PIXEL, COMPUTE, BINDING };

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
  TEXCOORD4,
  TEXCOORD5,
  TEXCOORD6,
  TEXCOORD7,
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

enum class Access_Bits : u32 {
  SHADER_READ  = 0b00001,
  SHADER_WRITE = 0b00010,
  MEMORY_WRITE = 0b00100,
  MEMORY_READ  = 0b01000,
};

enum class Image_Layout {
  SHADER_READ_WRITE_OPTIMAL,
  SHADER_READ_ONLY_OPTIMAL,
  TRANSFER_DST_OPTIMAL,
  TRANSFER_SRC_OPTIMAL,
};

struct Clear_Value {
  union {
    struct {
      f32 v_f32[4];
    };
    struct {
      u32 v_u32[4];
    };
    struct {
      i32 v_i32[4];
    };
  };
};

struct Image_Subresource {
  u32                      layer;
  u32                      num_layers;
  u32                      level;
  u32                      num_levels;
  static Image_Subresource top_level() {
    Image_Subresource out;
    out.layer      = 0;
    out.num_layers = 1;
    out.level      = 0;
    out.num_levels = 1;
    return out;
  }
};

struct Image_Copy {
  u32               layer;
  u32               num_layers;
  u32               level;
  u32               offset_x;
  u32               offset_y;
  u32               offset_z;
  u32               size_x;
  u32               size_y;
  u32               size_z;
  static Image_Copy top_level() {
    Image_Copy out;
    MEMZERO(out);
    out.layer      = 0;
    out.num_layers = 1;
    out.level      = 0;
    return out;
  }
};

class IFactory {
  public:
  virtual Resource_ID  create_image(Image_Create_Info info)             = 0;
  virtual Resource_ID  create_buffer(Buffer_Create_Info info)           = 0;
  virtual Resource_ID  create_shader_raw(Stage_t type, string_ref text,
                                         Pair<string_ref, string_ref> *defines,
                                         size_t num_defines)            = 0;
  virtual Resource_ID  create_sampler(Sampler_Create_Info const &info)  = 0;
  virtual void         release_resource(Resource_ID id)                 = 0;
  virtual Resource_ID  get_resource(string_ref res_name)                = 0;
  virtual void         assign_name(Resource_ID res_id, string_ref name) = 0;
  virtual Resource_ID  get_swapchain_image()                            = 0;
  virtual Image2D_Info get_swapchain_image_info()                       = 0;
  virtual Image_Info   get_image_info(Resource_ID res_id)               = 0;
  virtual void *       map_buffer(Resource_ID id)                       = 0;
  virtual void         unmap_buffer(Resource_ID id)                     = 0;
};

class Imm_Ctx : public IFactory {
  public:
  ////////////////////////////
  // Immediate mode context //
  ////////////////////////////
  virtual void clear_state()                                           = 0;
  virtual void clear_bindings()                                        = 0;
  virtual void push_state()                                            = 0;
  virtual void pop_state()                                             = 0;
  virtual void image_barrier(Resource_ID image_id, u32 access_flags,
                             Image_Layout layout)                      = 0;
  virtual void buffer_barrier(Resource_ID buf_id, u32 access_flags)    = 0;
  virtual bool get_fence_state(Resource_ID fence_id)                   = 0;
  virtual void RS_set_line_width(float width)                          = 0;
  virtual void RS_set_depth_bias(float b)                              = 0;
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
  virtual void fill_buffer(Resource_ID id, size_t offset, size_t size,
                           u32 value)                                  = 0;
  virtual void clear_image(Resource_ID id, Image_Subresource const &range,
                           Clear_Value const &cv)                      = 0;
  virtual void OM_set_blend_state(u32 rt_index, Blend_State const &bl) = 0;

  virtual void bind_uniform_buffer(u32 set, u32 binding, Resource_ID buf_id,
                                   size_t offset, size_t size)              = 0;
  virtual void bind_sampler(u32 set, u32 binding, Resource_ID sampler_id)   = 0;
  virtual void bind_storage_buffer(u32 set, u32 binding, Resource_ID buf_id,
                                   size_t offset, size_t size)              = 0;
  virtual void bind_image(u32 set, u32 binding, u32 index, Resource_ID image_id,
                          Image_Subresource const &range, Format format)    = 0;
  virtual void bind_rw_image(u32 set, u32 binding, u32 index,
                             Resource_ID              image_id,
                             Image_Subresource const &range, Format format) = 0;
  virtual void push_constants(void const *data, size_t offset, size_t size) = 0;
  virtual void draw_indexed(u32 indices, u32 instances, u32 first_index,
                            u32 first_instance, i32 vertex_offset)          = 0;
  virtual void draw(u32 vertices, u32 instances, u32 first_vertex,
                    u32 first_instance)                                     = 0;
  virtual void multi_draw_indexed_indirect(Resource_ID arg_buf_id,
                                           u32         arg_buf_offset,
                                           Resource_ID cnt_buf_id,
                                           u32 cnt_buf_offset, u32 max_count,
                                           u32 stride)                      = 0;
  virtual void dispatch(u32 dim_x, u32 dim_y, u32 dim_z)                    = 0;
  virtual void set_viewport(float x, float y, float width, float height,
                            float mindepth, float maxdepth)                 = 0;
  virtual void set_scissor(u32 x, u32 y, u32 width, u32 height)             = 0;
  virtual void copy_buffer_to_image(Resource_ID buf_id, size_t buffer_offset,
                                    Resource_ID       img_id,
                                    Image_Copy const &dst_info)             = 0;
  virtual void copy_image_to_buffer(Resource_ID buf_id, size_t buffer_offset,
                                    Resource_ID       img_id,
                                    Image_Copy const &dst_info)             = 0;
  virtual void copy_buffer(Resource_ID src_buf_id, size_t src_offset,
                           Resource_ID dst_buf_id, size_t dst_offset,
                           u32 size)                                        = 0;
  virtual Image_Info get_image_info(Resource_ID res_id)                     = 0;
};

struct Clear_Color {
  float r, g, b, a;
  bool  clear;
};

struct Clear_Depth {
  float d;
  bool  clear;
};

enum class Fence_Position { PASS_FINISED };

class IPass;

class IPass_Context : public IFactory {
  public:
  static constexpr u32 BUFFER_ALIGNMENT = 0x100;
  static size_t        align_up(size_t size) {
    return (size + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
  }
  static size_t align_down(size_t size) {
    return (size) & ~(BUFFER_ALIGNMENT - 1);
  }
  virtual void add_render_target(string_ref name, Image_Create_Info const &info,
                                 u32 layer, u32 level,
                                 Clear_Color const &cl) = 0;
  virtual void add_render_target(Resource_ID id, u32 layer, u32 level,
                                 Clear_Color const &cl) = 0;
  virtual void add_depth_target(string_ref name, Image_Create_Info const &info,
                                u32 layer, u32 level,
                                Clear_Depth const &cl)  = 0;
  virtual void add_depth_target(Resource_ID id, u32 layer, u32 level,
                                Clear_Depth const &cl)  = 0;

  virtual Resource_ID get_fence(Fence_Position position) = 0;
  virtual void *      get_window_handle()                = 0;
  virtual IPass *     get_pass(string_ref name)          = 0;
  virtual double      get_pass_duration(string_ref name) = 0;
};

enum class Pass_t { COMPUTE, RENDER };

class IPass {
  public:
  virtual void       on_begin(IPass_Context *rm) = 0;
  virtual void       exec(rd::Imm_Ctx *ctx)      = 0;
  virtual void       on_end(IPass_Context *rm)   = 0;
  virtual void       release(IPass_Context *rm)  = 0;
  virtual string_ref get_name()                  = 0;
};

class IEvent_Consumer {
  public:
  virtual void consume(void *event) = 0;
};

struct Pass_Mng {
  static Pass_Mng *create(Impl_t type);
  virtual void     loop()                                        = 0;
  virtual void     add_pass(Pass_t type, IPass *pass)            = 0;
  virtual void     set_event_consumer(IEvent_Consumer *consumer) = 0;
  virtual void     release()                                     = 0;
};

} // namespace rd

#endif // RENDERING_HPP
