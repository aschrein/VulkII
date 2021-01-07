#ifndef RENDERING_HPP
#define RENDERING_HPP

#include "utils.hpp"

struct ID {
  u32  _id;
  u32  index() const { return _id - 1; }
  bool is_null() const { return _id == 0; }
  bool is_valid() const { return _id != 0; }
  bool operator==(ID const &that) const { return _id == that._id; }
  bool operator!=(ID const &that) const { return _id != that._id; }
};

static_assert(sizeof(ID) == 4, "blimey!");

struct Ptr2 {
  void *ptr1;
  void *ptr2;
};

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
  bool               is_null() const { return id.is_null(); }
  bool               is_valid() const { return type != 0 && !id.is_null(); }
  void               reset() { memset(this, 0, sizeof(*this)); }
  bool               operator==(Resource_ID const &that) const { return data == that.data; }
};
ASSERT_ISPOD(Resource_ID);
static_assert(sizeof(Resource_ID) == 8, "blimey!");

static inline u64 hash_of(ID id) { return hash_of(id._id); }

static inline u64 hash_of(Resource_ID res) { return hash_of(res.data); }

namespace rd {
enum class Impl_t { VULKAN, DX12, Null };
enum class Cmp { LT, LE, GT, GE, EQ };
enum class Primitive { TRIANGLE_LIST, TRIANGLE_STRIP, LINE_LIST };
enum class Front_Face { CW, CCW };
enum class Cull_Mode { NONE, FRONT, BACK };
enum class Index_t { UINT32, UINT16 };
enum class Format : u32 {
  UNKNOWN = 0,
  NATIVE  = 1,
  BGRA8_UNORM,
  BGRA8_SRGBA,
  BGR8_UNORM,
  RGBA8_UNORM,
  RGBA8_SNORM,
  RGBA8_SRGBA,
  RGBA8_UINT,
  RGBA32_FLOAT,
  RGB32_FLOAT,
  RG32_FLOAT,
  R32_FLOAT,
  R32_UINT,
  R16_FLOAT,
  R16_UNORM,
  R8_UNORM,
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

static inline bool is_depth_format(Format format) {
  switch (format) {
  case Format::D32_FLOAT: return true;
  default: return false;
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

enum class Memory_Type : uint32_t {
  CPU_WRITE_GPU_READ = 1,
  CPU_READ_WRITE     = 2,
  GPU_LOCAL          = 4,
};

struct Image_Create_Info {
  Format format;
  u32    usage_bits;
  u32    width, height, depth, levels, layers;
};

struct Buffer_Create_Info {
  u32         usage_bits;
  Memory_Type memory_type;
  u32         size;
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

enum class Buffer_Access {
  GENERIC,
  UNIFORM,
  VERTEX_BUFFER,
  INDEX_BUFFER,
  UAV,
  TRANSFER_DST,
  TRANSFER_SRC,
  HOST_READ,
  HOST_WRITE,
  HOST_READ_WRITE,
};

enum class Image_Access {
  GENERIC,
  COLOR_TARGET,
  DEPTH_TARGET,
  UAV,
  SAMPLED,
  TRANSFER_DST,
  TRANSFER_SRC,
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
  i32                      layer;
  i32                      num_layers;
  i32                      level;
  i32                      num_levels;
  static Image_Subresource top_level() {
    Image_Subresource out;
    out.layer      = 0;
    out.num_layers = 1;
    out.level      = 0;
    out.num_levels = 1;
    return out;
  }
  static Image_Subresource all_levels() {
    Image_Subresource out;
    out.layer      = 0;
    out.num_layers = -1;
    out.level      = 0;
    out.num_levels = -1;
    return out;
  }
};
ASSERT_ISPOD(Image_Subresource);
struct Image_Copy {
  u32 buffer_row_pitch;
  // u32               buffer_slice_pitch;
  u32               layer;
  u32               level;
  u32               offset_x;
  u32               offset_y;
  u32               offset_z;
  u32               size_x;
  u32               size_y;
  u32               size_z;
  static Image_Copy top_level(u32 pitch = 0) {
    Image_Copy out;
    MEMZERO(out);
    out.buffer_row_pitch = pitch;
    out.layer            = 0;
    out.level            = 0;
    return out;
  }
};

ASSERT_ISPOD(Image_Copy);
class ICtx;

struct Clear_Color {
  float r, g, b, a;
  bool  clear;
};

struct Clear_Depth {
  float d;
  bool  clear;
};

enum class Fence_Position { PASS_FINISED };

enum class Pass_t { COMPUTE, RENDER };

struct RT_Ref {
  bool   enabled;
  Format format;
  union {
    Clear_Color clear_color;
    Clear_Depth clear_depth;
  };
  void reset() { MEMZERO(*this); }
};

ASSERT_ISPOD(RT_Ref);

struct Render_Pass_Create_Info {
  InlineArray<RT_Ref, 0x10> rts;
  RT_Ref                    depth_target;
  void                      reset() { MEMZERO(*this); }
};

static_assert(std::is_pod<Render_Pass_Create_Info>::value, "");

struct RT_View {
  bool        enabled;
  Resource_ID image;
  Format      format;
  u32         layer;
  u32         level;
  void        reset() { MEMZERO(*this); }
};

ASSERT_ISPOD(RT_View);

struct Frame_Buffer_Create_Info {
  InlineArray<RT_View, 0x10> rts;
  RT_View                    depth_target;
  void                       reset() { MEMZERO(*this); }
};

static_assert(std::is_pod<Frame_Buffer_Create_Info>::value, "");

enum class Binding_t : u32 { //
  SAMPLER,
  UNIFORM_BUFFER,
  READ_ONLY_BUFFER,
  UAV_BUFFER,
  TEXTURE,
  UAV_TEXTURE
};

struct Binding_Desc {
  Binding_t type;
  u32       num_array_elems;
};

struct Binding_Space_Create_Info {
  static constexpr u32                    MAX_BINDINGS = 0x40;
  InlineArray<Binding_Desc, MAX_BINDINGS> bindings;
  void                                    reset() { MEMZERO(*this); }
};

static_assert(std::is_pod<Binding_Space_Create_Info>::value, "");

struct Binding_Table_Create_Info {
  static constexpr u32                               MAX_SPACES = 0x10;
  InlineArray<Binding_Space_Create_Info, MAX_SPACES> spaces;
  u32                                                push_constants_size;
  void                                               reset() { MEMZERO(*this); }
};

static inline u64 hash_of(RT_View const &pc) {
  return hash_of(string_ref{(char *)&pc, sizeof(pc)});
}

static inline bool operator==(RT_View const &a, RT_View const &b) {
  return memcmp(&a, &b, sizeof(a)) == 0;
}

static inline u64 hash_of(Render_Pass_Create_Info const &pc) {
  return hash_of(string_ref{(char *)&pc, sizeof(pc)});
}

static inline bool operator==(Render_Pass_Create_Info const &a, Render_Pass_Create_Info const &b) {
  return memcmp(&a, &b, sizeof(a)) == 0;
}

struct Vertex_Binding {
  u32        binding;
  u32        stride;
  Input_Rate inputRate;
};

static_assert(std::is_pod<Vertex_Binding>::value, "");

struct Graphics_Pipeline_State {
  Vertex_Binding bindings[0x10];
  Attribute_Info attributes[0x10];
  u32            num_vs_bindings;
  u32            num_attributes;
  Primitive      topology;
  RS_State       rs_state;
  DS_State       ds_state;
  Resource_ID    ps;
  Resource_ID    vs;
  u32            num_rts;
  Blend_State    blend_states[8];
  MS_State       ms_state;

  // Utility functions.
  void IA_set_topology(Primitive topology) { this->topology = topology; }
  void IA_set_vertex_binding(u32 index, u32 stride, Input_Rate rate) {
    num_vs_bindings           = MAX(num_vs_bindings, index + 1);
    bindings[index].binding   = index;
    bindings[index].inputRate = rate;
    bindings[index].stride    = stride;
  }
  void IA_set_attribute(Attribute_Info const &info) {
    num_attributes            = MAX(num_attributes, info.location + 1);
    attributes[info.location] = info;
  }
  void VS_set_shader(Resource_ID id) { vs = id; }
  void PS_set_shader(Resource_ID id) { ps = id; }
  void RS_set_state(RS_State const &rs_state) { this->rs_state = rs_state; }
  void DS_set_state(DS_State const &ds_state) { this->ds_state = ds_state; }
  void MS_set_state(MS_State const &ms_state) { this->ms_state = ms_state; }
  void OM_set_blend_state(u32 rt_index, Blend_State const &bl) {
    num_rts                      = MAX(num_rts, rt_index + 1);
    this->blend_states[rt_index] = bl;
  }

  bool operator==(const Graphics_Pipeline_State &that) const {
    return memcmp(this, &that, sizeof(*this)) == 0;
  }
  void reset() {
    memset(this, 0, sizeof(*this)); // Important for memhash
  }
};

static_assert(std::is_pod<Graphics_Pipeline_State>::value, "");

static inline u64 hash_of(Graphics_Pipeline_State const &state) {
  return hash_of(string_ref{(char const *)&state, sizeof(state)});
}

// Single threaded entity to manage descriptor binding. Semi-lightweight, not much allocations going
// on.
class IBinding_Table {
  public:
  // size=0 means whole size.
  virtual void bind_cbuffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                            size_t size)                                    = 0;
  virtual void bind_sampler(u32 space, u32 binding, Resource_ID sampler_id) = 0;
  // ByteAddressBuffer. size=0 means whole size.
  virtual void bind_UAV_buffer(u32 space, u32 binding, Resource_ID buf_id, size_t offset,
                               size_t size)                                    = 0;
  virtual void bind_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                            Image_Subresource const &range, Format format)     = 0;
  virtual void bind_UAV_texture(u32 space, u32 binding, u32 index, Resource_ID image_id,
                                Image_Subresource const &range, Format format) = 0;
  virtual void push_constants(void const *data, size_t offset, size_t size)    = 0;
  // The pointer for this object is invalid after this call
  virtual void release() = 0;
};
// Multi threaded entity, accesses are wrapped in synchronization primitives internally. Used to
// create resources. start_frame/end_frame needed to switch between swap chain images and also to
// control deferred resource release.
class IDevice {
  public:
  // Uniform buffer offset alignment
  static constexpr u32 BUFFER_ALIGNMENT      = 0x100;
  static constexpr u32 BUFFER_COPY_ALIGNMENT = 512;
  // For copy_buffer_to_image and copy_image_to_buffer the pitch must be aligned up to this value.
  static constexpr u32 TEXTURE_DATA_PITCH_ALIGNMENT = 256;
  static size_t        align_up(size_t size, size_t alignment = BUFFER_ALIGNMENT) {
    return (size + alignment - 1) & ~(alignment - 1);
  }
  static size_t align_down(size_t size, size_t alignment = BUFFER_ALIGNMENT) {
    return (size) & ~(alignment - 1);
  }

  virtual Resource_ID create_image(Image_Create_Info info)                                     = 0;
  virtual Resource_ID create_buffer(Buffer_Create_Info info)                                   = 0;
  virtual Resource_ID create_shader(Stage_t type, string_ref text,
                                    Pair<string_ref, string_ref> *defines, size_t num_defines) = 0;
  virtual Resource_ID create_sampler(Sampler_Create_Info const &info)                          = 0;
  // Deferred release. Must call new_frame 3-6 times for the actual release to make sure it's not
  // used by the GPU.
  virtual void            release_resource(Resource_ID id)                                     = 0;
  virtual Resource_ID     create_event()                                                       = 0;
  virtual Resource_ID     create_timestamp()                                                   = 0;
  virtual Resource_ID     get_swapchain_image()                                                = 0;
  virtual Image2D_Info    get_swapchain_image_info()                                           = 0;
  virtual u32             get_num_swapchain_images()                                           = 0;
  virtual Image_Info      get_image_info(Resource_ID res_id)                                   = 0;
  virtual void *          map_buffer(Resource_ID id)                                           = 0;
  virtual void            unmap_buffer(Resource_ID id)                                         = 0;
  virtual Resource_ID     create_render_pass(Render_Pass_Create_Info const &info)              = 0;
  virtual Resource_ID     create_frame_buffer(Resource_ID                     render_pass,
                                              Frame_Buffer_Create_Info const &info)            = 0;
  virtual Resource_ID     create_compute_pso(Resource_ID signature, Resource_ID cs)            = 0;
  virtual Resource_ID     create_graphics_pso(Resource_ID signature, Resource_ID render_pass,
                                              Graphics_Pipeline_State const &)                 = 0;
  virtual Resource_ID     create_signature(Binding_Table_Create_Info const &info)              = 0;
  virtual IBinding_Table *create_binding_table(Resource_ID signature)                          = 0;
  virtual ICtx *          start_render_pass(Resource_ID render_pass, Resource_ID frame_buffer) = 0;
  virtual void            end_render_pass(ICtx *ctx)                                           = 0;
  virtual ICtx *          start_compute_pass()                                                 = 0;
  virtual void            end_compute_pass(ICtx *ctx)                                          = 0;
  virtual bool            get_timestamp_state(Resource_ID)                                     = 0;
  virtual double          get_timestamp_ms(Resource_ID t0, Resource_ID t1)                     = 0;
  virtual void            wait_idle()                                                          = 0;
  virtual bool            get_event_state(Resource_ID id)                                      = 0;
  virtual Impl_t          getImplType()                                                        = 0;
  virtual void            release()                                                            = 0;
  // Does the deferred release iteration and increments the swap chain image if there's any.
  virtual void start_frame() = 0;
  virtual void end_frame()   = 0;
};
// Single threaded entity.
// Used to record commands to command list/ buffer and later submit it via
// end_compute_pass/end_render_pass.
class ICtx {
  public:
  virtual void bind_table(IBinding_Table *table) = 0;
  // Graphics
  virtual void start_render_pass()                                                     = 0;
  virtual void end_render_pass()                                                       = 0;
  virtual void bind_graphics_pso(Resource_ID pso)                                      = 0;
  virtual void draw_indexed(u32 indices, u32 instances, u32 first_index, u32 first_instance,
                            i32 vertex_offset)                                         = 0;
  virtual void bind_index_buffer(Resource_ID id, size_t offset, Index_t format)        = 0;
  virtual void bind_vertex_buffer(u32 index, Resource_ID buffer, size_t offset)        = 0;
  virtual void draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) = 0;
  virtual void multi_draw_indexed_indirect(Resource_ID arg_buf_id, u32 arg_buf_offset,
                                           Resource_ID cnt_buf_id, u32 cnt_buf_offset,
                                           u32 max_count, u32 stride)                  = 0;

  virtual void set_viewport(float x, float y, float width, float height, float mindepth,
                            float maxdepth)                     = 0;
  virtual void set_scissor(u32 x, u32 y, u32 width, u32 height) = 0;
  // Compute
  virtual void bind_compute(Resource_ID id)              = 0;
  virtual void dispatch(u32 dim_x, u32 dim_y, u32 dim_z) = 0;
  // Memory movement
  virtual void fill_buffer(Resource_ID id, size_t offset, size_t size, u32 value) = 0;
  virtual void clear_image(Resource_ID id, Image_Subresource const &range,
                           Clear_Value const &cv)                                 = 0;
  // Unsupported by DX12?
  //virtual void update_buffer(Resource_ID buf_id, size_t offset, void const *data,
  //                           size_t data_size)                                    = 0;
  virtual void copy_buffer_to_image(Resource_ID buf_id, size_t buffer_offset, Resource_ID img_id,
                                    Image_Copy const &dst_info)                   = 0;
  virtual void copy_image_to_buffer(Resource_ID buf_id, size_t buffer_offset, Resource_ID img_id,
                                    Image_Copy const &dst_info)                   = 0;
  virtual void copy_buffer(Resource_ID src_buf_id, size_t src_offset, Resource_ID dst_buf_id,
                           size_t dst_offset, u32 size)                           = 0;
  // Synchronization
  virtual void image_barrier(Resource_ID image_id, Image_Access access) = 0;
  virtual void buffer_barrier(Resource_ID buf_id, Buffer_Access access) = 0;
  virtual void insert_event(Resource_ID id)                             = 0;
  virtual void insert_timestamp(Resource_ID timestamp_id)               = 0;
};

IDevice *create_vulkan(void *window_handler);
#ifdef WIN32
IDevice *create_dx12(void *window_handler);
#else
static inline IDevice *create_dx12(void *window_handler) { return NULL; }
#endif

} // namespace rd

#endif // RENDERING_HPP
