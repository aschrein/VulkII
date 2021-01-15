#include "include/utils.hlsli"

struct PushConstants {
  float4x4 model;
  u32      normal_offset;
  u32      normal_stride;
  u32      position_offset;
  u32      position_stride;
  u32      first_vertex;
  u32      index_offset;
  u32      index_count;
  u32      index_stride;
  u32      flags;
};

#define RASTERIZATION_FLAG_CULL_PIXELS 0x1
#define RASTERIZATION_GROUP_SIZE 64

struct FrameConstants {
  float4x4 viewproj;
};

struct GI_PushConstants {
  float4x4 model;
  u32      grid_size;
  u32      flags;
};

#define GI_RASTERIZATION_GROUP_SIZE 16
