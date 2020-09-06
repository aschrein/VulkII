#include "rendering.hpp"
#include "rendering_utils.hpp"

//
// TressFXTFXFileHeader Structure
//
// This structure defines the header of the file. The actual vertex data follows this as specified
// by the offsets.
struct TressFXTFXFileHeader {
  float        version;        // Specifies TressFX version number
  unsigned int numHairStrands; // Number of hair strands in this file. All strands in this file are
                               // guide strands. Follow hair strands are generated procedurally.
  unsigned int numVerticesPerStrand; // From 4 to 64 inclusive (POW2 only). This should be a fixed
                                     // value within tfx value. The total vertices from the tfx file
                                     // is numHairStrands * numVerticesPerStrand.

  // Offsets to array data starts here. Offset values are in bytes, aligned on 8 bytes boundaries,
  // and relative to beginning of the .tfx file
  unsigned int offsetVertexPosition; // Array size: FLOAT4[numHairStrands]
  unsigned int offsetStrandUV; // Array size: FLOAT2[numHairStrands], if 0 no texture coordinates
  unsigned int offsetVertexUV; // Array size: FLOAT2[numHairStrands * numVerticesPerStrand], if 0,
                               // no per vertex texture coordinates
  unsigned int offsetStrandThickness; // Array size: float[numHairStrands]
  unsigned int offsetVertexColor; // Array size: FLOAT4[numHairStrands * numVerticesPerStrand], if
                                  // 0, no vertex colors

  unsigned int reserved[32]; // Reserved for future versions
};

struct TressFXTFXBoneFileHeader {
  float        version;
  unsigned int numHairStrands;
  unsigned int numInfluenceBones;
  unsigned int offsetBoneNames;
  unsigned int offsetSkinningData;
  unsigned int reserved[32];
};

static void GetTangentVectors(const float4 &n, float4 &t0, float4 &t1) {
  if (fabsf(n[2]) > 0.707f) {
    float a = n[1] * n[1] + n[2] * n[2];
    float k = 1.0f / sqrtf(a);
    t0[0]   = 0;
    t0[1]   = -n[2] * k;
    t0[2]   = n[1] * k;

    t1[0] = a * k;
    t1[1] = -n[0] * t0[2];
    t1[2] = n[0] * t0[1];
  } else {
    float a = n[0] * n[0] + n[1] * n[1];
    float k = 1.0f / sqrtf(a);
    t0[0]   = -n[1] * k;
    t0[1]   = n[0] * k;
    t0[2]   = 0;

    t1[0] = -n[2] * t0[1];
    t1[1] = n[2] * t0[0];
    t1[2] = a * k;
  }
}

static float GetRandom(float Min, float Max) {
  return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

#define EI_Read(ptr, size, pFile) fread(ptr, size, 1, pFile)
#define EI_Seek(pFile, offset) fseek(pFile, offset, SEEK_SET)
#define EI_LogWarning(msg) printf("%s", msg)

#define AMD_TRESSFX_VERSION_MAJOR 4
#define AMD_TRESSFX_VERSION_MINOR 0
#define AMD_TRESSFX_VERSION_PATCH 0

#define TRESSFX_SIM_THREAD_GROUP_SIZE 64


struct TressFX_Hair {
  i32           m_numGuideStrands;
  i32           m_numVerticesPerStrand;
  i32           m_numFollowStrandsPerGuide;
  i32           m_numTotalStrands;
  i32           m_numGuideVertices;
  i32           m_numTotalVertices;
  Array<float4> m_positions;
  Array<float4> m_tangents;
  Array<u32>    m_triangleIndices;
  Array<float4> m_followRootOffsets;
  Array<float2> m_strandUV;

  Resource_ID gfx_positions;
  Resource_ID gfx_tangents;
  Resource_ID gfx_indices;

  void init_gfx(rd::IFactory *factory) {
    {
      rd::Buffer_Create_Info info;
      MEMZERO(info);
      info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits =
          (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      info.size     = sizeof(float4) * m_positions.size;
      gfx_positions = factory->create_buffer(info);
      init_buffer(factory, gfx_positions, m_positions.ptr, info.size);
    }
    {
      rd::Buffer_Create_Info info;
      MEMZERO(info);
      info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits =
          (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      info.size    = sizeof(float4) * m_tangents.size;
      gfx_tangents = factory->create_buffer(info);
      init_buffer(factory, gfx_tangents, m_tangents.ptr, info.size);
    }
    {
      rd::Buffer_Create_Info info;
      MEMZERO(info);
      info.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
      info.usage_bits = (u32)rd::Buffer_Usage_Bits::USAGE_INDEX_BUFFER |
                        (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
      info.size   = sizeof(u32) * m_triangleIndices.size;
      gfx_indices = factory->create_buffer(info);
      init_buffer(factory, gfx_indices, m_triangleIndices.ptr, info.size);
    }
  }
  void draw(rd::IFactory *factory) {}
  void release_gfx(rd::IFactory *factory) {
    factory->release_resource(gfx_positions);
    factory->release_resource(gfx_tangents);
    factory->release_resource(gfx_indices);
  }
  void init() {
    MEMZERO(*this);
    m_positions.init();
    m_followRootOffsets.init();
    m_tangents.init();
    m_triangleIndices.init();
    m_strandUV.init();
  }
  void release() {
    m_positions.release();
    m_followRootOffsets.release();
    m_tangents.release();
    m_triangleIndices.release();
    m_strandUV.release();
  }
};

bool LoadHairData(string_ref path, TressFX_Hair &out) {
  out.init();
  TressFXTFXFileHeader header = {};
  TMP_STORAGE_SCOPE;
  FILE *ioObject = fopen(stref_to_tmp_cstr(path), "rb");
  if (ioObject == NULL) return false;
  defer(fclose(ioObject));
  // read the header
  EI_Seek(ioObject, 0); // make sure the stream pos is at the beginning.
  EI_Read((void *)&header, sizeof(TressFXTFXFileHeader), ioObject);

  // If the tfx version is lower than the current major version, exit.
  if (header.version < AMD_TRESSFX_VERSION_MAJOR) {
    return false;
  }

  unsigned int numStrandsInFile = header.numHairStrands;

  // We make the number of strands be multiple of TRESSFX_SIM_THREAD_GROUP_SIZE.
  out.m_numGuideStrands = (numStrandsInFile - numStrandsInFile % TRESSFX_SIM_THREAD_GROUP_SIZE) +
                          TRESSFX_SIM_THREAD_GROUP_SIZE;

  out.m_numVerticesPerStrand = header.numVerticesPerStrand;

  // Make sure number of vertices per strand is greater than two and less than or equal to
  // thread group size (64). Also thread group size should be a mulitple of number of
  // vertices per strand. So possible number is 4, 8, 16, 32 and 64.
  ASSERT_ALWAYS(out.m_numVerticesPerStrand > 2 &&
                out.m_numVerticesPerStrand <= TRESSFX_SIM_THREAD_GROUP_SIZE &&
                TRESSFX_SIM_THREAD_GROUP_SIZE % out.m_numVerticesPerStrand == 0);

  out.m_numFollowStrandsPerGuide = 0;
  out.m_numTotalStrands =
      out.m_numGuideStrands; // Until we call GenerateFollowHairs, the number of total
                             // strands is equal to the number of guide strands.
  out.m_numGuideVertices = out.m_numGuideStrands * out.m_numVerticesPerStrand;
  out.m_numTotalVertices = out.m_numGuideVertices; // Again, the total number of vertices is equal
                                                   // to the number of guide vertices here.

  ASSERT_ALWAYS(out.m_numTotalVertices % TRESSFX_SIM_THREAD_GROUP_SIZE ==
                0); // number of total vertices should be multiple of thread group size.
                    // This assert is actually redundant because we already made m_numGuideStrands
                    // and m_numTotalStrands are multiple of thread group size.
                    // Just demonstrating the requirement for number of vertices here in case
                    // you are to make your own loader.

  out.m_positions.resize(out.m_numTotalVertices); // size of m_positions = number of total vertices
                                                  // * sizeo of each position vector.

  // Read position data from the io stream.
  EI_Seek(ioObject, header.offsetVertexPosition);
  EI_Read(
      (void *)out.m_positions.ptr, numStrandsInFile * out.m_numVerticesPerStrand * sizeof(float4),
      ioObject); // note that the position data in io stream contains only guide hairs. If we call
                 // GenerateFollowHairs to generate follow hairs, m_positions will be re-allocated.
  // for (i32 i = 0; i < numStrandsInFile * out.m_numVerticesPerStrand; ++i) {
  //  float4 in          = out.m_positions[i];
  //  out.m_positions[i] = float4(in.x, in.z, in.y, in.w);
  //}
  // We need to make up some strands to fill up the buffer because the number of strands from stream
  // is not necessarily multile of thread size.
  i32 numStrandsToMakeUp = out.m_numGuideStrands - numStrandsInFile;

  for (i32 i = 0; i < numStrandsToMakeUp; ++i) {
    for (i32 j = 0; j < out.m_numVerticesPerStrand; ++j) {
      i32 indexLastVertex          = (numStrandsInFile - 1) * out.m_numVerticesPerStrand + j;
      i32 indexVertex              = (numStrandsInFile + i) * out.m_numVerticesPerStrand + j;
      out.m_positions[indexVertex] = out.m_positions[indexLastVertex];
    }
  }

  // Read strand UVs
  EI_Seek(ioObject, header.offsetStrandUV);
  out.m_strandUV.resize(out.m_numTotalStrands); // If we call GenerateFollowHairs to generate follow
                                                // hairs, m_strandUV will be re-allocated.

  EI_Read((void *)out.m_strandUV.ptr, numStrandsInFile * sizeof(float2), ioObject);

  // Fill up the last empty space
  i32 indexLastStrand = (numStrandsInFile - 1);

  for (int i = 0; i < numStrandsToMakeUp; ++i) {
    i32 indexStrand             = (numStrandsInFile + i);
    out.m_strandUV[indexStrand] = out.m_strandUV[indexLastStrand];
  }

  out.m_followRootOffsets.resize(out.m_numTotalStrands);

  // Fill m_followRootOffsets with zeros
  memset(out.m_followRootOffsets.ptr, 0, out.m_numTotalStrands * sizeof(float4));

  out.m_tangents.resize(out.m_numTotalVertices);

  // compute tangent vectors
  {
    float4 *pos = (float4 *)out.m_positions.ptr;
    float4 *tan = (float4 *)out.m_tangents.ptr;

    for (int iStrand = 0; iStrand < out.m_numTotalStrands; ++iStrand) {
      int indexRootVertMaster = iStrand * out.m_numVerticesPerStrand;

      // vertex 0
      {
        float4 &vert_0 = pos[indexRootVertMaster];
        float4 &vert_1 = pos[indexRootVertMaster + 1];

        float4 tangent           = normalize(vert_1 - vert_0);
        tan[indexRootVertMaster] = tangent;
      }

      // vertex 1 through n-1
      for (int i = 1; i < (int)out.m_numVerticesPerStrand - 1; i++) {
        float4 &vert_i_minus_1 = pos[indexRootVertMaster + i - 1];
        float4 &vert_i         = pos[indexRootVertMaster + i];
        float4 &vert_i_plus_1  = pos[indexRootVertMaster + i + 1];

        float4 tangent_pre  = normalize(vert_i - vert_i_minus_1);
        float4 tangent_next = normalize(vert_i_plus_1 - vert_i);
        float4 tangent      = normalize(tangent_pre + tangent_next);

        tan[indexRootVertMaster + i] = tangent;
      }
    }
  }

  // compute thickness coefficients
  // ComputeThicknessCoeffs();

  // compute rest lengths
  // ComputeRestLengths();

  out.m_triangleIndices.resize(6 * out.m_numTotalStrands * (out.m_numVerticesPerStrand - 1));
  // triangle index
  {
    ASSERT_ALWAYS(out.m_numTotalVertices == out.m_numTotalStrands * out.m_numVerticesPerStrand);
    ASSERT_ALWAYS(out.m_triangleIndices.size != 0);

    i32 id     = 0;
    int iCount = 0;

    for (int i = 0; i < out.m_numTotalStrands; i++) {
      for (int j = 0; j < out.m_numVerticesPerStrand - 1; j++) {
        out.m_triangleIndices[iCount++] = 2 * id;
        out.m_triangleIndices[iCount++] = 2 * id + 1;
        out.m_triangleIndices[iCount++] = 2 * id + 2;
        out.m_triangleIndices[iCount++] = 2 * id + 2;
        out.m_triangleIndices[iCount++] = 2 * id + 1;
        out.m_triangleIndices[iCount++] = 2 * id + 3;

        id++;
      }

      id++;
    }

    ASSERT_ALWAYS(iCount ==
                  6 * out.m_numTotalStrands *
                      (out.m_numVerticesPerStrand - 1)); // iCount == GetNumHairTriangleIndices()
  }

  return true;
}

// This generates follow hairs around loaded guide hairs procedually with random distribution within
// the max radius input. Calling this is optional.
bool GenerateFollowHairs(TressFX_Hair &out, int numFollowHairsPerGuideHair,
                         float tipSeparationFactor, float maxRadiusAroundGuideHair) {
  ASSERT_ALWAYS(numFollowHairsPerGuideHair >= 0);

  out.m_numFollowStrandsPerGuide = numFollowHairsPerGuideHair;

  // Nothing to do, just exit.
  if (numFollowHairsPerGuideHair == 0) return false;

  // Recompute total number of hair strands and vertices with considering number of follow hairs per
  // a guide hair.
  out.m_numTotalStrands  = out.m_numGuideStrands * (out.m_numFollowStrandsPerGuide + 1);
  out.m_numTotalVertices = out.m_numTotalStrands * out.m_numVerticesPerStrand;

  // keep the old buffers until the end of this function.
  Array<float4> positionsGuide = out.m_positions.clone();
  Array<float2> strandUVGuide  = out.m_strandUV.clone();
  defer({
    positionsGuide.release();
    strandUVGuide.release();
  });
  // re-allocate all buffers
  out.m_positions.resize(out.m_numTotalVertices);
  out.m_strandUV.resize(out.m_numTotalStrands);

  out.m_followRootOffsets.resize(out.m_numTotalStrands);

  // type-cast to float4 to handle data easily.
  float4 *pos          = out.m_positions.ptr;
  float4 *followOffset = out.m_followRootOffsets.ptr;

  // Generate follow hairs
  for (int i = 0; i < out.m_numGuideStrands; i++) {
    int indexGuideStrand    = i * (out.m_numFollowStrandsPerGuide + 1);
    int indexRootVertMaster = indexGuideStrand * out.m_numVerticesPerStrand;

    memcpy(&pos[indexRootVertMaster], &positionsGuide[i * out.m_numVerticesPerStrand],
           sizeof(float4) * out.m_numVerticesPerStrand);
    out.m_strandUV[indexGuideStrand] = strandUVGuide[i];

    followOffset[indexGuideStrand]   = float4(0, 0, 0, 0);
    followOffset[indexGuideStrand].w = (float)indexGuideStrand;
    float4 v01                       = pos[indexRootVertMaster + 1] - pos[indexRootVertMaster];
    v01                              = normalize(v01);

    // Find two orthogonal unit tangent vectors to v01
    float4 t0, t1;
    GetTangentVectors(v01, t0, t1);

    for (int j = 0; j < out.m_numFollowStrandsPerGuide; j++) {
      int indexStrandFollow   = indexGuideStrand + j + 1;
      int indexRootVertFollow = indexStrandFollow * out.m_numVerticesPerStrand;

      out.m_strandUV[indexStrandFollow] = out.m_strandUV[indexGuideStrand];

      // offset vector from the guide strand's root vertex position
      float4 offset = GetRandom(-maxRadiusAroundGuideHair, maxRadiusAroundGuideHair) * t0 +
                      GetRandom(-maxRadiusAroundGuideHair, maxRadiusAroundGuideHair) * t1;
      followOffset[indexStrandFollow]   = offset;
      followOffset[indexStrandFollow].w = (float)indexGuideStrand;

      for (int k = 0; k < out.m_numVerticesPerStrand; k++) {
        const float4 *guideVert  = &pos[indexRootVertMaster + k];
        float4 *      followVert = &pos[indexRootVertFollow + k];

        float factor =
            tipSeparationFactor * ((float)k / ((float)out.m_numVerticesPerStrand)) + 1.0f;
        *followVert     = *guideVert + offset * factor;
        (*followVert).w = guideVert->w;
      }
    }
  }

  return true;
}

Config g_config;
Camera g_camera;

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
(
 (add u32  g_buffer_width 512 (min 4) (max 1024))
 (add u32  g_buffer_height 512 (min 4) (max 1024))
 (add bool forward 1)
 (add bool "depth test" 1)
 (add f32  strand_size 1.0 (min 0.1) (max 16.0))
)
)"));

  char *state = read_file_tmp("scene_state");

  if (state != NULL) {
    TMP_STORAGE_SCOPE;
    List *cur = List::parse(stref_s(state), Tmp_List_Allocator());
    init_traverse(cur);
  }
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

struct Hair_Renderer {
  TressFX_Hair hair;
  Resource_ID  hair_img;
  Resource_ID  hair_depth;
  struct PPLL_Node {
    u32 color;
    u32 next;
    u32 data;
    u32 depth;
  };
  Resource_ID ppll_heads;   // uint[width * height]
  Resource_ID ppll_nodes;   // Node[width * height * MAX_NODES]
  Resource_ID ppll_counter; // uint[1]

  TimeStamp_Pool prepass_timestamp;
  TimeStamp_Pool clear_timestamp;
  TimeStamp_Pool resolve_timestamp;

  static constexpr u32 MAX_NODES = 16u;

  void init(rd::IFactory *factory) { //
    MEMZERO(*this);
    ASSERT_ALWAYS(LoadHairData(stref_s("Ratboy_short.tfx"), hair));
    hair.init_gfx(factory);
  }
  void render(rd::IFactory *factory) {
    clear_timestamp.update(factory);
    prepass_timestamp.update(factory);
    resolve_timestamp.update(factory);
    bool recreate = false;
    u32  width    = g_config.get_u32("g_buffer_width");
    u32  height   = g_config.get_u32("g_buffer_height");
    if (hair_img.is_null()) recreate = true;
    if (recreate == false) {
      auto img_info = factory->get_image_info(hair_img);
      recreate      = img_info.width != width || img_info.height != height;
    }
    if (recreate) {
      {
        rd::Buffer_Create_Info info;
        MEMZERO(info);
        info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits =
            (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        info.size = width * height * 4;
        if (ppll_heads.is_null() == false) factory->release_resource(ppll_heads);
        ppll_heads = factory->create_buffer(info);
      }
      {
        rd::Buffer_Create_Info info;
        MEMZERO(info);
        info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits =
            (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        info.size = width * height * sizeof(PPLL_Node) * MAX_NODES;
        if (ppll_nodes.is_null() == false) factory->release_resource(ppll_nodes);
        ppll_nodes = factory->create_buffer(info);
      }
      {
        rd::Buffer_Create_Info info;
        MEMZERO(info);
        info.mem_bits = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        info.usage_bits =
            (u32)rd::Buffer_Usage_Bits::USAGE_UAV | (u32)rd::Buffer_Usage_Bits::USAGE_TRANSFER_DST;
        info.size = 4;
        if (ppll_counter.is_null() == false) factory->release_resource(ppll_counter);
        ppll_counter = factory->create_buffer(info);
      }
      {
        rd::Image_Create_Info ci;
        MEMZERO(ci);
        ci.format     = rd::Format::RGBA32_FLOAT;
        ci.depth      = 1;
        ci.width      = width;
        ci.height     = height;
        ci.layers     = 1;
        ci.levels     = 1;
        ci.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        ci.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_TRANSFER_DST | //
                        (u32)rd::Image_Usage_Bits::USAGE_RT |           //
                        (u32)rd::Image_Usage_Bits::USAGE_UAV |          //
                        (u32)rd::Image_Usage_Bits::USAGE_SAMPLED;
        if (hair_img.is_null() == false) factory->release_resource(hair_img);
        hair_img = factory->create_image(ci);
      }
      {
        rd::Image_Create_Info ci;
        MEMZERO(ci);
        ci.format     = rd::Format::D32_FLOAT;
        ci.depth      = 1;
        ci.width      = width;
        ci.height     = height;
        ci.layers     = 1;
        ci.levels     = 1;
        ci.mem_bits   = (u32)rd::Memory_Bits::DEVICE_LOCAL;
        ci.usage_bits = (u32)rd::Image_Usage_Bits::USAGE_DT;
        if (hair_depth.is_null() == false) factory->release_resource(hair_depth);
        hair_depth = factory->create_image(ci);
      }
    }
    // Init
    {
      rd::Imm_Ctx *ctx = factory->start_compute_pass();
      clear_timestamp.insert(factory, ctx);
      ctx->fill_buffer(ppll_counter, 0, 4, 1);
      ctx->fill_buffer(ppll_heads, 0, width * height * 4, 0);
      clear_timestamp.insert(factory, ctx);
      factory->end_compute_pass(ctx);
    }
    {
      rd::Render_Pass_Create_Info info;
      MEMZERO(info);
      info.width  = width;
      info.height = height;
      rd::RT_View rt0;
      MEMZERO(rt0);
      rt0.image             = hair_img;
      rt0.format            = rd::Format::NATIVE;
      rt0.clear_color.clear = true;
      rt0.clear_color.r     = 0.0f;
      info.rts.push(rt0);
      if (g_config.get_bool("depth test")) {
        info.depth_target.image             = hair_depth;
        info.depth_target.clear_depth.clear = true;
        info.depth_target.format            = rd::Format::NATIVE;
      }
      rd::Imm_Ctx *ctx = factory->start_render_pass(info);
      prepass_timestamp.insert(factory, ctx);
      ctx->VS_set_shader(factory->create_shader_raw(rd::Stage_t::VERTEX, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name view))
  (add_field (type float4x4)  (name proj))
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float3)    (name camera_offset))
  (add_field (type float4)    (name viewport))
  (add_field (type uint)      (name NumVerticesPerStrand))
  (add_field (type float)     (name strand_size))
)
@(DECLARE_BUFFER (set 0) (binding 1) (type float4) (name positions))
@(DECLARE_BUFFER (set 0) (binding 2) (type float4) (name tangents))

@(DECLARE_OUTPUT (location 0) (type float2) (name pixel_uv))
@(DECLARE_OUTPUT (location 1) (type float) (name pixel_depth))

@(ENTRY)
  u32 pos_id         = (VERTEX_INDEX) / 2;
  float offset       = float(VERTEX_INDEX & 1) * 2.0 - 1.0;
  float4 in_pos      = buffer_load(positions, pos_id);
  float4 in_tangents = buffer_load(tangents, pos_id);
  pixel_uv.x         = offset * 0.5 + 0.5;
  pixel_uv.y         = float(pos_id % NumVerticesPerStrand) / (NumVerticesPerStrand - 1);
  float3 n           = normalize(cross(in_tangents.xyz, in_pos.xyz - camera_offset));
  float4 pn          = viewproj * float4(n, 0.0);
  float4 ppos        = viewproj * float4(in_pos.xyz, 1.0);
  float4 pt          = viewproj * float4(in_tangents.xyz, 1.0);
  ppos.xy += (strand_size + 0.71) * offset * viewport.ww * normalize(pn.xy) * ppos.w;
  ppos.z  += ppos.z * float((VERTEX_INDEX & 0xf)) * 1.0e-6;
  pixel_depth = ppos.z / ppos.w;
  @(EXPORT_POSITION ppos);
@(END)
)"),
                                                    NULL, 0));
      ctx->PS_set_shader(factory->create_shader_raw(rd::Stage_t::PIXEL, stref_s(R"(
@(DECLARE_UNIFORM_BUFFER
  (set 0)
  (binding 0)
  (add_field (type float4x4)  (name view))
  (add_field (type float4x4)  (name proj))
  (add_field (type float4x4)  (name viewproj))
  (add_field (type float3)    (name camera_offset))
  (add_field (type float4)    (name viewport))
  (add_field (type uint)      (name NumVerticesPerStrand))
  (add_field (type float)     (name strand_size))
)
struct PPLL_Node {
  u32 color;
  u32 next;
  u32 data;
  u32 depth;
};
@(DECLARE_BUFFER (set 1) (binding 0) (type uint)      (name ppll_heads))
@(DECLARE_BUFFER (set 1) (binding 1) (type PPLL_Node) (name ppll_nodes))
@(DECLARE_BUFFER (set 1) (binding 2) (type uint)      (name ppll_counter))
u32 float4_to_u32(float4 c) {
  u32 r = u32(clamp(c.x, 0.0, 1.0) * 255.0);
  u32 g = u32(clamp(c.y, 0.0, 1.0) * 255.0);
  u32 b = u32(clamp(c.z, 0.0, 1.0) * 255.0);
  u32 a = u32(clamp(c.w, 0.0, 1.0) * 255.0);
  return
        (r << 24) |
        (g << 16) |
        (b <<  8) |
        (a <<  0);
}
@(DECLARE_INPUT (location 0) (type float2) (name pixel_uv))
@(DECLARE_INPUT (location 1) (type float) (name pixel_depth))
@(DECLARE_RENDER_TARGET  (location 0))
@(ENTRY)
  PPLL_Node node;
  uint frag_index = uint(FRAGMENT_COORDINATES.x) +
                    uint(FRAGMENT_COORDINATES.y) * uint(viewport.x);
  node.data  = 0;
  node.depth = bitcast_f32_to_u32(pixel_depth);
  u32 node_index = buffer_atomic_add(ppll_counter, 0, 1);
  node.color = float4_to_u32(float4(pixel_uv, 0.3, 0.1));
  node.next  = buffer_atomic_exchange(ppll_heads, frag_index, node_index);
  buffer_store(ppll_nodes, node_index, node);
  @(EXPORT_COLOR 0 float4(pixel_uv, 0.0, 1.0));
@(END)
)"),
                                                    NULL, 0));
      rd::Blend_State bs;
      MEMZERO(bs);
      bs.enabled = false;
      if (g_config.get_bool("forward")) {
        bs.color_write_mask =
            (u32)rd::Color_Component_Bit::R_BIT | (u32)rd::Color_Component_Bit::G_BIT |
            (u32)rd::Color_Component_Bit::B_BIT | (u32)rd::Color_Component_Bit::A_BIT;
      }
      ito(1) ctx->OM_set_blend_state(i, bs);
      ctx->IA_set_topology(rd::Primitive::TRIANGLE_LIST);
      rd::RS_State rs_state;
      MEMZERO(rs_state);
      rs_state.polygon_mode = rd::Polygon_Mode::FILL;
      rs_state.front_face   = rd::Front_Face::CW;
      rs_state.cull_mode    = rd::Cull_Mode::NONE;
      ctx->RS_set_state(rs_state);
      rd::DS_State ds_state;
      MEMZERO(ds_state);
      if (g_config.get_bool("depth test")) {
        ds_state.cmp_op             = rd::Cmp::GE;
        ds_state.enable_depth_test  = true;
        ds_state.enable_depth_write = true;
      }
      ctx->DS_set_state(ds_state);
      rd::MS_State ms_state;
      MEMZERO(ms_state);
      ms_state.sample_mask = 0xffffffffu;
      ms_state.num_samples = 1;
      ctx->MS_set_state(ms_state);
      struct Uniform {
        afloat4x4 view;
        afloat4x4 proj;
        afloat4x4 viewproj;
        afloat3   camera_offset;
        afloat4   viewport;
        u32       NumVerticesPerStrand;
        f32       strand_size;
      } ubo_data;
      ubo_data.view        = g_camera.view;
      ubo_data.proj        = g_camera.proj;
      ubo_data.viewproj    = g_camera.viewproj();
      ubo_data.strand_size = g_config.get_f32("strand_size");

      ubo_data.camera_offset        = g_camera.pos;
      ubo_data.viewport             = float4( //
          (f32)width,             //
          (f32)height,            //
          1.0f / (f32)width,      //
          1.0f / (f32)height);
      ubo_data.NumVerticesPerStrand = hair.m_numVerticesPerStrand;
      Resource_ID ubo               = create_uniform(factory, ubo_data);
      defer(factory->release_resource(ubo));
      ctx->bind_uniform_buffer(0, 0, ubo, 0, 0);
      ctx->bind_storage_buffer(0, 1, hair.gfx_positions, 0, 0);
      ctx->bind_storage_buffer(0, 2, hair.gfx_tangents, 0, 0);
      ctx->bind_storage_buffer(1, 0, ppll_heads, 0, 0);
      ctx->bind_storage_buffer(1, 1, ppll_nodes, 0, 0);
      ctx->bind_storage_buffer(1, 2, ppll_counter, 0, 0);
      ctx->set_viewport(0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
      ctx->set_scissor(0, 0, g_config.get_u32("g_buffer_width"), (float)height);
      ctx->IA_set_index_buffer(hair.gfx_indices, 0, rd::Index_t::UINT32);
      ctx->draw_indexed(hair.m_triangleIndices.size, 1, 0, 0, 0);
      prepass_timestamp.insert(factory, ctx);
      factory->end_render_pass(ctx);
    }
    // Resolve
    if (!g_config.get_bool("forward")) {

      rd::Imm_Ctx *ctx = factory->start_compute_pass();
      resolve_timestamp.insert(factory, ctx);
      ctx->CS_set_shader(factory->create_shader_raw(rd::Stage_t::COMPUTE, stref_s(R"(
struct PPLL_Node {
  u32 color;
  u32 next;
  u32 data;
  f32 depth;
};
@(DECLARE_IMAGE
  (type WRITE_ONLY)
  (dim 2D)
  (set 0)
  (binding 0)
  (format RGBA32_FLOAT)
  (name out_image)
)
@(DECLARE_BUFFER (set 1) (binding 0) (type uint)      (name ppll_heads))
@(DECLARE_BUFFER (set 1) (binding 1) (type PPLL_Node) (name ppll_nodes))
@(DECLARE_BUFFER (set 1) (binding 2) (type uint)      (name ppll_counter))

float4 u32_to_float4(u32 c) {
  return float4(
    float((c >> 24) & 0xffu) / 255.0,
    float((c >> 16) & 0xffu) / 255.0,
    float((c >> 8)  & 0xffu) / 255.0,
    float((c >> 0)  & 0xffu) / 255.0
  );
}

#define NODES_PER_PIXEL 8

@(GROUP_SIZE 16 16 1)
@(ENTRY)
  int2 dim = imageSize(out_image);
  if (GLOBAL_THREAD_INDEX.x > dim.x || GLOBAL_THREAD_INDEX.y > dim.y)
    return;
  u32 frag_index = u32(GLOBAL_THREAD_INDEX.x + GLOBAL_THREAD_INDEX.y * dim.x);
  u32 head = buffer_load(ppll_heads, frag_index);
  if (head == 0)
    return;
  
  PPLL_Node nodes[NODES_PER_PIXEL];
  for (u32 i = 0; i < NODES_PER_PIXEL; i++) {
    nodes[i].depth = 0.0;
  }
  u32 cnt = 0;
  for (u32 k = 0; k < 16; k++) {
    PPLL_Node node = buffer_load(ppll_nodes, head);
    u32   min_id    = 0;
    float min_depth = 1.0;
    for (u32 i = 0; i < NODES_PER_PIXEL; i++) {
      float depth = nodes[i].depth;            
      if (min_depth > depth) {               
        min_depth = depth;                   
        min_id    = i;                       
      }                                      
    }                                            

    if (node.depth > min_depth)
      nodes[min_id] = node;
    head = node.next;
    if (head == 0)
      break;
    cnt += 1;
  }
  float4 color = float4(0.0, 0.0, 0.0, 1.0);
  // float4 color = float4(float(cnt) / 16.0, 0.0, 0.0, 1.0);
  for (u32 k = 0; k < NODES_PER_PIXEL; k++) {
    u32   min_id    = 0;
    float min_depth = 1.0;
    for (u32 i = 0; i < NODES_PER_PIXEL; i++) {
      float depth = nodes[i].depth;            
      if (min_depth > depth && depth != 0.0) {               
        min_depth = depth;                   
        min_id    = i;                       
      }                                      
    }
    if (min_depth == 0.0)
      break;
    nodes[min_id].depth = 0.0;
    float4 node_color = u32_to_float4(nodes[min_id].color);
    color = node_color.xyzw;// * node_color.w + color.xyzw * (1.0 - node_color.w);
  }
  image_store(out_image, GLOBAL_THREAD_INDEX.xy, float4(color.xyz, 1.0));
@(END)
)"),
                                                    NULL, 0));
      ctx->bind_rw_image(0, 0, 0, hair_img, rd::Image_Subresource::top_level(), rd::Format::NATIVE);
      ctx->image_barrier(hair_img, (u32)rd::Access_Bits::MEMORY_WRITE,
                         rd::Image_Layout::SHADER_READ_WRITE_OPTIMAL);
      ctx->buffer_barrier(ppll_heads, (u32)rd::Access_Bits::MEMORY_READ);
      ctx->buffer_barrier(ppll_nodes, (u32)rd::Access_Bits::MEMORY_READ);
      ctx->buffer_barrier(ppll_counter, (u32)rd::Access_Bits::MEMORY_READ);
      ctx->bind_storage_buffer(1, 0, ppll_heads, 0, 0);
      ctx->bind_storage_buffer(1, 1, ppll_nodes, 0, 0);
      ctx->bind_storage_buffer(1, 2, ppll_counter, 0, 0);
      ctx->dispatch((width + 15) / 16, (height + 15) / 16, 1);
      resolve_timestamp.insert(factory, ctx);
      factory->end_compute_pass(ctx);
    }
  }
  void release(rd::IFactory *factory) { //
    hair.release_gfx(factory);
    factory->release_resource(hair_img);
    factory->release_resource(hair_depth);
  }
};

class Event_Consumer : public IGUI_Pass {
  Hair_Renderer hr;

  public:
  void init(rd::Pass_Mng *pmng) { //
    IGUI_Pass::init(pmng);
    g_camera.init();
  }
  void on_gui(rd::IFactory *factory) override { //
    ImGui::Begin("Config");
    g_config.on_imgui();
    ImGui::LabelText("clear pass", "%f ms", hr.clear_timestamp.duration);
    ImGui::LabelText("pre  pass", "%f ms", hr.prepass_timestamp.duration);
    ImGui::LabelText("resolve pass", "%f ms", hr.resolve_timestamp.duration);
    ImGui::End();
    ImGui::Begin("main viewport");

    auto wsize = get_window_size();
    ImGui::Image(bind_texture(hr.hair_img, 0, 0, rd::Format::NATIVE), ImVec2(wsize.x, wsize.y));
    auto wpos       = ImGui::GetCursorScreenPos();
    auto iinfo      = factory->get_image_info(hr.hair_img);
    g_camera.aspect = float(iinfo.height) / iinfo.width;
    ImGuiIO &io     = ImGui::GetIO();
    if (ImGui::IsWindowHovered()) {
      auto scroll_y = ImGui::GetIO().MouseWheel;
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
      ImVec2 mpos    = ImGui::GetMousePos();
      i32    cur_m_x = mpos.x;
      i32    cur_m_y = mpos.y;
      if (io.MouseDown[0] && last_m_x > 0) {
        i32 dx = cur_m_x - last_m_x;
        i32 dy = cur_m_y - last_m_y;
        g_camera.phi += (float)(dx)*g_camera.aspect * 5.0e-3f;
        g_camera.theta -= (float)(dy)*5.0e-3f;
      }
      last_m_x = cur_m_x;
      last_m_y = cur_m_y;
    }
    g_camera.update();
    ImGui::End();
  }
  void on_init(rd::IFactory *factory) override { //
    hr.init(factory);
  }
  void on_release(rd::IFactory *factory) override { //
    IGUI_Pass::release(factory);
    hr.release(factory);
  }
  void consume(void *_event) override { //
    IGUI_Pass::consume(_event);
  }
  void on_frame(rd::IFactory *factory) override { //
    hr.render(factory);
    IGUI_Pass::on_frame(factory);
  }
};

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  rd::Pass_Mng *pmng = rd::Pass_Mng::create(rd::Impl_t::VULKAN);
  IGUI_Pass *   gui  = new Event_Consumer;
  pmng->set_event_consumer(gui);
  pmng->loop();
  return 0;
}
