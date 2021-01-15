#ifdef HLSL
#  define u32 uint
#  define i32 int
#  define f32 float
#  define HLSL
#  define f64 double
#else
template <typename T> class ConstantBuffer {};
template <typename T> class Texture2D {};
template <typename T> class RWTexture2D {};
class SamplerState {};
#endif

#define float2_splat(x) float2(x, x)
#define float3_splat(x) float3(x, x, x)
#define float4_splat(x) float4(x, x, x, x)