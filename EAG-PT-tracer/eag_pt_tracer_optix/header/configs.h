// [surfels]

// #define SURFEL_PIXEL_VALUE_COUNT_RADIANCE 3
// #define SURFEL_PIXEL_VALUE_COUNT_EMISSIVE 1
// #define SURFEL_PIXEL_VALUE_COUNT_ALBEDO 3

// #define SURFEL_PIXEL_VALUE_COUNT (SURFEL_PIXEL_VALUE_COUNT_RADIANCE + SURFEL_PIXEL_VALUE_COUNT_EMISSIVE + SURFEL_PIXEL_VALUE_COUNT_ALBEDO)

// #define SURFEL_PIXEL_VALUE_OFFSET_RADIANCE 0
// #define SURFEL_PIXEL_VALUE_OFFSET_EMISSIVE (SURFEL_PIXEL_VALUE_COUNT_RADIANCE)
// #define SURFEL_PIXEL_VALUE_OFFSET_ALBEDO (SURFEL_PIXEL_VALUE_COUNT_RADIANCE + SURFEL_PIXEL_VALUE_COUNT_EMISSIVE)

// [anyhit]

// should test on test scenes to find the best one
#define ANYHIT_CHUNK_BUFFER_SIZE 16

// so enough for indoor scenes
#define SCENE_MAX_DISTANCE 999999

// need more exploration
#define MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION 0.00001f

// can lower this to speed up (ref: 3DGRT)
#define TRANSMITTANCE_THRESHOLD 0.0001f

// generally 233 should be enough
#define MAXIMAL_HITCOUNT_ALONG_THE_RAY 2333

// [path tracing]

// iterative path tracing, instead of recursive.
// pipelineLinkOptions.maxTraceDepth = 1; is correct.
// set to 1 means 0-bounce
// #define BOUNCE_COUNT_LIMIT 7

// spp z: pass from python
// #define PATH_TRACING_SAMPLE_PER_PIXEL 256

// to avoid self-intersection when bouncing
#define NEAREST_DISTANCE_TO_AVOID_FRONT_SURFELS 0.2f

#define EMISSIVE_IS_LIGHT_SOURCE_THRESHOLD 0.1f

// [MIS]

#define USE_COSINE_WEIGHTED_SAMPLING

#define DIRECT_LIGHT_SAMPLING_TU_TV_MULTIPLE 3.0f

// [payload]

#define PAYLOAD_COUNT_TO_USE 2
