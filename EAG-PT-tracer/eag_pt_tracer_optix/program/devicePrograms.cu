#include <optix_device.h>

#include "../header/configs.h"
#include "../header/LaunchParams.h"
#include "../header/vec_math.h"
#include <stdint.h>
#include <curand_kernel.h>

using namespace osc;

namespace osc
{
    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void *ptr = reinterpret_cast<void *>(uptr);
        return ptr;
    }

    static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template <typename T>
    static __forceinline__ __device__ T *getPerRayData()
    {
        const uint32_t per_ray_data_u0 = optixGetPayload_0();
        const uint32_t per_ray_data_u1 = optixGetPayload_1();
        return reinterpret_cast<T *>(unpackPointer(per_ray_data_u0, per_ray_data_u1));
    }

    struct IntersectionInfo
    {
        float distance;
        int surfel_id;
        float2 surfel_uv;
    };
    typedef struct IntersectionInfo IntersectionInfo;

    // Define the ray payload.
    // Ray pyaload is used to pass data between optixTrace
    // and the programs invoked during ray traversal.
    struct PerRayData
    {
        IntersectionInfo *buffer; // hit buffer for one chunk
    };
    typedef struct PerRayData PerRayData;

    //------------------------------------------------------------------------------

    __device__ void trace_forth_with_material(
        // [input]

        const float3 &ray_origin, const float3 &ray_direction,

        PerRayData per_ray_data,
        uint32_t &per_ray_data_u0,
        uint32_t &per_ray_data_u1,

        // [output]

        int &Hitcount,
        float &Alpha,

        float &Distance,
        float3 &Normal,

        float3 &Radiance,
        float &Emissive,
        float3 &Albedo,
        float &Roughness,
        float &Metallic,
        int *dominant_surfel_id = nullptr,
        float2 *dominant_surfel_uv = nullptr,
        float *dominant_surfel_distance = nullptr,
        float3 *EmissionRadiance = nullptr)
    {
        // avoid surfels near camera
        float distance_to_start_tracing_ray = NEAREST_DISTANCE_TO_AVOID_FRONT_SURFELS;

        Hitcount = 0;

        float Transmittance = 1.0f;
        float Transmittance_tobe = 1.0f;

        Distance = 0.0f;
        Normal = make_float3(0.0f, 0.0f, 0.0f);

        Radiance = make_float3(0.0f, 0.0f, 0.0f);
        Emissive = 0.0f;
        Albedo = make_float3(0.0f, 0.0f, 0.0f);
        Roughness = 0.0f;
        Metallic = 0.0f;
        float dominant_weight = 0.0f;
        if (dominant_surfel_id != nullptr)
            *dominant_surfel_id = -1;
        if (dominant_surfel_uv != nullptr)
            *dominant_surfel_uv = make_float2(0.0f);
        if (dominant_surfel_distance != nullptr)
            *dominant_surfel_distance = 0.0f;
        if (EmissionRadiance != nullptr)
            *EmissionRadiance = make_float3(0.0f);

        // tracing (0-bounce) along the ray
        while (1)
        {
            // initialized chunk buffer
            for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; i++)
                per_ray_data.buffer[i].distance = SCENE_MAX_DISTANCE;

            // trace a chunk and collect next k closest intersected surfels along the ray
            optixTrace(optixLaunchParams.traversable,
                       ray_origin,
                       ray_direction,
                       distance_to_start_tracing_ray, // tmin
                       SCENE_MAX_DISTANCE,            // tmax
                       0.0f,                          // rayTime
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_NONE,
                       0,                   // SBT offset
                       0,                   // SBT stride
                       0,                   // missSBTIndex
                       per_ray_data_u0, per_ray_data_u1);

            // accumulate the chunk
            for (int i_surfel_in_chunk = 0; i_surfel_in_chunk < ANYHIT_CHUNK_BUFFER_SIZE; ++i_surfel_in_chunk)
            {
                // [get per_ray_data from buffer]

                const float ray_distance = per_ray_data.buffer[i_surfel_in_chunk].distance;
                if (ray_distance == SCENE_MAX_DISTANCE)
                    break;
                const int surfel_id = per_ray_data.buffer[i_surfel_in_chunk].surfel_id;
                const float2 surfel_uv = per_ray_data.buffer[i_surfel_in_chunk].surfel_uv;

                // [get surfel properties by surfel_id]

                const float surfel_opacity = optixLaunchParams.surfels_opacities[surfel_id];
                const float4 surfel_quaternion = optixLaunchParams.surfels_quaternions[surfel_id];

                // normal

                const float surfel_quaternion_r = surfel_quaternion.x;
                const float surfel_quaternion_x = surfel_quaternion.y;
                const float surfel_quaternion_y = surfel_quaternion.z;
                const float surfel_quaternion_z = surfel_quaternion.w;
                // the third column
                float3 surfel_normal = make_float3(2.0f * (surfel_quaternion_x * surfel_quaternion_z + surfel_quaternion_r * surfel_quaternion_y),
                                                   2.0f * (surfel_quaternion_y * surfel_quaternion_z - surfel_quaternion_r * surfel_quaternion_x),
                                                   1.0f - 2.0f * (surfel_quaternion_x * surfel_quaternion_x + surfel_quaternion_y * surfel_quaternion_y));
                if (dot(surfel_normal, ray_direction) > 0.0f)
                {
                    surfel_normal *= -1.0f;
                }

                // [accumulate start]

                const float gaussian_value = exp(-0.5f * (surfel_uv.x * surfel_uv.x + surfel_uv.y * surfel_uv.y));
                const float eta = surfel_opacity * gaussian_value;
                const float weight = Transmittance * eta;

                if (weight < 1.0e-6f)
                {
                    continue;
                }

                Transmittance_tobe = Transmittance * (1.0f - eta);
                if (Transmittance_tobe < TRANSMITTANCE_THRESHOLD)
                {
                    break;
                }

                // this is the distance from the ray_origin_safe
                Distance += weight * ray_distance;
                Normal += weight * surfel_normal;

                Radiance += weight * optixLaunchParams.surfels_radiances[surfel_id];
                Emissive += weight * optixLaunchParams.surfels_emissives[surfel_id];
                if (EmissionRadiance != nullptr)
                {
                    *EmissionRadiance += weight *
                                         optixLaunchParams.surfels_radiances[surfel_id] *
                                         fminf(fmaxf(optixLaunchParams.surfels_emissives[surfel_id], 0.0f), 1.0f);
                }
                Albedo += weight * optixLaunchParams.surfels_albedos[surfel_id];
                if (optixLaunchParams.surfels_roughnesses != nullptr)
                    Roughness += weight * optixLaunchParams.surfels_roughnesses[surfel_id];
                if (optixLaunchParams.surfels_metallics != nullptr)
                    Metallic += weight * optixLaunchParams.surfels_metallics[surfel_id];

                if (weight > dominant_weight)
                {
                    dominant_weight = weight;
                    if (dominant_surfel_id != nullptr)
                        *dominant_surfel_id = surfel_id;
                    if (dominant_surfel_uv != nullptr)
                        *dominant_surfel_uv = surfel_uv;
                    if (dominant_surfel_distance != nullptr)
                        *dominant_surfel_distance = ray_distance;
                }

                // [accumulate end]

                Hitcount++;
                if (Hitcount >= MAXIMAL_HITCOUNT_ALONG_THE_RAY)
                    break;

                Transmittance = Transmittance_tobe;
            }

            // break again to end the trace_forth(); previous break only end the chunk
            if ((per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance == SCENE_MAX_DISTANCE) || (Transmittance_tobe < TRANSMITTANCE_THRESHOLD) || (Hitcount >= MAXIMAL_HITCOUNT_ALONG_THE_RAY))
                break;

            // update tmin for tracing next chunk
            distance_to_start_tracing_ray = per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance + MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION;
        }

        Alpha = 1.0f - Transmittance;
    }

    __device__ void trace_forth(
        const float3 &ray_origin, const float3 &ray_direction,
        PerRayData per_ray_data, uint32_t &per_ray_data_u0, uint32_t &per_ray_data_u1,
        int &Hitcount, float &Alpha, float &Distance, float3 &Normal,
        float3 &Radiance, float &Emissive, float3 &Albedo)
    {
        float roughness = 0.0f;
        float metallic = 0.0f;
        trace_forth_with_material(ray_origin, ray_direction, per_ray_data, per_ray_data_u0, per_ray_data_u1,
            Hitcount, Alpha, Distance, Normal, Radiance, Emissive, Albedo, roughness, metallic);
    }

    __device__ void sample_upper_hemisphere_direction(
        const float3 &normal,
        curandState &curand_state,
        float3 &sampled_direction,
        float &sampled_probability)
    {
        // [cosine-weighted sample (material sample)]

        // [sample a new direction - cosine-weighted and uniform-sampling]

        // https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
        const float random_u1 = curand_uniform(&curand_state);
        const float random_u2 = curand_uniform(&curand_state);

        // // these two values keeps the same when re-run the program
        // if ((launchIndex.x == WIDTH / 2) && (launchIndex.y == HEIGHT / 2) && (launchIndex.z == 0))
        // {
        //     printf("############################################\n");
        //     printf("random_u1=%f, random_u2=%f\n", random_u1, random_u2);
        //     printf("############################################\n");
        // }

#ifdef USE_COSINE_WEIGHTED_SAMPLING
        const float z = max(sqrtf(random_u1), 1e-6f);
#else
        const float z = max(random_u1, 1e-6f);
#endif

        const float r = max(sqrtf(1.0f - z * z), 1e-6f);
        const float phi = 2.0f * M_PIf * random_u2;
        const float x = r * cosf(phi);
        const float y = r * sinf(phi);

        const float3 local_direction = normalize(make_float3(x, y, z));

        float3 arbitrary = make_float3(0.0f, 0.0f, 1.0f);
        if (normal.z > 0.999f)
            arbitrary = make_float3(0.0f, 1.0f, 0.0f);

        const float3 tangent = normalize(cross(normal, arbitrary));
        const float3 bitangent = normalize(cross(normal, tangent));

        // update the incident direction
        sampled_direction = normalize(local_direction.x * tangent + local_direction.y * bitangent + local_direction.z * normal);

#ifdef USE_COSINE_WEIGHTED_SAMPLING
        sampled_probability = z / M_PIf;
#else
        sampled_probability = 1.0f / (2.0f * M_PIf);
#endif
    }

    extern "C" __global__ void __raygen__nobounce()
    {
        // [get ids of the current thread]

        const uint3 launchIndex = optixGetLaunchIndex();

        const int i_pixel = launchIndex.y * optixLaunchParams.WIDTH + launchIndex.x;

        // if (i_pixel == 0)
        // {
        //     printf("[DEBUG] calling __raygen__0bounce()\n");
        // }

        // [prepare and share a single chunk buffer for anyhit through a pixel]

        PerRayData per_ray_data;
        IntersectionInfo buffer[ANYHIT_CHUNK_BUFFER_SIZE];
        per_ray_data.buffer = buffer;

        uint32_t per_ray_data_u0, per_ray_data_u1;
        packPointer(&per_ray_data, per_ray_data_u0, per_ray_data_u1);

        // [get camera origin and direction]

        const float3 camera_ray_origin = optixLaunchParams.rays_origins[i_pixel];
        const float3 camera_ray_direction = optixLaunchParams.rays_directions[i_pixel];

        // [shoot rays from camera and use trace_forth() once for all paths]

        int camera_Hitcount;
        float camera_Alpha;

        float camera_Distance;
        float3 camera_Normal;

        float3 camera_Radiance;
        float camera_Emissive;
        float3 camera_Albedo;

        trace_forth(
            camera_ray_origin, camera_ray_direction,
            per_ray_data, per_ray_data_u0, per_ray_data_u1,
            camera_Hitcount, camera_Alpha, camera_Distance, camera_Normal, camera_Radiance, camera_Emissive, camera_Albedo);

        // [return 0-bounce results]

        optixLaunchParams.pixels_hitcounts[i_pixel] = camera_Hitcount;
        optixLaunchParams.pixels_alphas[i_pixel] = camera_Alpha;

        optixLaunchParams.pixels_normals[i_pixel] = camera_Normal;
        optixLaunchParams.pixels_distances[i_pixel] = camera_Distance;

        optixLaunchParams.pixels_radiances[i_pixel] = camera_Radiance;
        optixLaunchParams.pixels_emissives[i_pixel] = camera_Emissive;
        optixLaunchParams.pixels_albedos[i_pixel] = camera_Albedo;
    }

    extern "C" __global__ void __raygen__materialpass_backward()
    {
        const uint3 launchIndex = optixGetLaunchIndex();
        const int i_pixel = launchIndex.y * optixLaunchParams.WIDTH + launchIndex.x;
        PerRayData per_ray_data;
        IntersectionInfo buffer[ANYHIT_CHUNK_BUFFER_SIZE];
        per_ray_data.buffer = buffer;
        uint32_t u0, u1;
        packPointer(&per_ray_data, u0, u1);
        const float3 ray_origin = optixLaunchParams.rays_origins[i_pixel];
        const float3 ray_direction = optixLaunchParams.rays_directions[i_pixel];
        const float3 d_base = optixLaunchParams.d_L_d_pixels_basecolors[i_pixel];
        const float d_rough = optixLaunchParams.d_L_d_pixels_roughnesses[i_pixel];
        const float d_metal = optixLaunchParams.d_L_d_pixels_metallics[i_pixel];
        float tmin = NEAREST_DISTANCE_TO_AVOID_FRONT_SURFELS;
        float transmittance = 1.0f;
        while (true)
        {
            for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; ++i)
                per_ray_data.buffer[i].distance = SCENE_MAX_DISTANCE;
            optixTrace(optixLaunchParams.traversable, ray_origin, ray_direction, tmin,
                       SCENE_MAX_DISTANCE, 0.0f, OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_NONE, 0, 0, 0, u0, u1);
            bool ended = false;
            for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; ++i)
            {
                const float ray_distance = per_ray_data.buffer[i].distance;
                if (ray_distance == SCENE_MAX_DISTANCE) { ended = true; break; }
                const int id = per_ray_data.buffer[i].surfel_id;
                const float2 uv = per_ray_data.buffer[i].surfel_uv;
                const float eta = optixLaunchParams.surfels_opacities[id] *
                                  expf(-0.5f * (uv.x * uv.x + uv.y * uv.y));
                const float weight = transmittance * eta;
                if (weight < 1.0e-6f) continue;
                atomicAdd(&optixLaunchParams.d_L_d_surfels_albedos[id].x, d_base.x * weight);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_albedos[id].y, d_base.y * weight);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_albedos[id].z, d_base.z * weight);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_roughnesses[id], d_rough * weight);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_metallics[id], d_metal * weight);
                transmittance *= (1.0f - eta);
                if (transmittance < TRANSMITTANCE_THRESHOLD) { ended = true; break; }
                tmin = ray_distance + MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION;
            }
            if (ended) break;
            tmin = per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance +
                   MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION;
        }
    }

    __device__ __forceinline__ float3 fresnel_schlick(float cos_theta, const float3 &f0)
    {
        float x = fmaxf(0.0f, 1.0f - cos_theta);
        float factor = x * x * x * x * x;
        return f0 + (make_float3(1.0f) - f0) * factor;
    }

    __device__ __forceinline__ float ggx_distribution(float NoH, float roughness)
    {
        roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
        float a = roughness * roughness;
        float a2 = a * a;
        float denom = NoH * NoH * (a2 - 1.0f) + 1.0f;
        return a2 / fmaxf(M_PIf * denom * denom, 1.0e-7f);
    }

    __device__ __forceinline__ float smith_g1(float NoV, float roughness)
    {
        roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
        float a = roughness * roughness;
        float k = a * 0.5f;
        return NoV / fmaxf(NoV * (1.0f - k) + k, 1.0e-6f);
    }

    __device__ __forceinline__ float3 eval_disney_brdf(
        const float3 &normal, const float3 &wi, const float3 &wo,
        const float3 &basecolor, float metallic, float roughness)
    {
        float NoL = fmaxf(dot(normal, wi), 0.0f);
        float NoV = fmaxf(dot(normal, wo), 0.0f);
        if (NoL <= 0.0f || NoV <= 0.0f)
            return make_float3(0.0f);
        float3 h = normalize(wi + wo);
        float NoH = fmaxf(dot(normal, h), 0.0f);
        float VoH = fmaxf(dot(wo, h), 0.0f);
        float3 f0 = make_float3(0.04f) * (1.0f - metallic) + basecolor * metallic;
        float3 F = fresnel_schlick(VoH, f0);
        float D = ggx_distribution(NoH, roughness);
        float G = smith_g1(NoL, roughness) * smith_g1(NoV, roughness);
        float3 kd = basecolor * (1.0f - metallic) / M_PIf;
        float3 spec = F * (D * G / fmaxf(4.0f * NoL * NoV, 1.0e-6f));
        return kd + spec;
    }

    __device__ __forceinline__ float luminance(const float3 &value)
    {
        return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
    }

    __device__ __forceinline__ float max_component(const float3 &value)
    {
        return fmaxf(value.x, fmaxf(value.y, value.z));
    }

    __device__ __forceinline__ void make_orthonormal_basis(
        const float3 &normal, float3 &tangent, float3 &bitangent)
    {
        const float3 helper = fabsf(normal.z) < 0.999f
                                  ? make_float3(0.0f, 0.0f, 1.0f)
                                  : make_float3(0.0f, 1.0f, 0.0f);
        tangent = normalize(cross(helper, normal));
        bitangent = cross(normal, tangent);
    }

    __device__ __forceinline__ float3 to_world(
        const float3 &local, const float3 &normal)
    {
        float3 tangent, bitangent;
        make_orthonormal_basis(normal, tangent, bitangent);
        return normalize(local.x * tangent + local.y * bitangent + local.z * normal);
    }

    __device__ __forceinline__ float disney_specular_probability(
        const float3 &basecolor, float metallic)
    {
        metallic = fminf(fmaxf(metallic, 0.0f), 1.0f);
        const float3 f0 = make_float3(0.04f) * (1.0f - metallic) + basecolor * metallic;
        const float3 kd = basecolor * (1.0f - metallic);
        const float denominator = luminance(f0) + luminance(kd);
        const float probability = denominator > 1.0e-8f ? luminance(f0) / denominator : 0.5f;
        return fminf(fmaxf(probability, 0.1f), 0.9f);
    }

    __device__ __forceinline__ float cosine_hemisphere_pdf(
        const float3 &normal, const float3 &wi)
    {
        return fmaxf(dot(normal, wi), 0.0f) / M_PIf;
    }

    __device__ __forceinline__ float ggx_reflection_pdf(
        const float3 &normal, const float3 &wi, const float3 &wo, float roughness)
    {
        const float NoL = dot(normal, wi);
        const float NoV = dot(normal, wo);
        if (NoL <= 0.0f || NoV <= 0.0f)
            return 0.0f;
        const float3 half_vector = normalize(wi + wo);
        const float NoH = fmaxf(dot(normal, half_vector), 0.0f);
        const float VoH = fmaxf(dot(wo, half_vector), 0.0f);
        if (VoH <= 0.0f)
            return 0.0f;
        return ggx_distribution(NoH, roughness) * NoH / fmaxf(4.0f * VoH, 1.0e-7f);
    }

    __device__ __forceinline__ float disney_mixture_pdf(
        const float3 &normal, const float3 &wi, const float3 &wo,
        const float3 &basecolor, float metallic, float roughness)
    {
        const float p_spec = disney_specular_probability(basecolor, metallic);
        return (1.0f - p_spec) * cosine_hemisphere_pdf(normal, wi) +
               p_spec * ggx_reflection_pdf(normal, wi, wo, roughness);
    }

    __device__ bool sample_disney_brdf(
        const float3 &normal, const float3 &wo,
        const float3 &basecolor, float metallic, float roughness,
        curandState &curand_state, float3 &wi, float &pdf)
    {
        const float p_spec = disney_specular_probability(basecolor, metallic);
        const float choose = curand_uniform(&curand_state);
        const float u1 = fminf(curand_uniform(&curand_state), 0.99999994f);
        const float u2 = curand_uniform(&curand_state);

        if (choose < p_spec)
        {
            roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
            const float alpha = roughness * roughness;
            const float tan_theta2 = alpha * alpha * u1 / fmaxf(1.0f - u1, 1.0e-7f);
            const float cos_theta = rsqrtf(1.0f + tan_theta2);
            const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
            const float phi = 2.0f * M_PIf * u2;
            const float3 half_vector = to_world(
                make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta), normal);
            const float VoH = dot(wo, half_vector);
            if (VoH <= 0.0f)
            {
                pdf = 0.0f;
                return false;
            }
            wi = normalize(2.0f * VoH * half_vector - wo);
        }
        else
        {
            const float radius = sqrtf(u1);
            const float phi = 2.0f * M_PIf * u2;
            wi = to_world(make_float3(
                              radius * cosf(phi), radius * sinf(phi),
                              sqrtf(fmaxf(0.0f, 1.0f - u1))),
                          normal);
        }

        pdf = disney_mixture_pdf(normal, wi, wo, basecolor, metallic, roughness);
        return dot(normal, wi) > 0.0f && isfinite(pdf) && pdf > 1.0e-7f;
    }

    __device__ __forceinline__ float power_heuristic(float pdf_a, float pdf_b)
    {
        const float a2 = pdf_a * pdf_a;
        const float b2 = pdf_b * pdf_b;
        return (a2 + b2) > 0.0f ? a2 / (a2 + b2) : 0.0f;
    }

    __device__ __forceinline__ void surfel_frame(
        int surfel_id, float3 &tangent, float3 &bitangent, float3 &normal)
    {
        const float4 q = optixLaunchParams.surfels_quaternions[surfel_id];
        const float r = q.x, x = q.y, y = q.z, z = q.w;
        tangent = make_float3(
            1.0f - 2.0f * (y * y + z * z),
            2.0f * (x * y + r * z),
            2.0f * (x * z - r * y));
        bitangent = make_float3(
            2.0f * (x * y - r * z),
            1.0f - 2.0f * (x * x + z * z),
            2.0f * (y * z + r * x));
        normal = make_float3(
            2.0f * (x * z + r * y),
            2.0f * (y * z - r * x),
            1.0f - 2.0f * (x * x + y * y));
        tangent = normalize(tangent);
        bitangent = normalize(bitangent);
        normal = normalize(normal);
    }

    __device__ float trace_shadow_transmittance(
        const float3 &ray_origin, const float3 &ray_direction, float maximum_distance,
        PerRayData &per_ray_data, uint32_t &payload_u0, uint32_t &payload_u1)
    {
        if (maximum_distance <= MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION)
            return 0.0f;

        // A shadow segment starts on one Gaussian surface and ends on another.
        // Using the generic intersection epsilon here makes the overlapping
        // Gaussians around both endpoints shadow themselves. Match secondary
        // rays' surface clearance, while retaining an interior segment for
        // short light connections.
        const float endpoint_margin = fminf(
            NEAREST_DISTANCE_TO_AVOID_FRONT_SURFELS,
            maximum_distance * 0.25f);
        const float tmax = maximum_distance - endpoint_margin;
        if (tmax <= endpoint_margin)
            return 1.0f;

        float transmittance = 1.0f;
        float tmin = endpoint_margin;
        bool ended = false;
        while (!ended && tmin < tmax)
        {
            for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; ++i)
                per_ray_data.buffer[i].distance = SCENE_MAX_DISTANCE;
            optixTrace(
                optixLaunchParams.traversable, ray_origin, ray_direction,
                tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                0, 0, 0, payload_u0, payload_u1);
            for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; ++i)
            {
                const float distance = per_ray_data.buffer[i].distance;
                if (distance == SCENE_MAX_DISTANCE)
                {
                    ended = true;
                    break;
                }
                const int id = per_ray_data.buffer[i].surfel_id;
                const float2 uv = per_ray_data.buffer[i].surfel_uv;
                const float eta = fminf(fmaxf(
                    optixLaunchParams.surfels_opacities[id] *
                        expf(-0.5f * (uv.x * uv.x + uv.y * uv.y)),
                    0.0f), 1.0f);
                transmittance *= 1.0f - eta;
                if (transmittance < TRANSMITTANCE_THRESHOLD)
                {
                    ended = true;
                    break;
                }
            }
            if (!ended)
                tmin = per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance +
                       MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION;
        }
        return fminf(fmaxf(transmittance, 0.0f), 1.0f);
    }

    __device__ bool sample_emitter(
        const float3 &surface_position, curandState &curand_state,
        float3 &wi, float3 &emitted_radiance, float &distance,
        float &solid_angle_pdf)
    {
        if (optixLaunchParams.emissive_surfels_count <= 0)
            return false;
        const float cdf_sample = fminf(curand_uniform(&curand_state), 0.99999994f);
        int low = 0;
        int high = optixLaunchParams.emissive_surfels_count - 1;
        while (low < high)
        {
            const int middle = (low + high) / 2;
            if (cdf_sample <= optixLaunchParams.emissive_surfels_proportions_cdfs[middle])
                high = middle;
            else
                low = middle + 1;
        }
        const int emitter_index = low;
        const int surfel_id = optixLaunchParams.emissive_surfels_ids[emitter_index];
        const float selection_pdf = optixLaunchParams.emissive_surfels_proportions_pdfs[emitter_index];

        const float radial_sample = fminf(curand_uniform(&curand_state), 0.99999994f);
        const float angular_sample = curand_uniform(&curand_state);
        const float gaussian_z = 1.0f - expf(-4.5f);
        const float radius = sqrtf(-2.0f * logf(fmaxf(1.0f - radial_sample * gaussian_z, 1.0e-7f)));
        const float phi = 2.0f * M_PIf * angular_sample;
        const float u = radius * cosf(phi);
        const float v = radius * sinf(phi);

        float3 tangent, bitangent, light_normal;
        surfel_frame(surfel_id, tangent, bitangent, light_normal);
        const float2 scale = optixLaunchParams.surfels_scales[surfel_id];
        const float3 light_position = optixLaunchParams.surfels_positions[surfel_id] +
                                      tangent * (u * scale.x) + bitangent * (v * scale.y);
        const float3 delta = light_position - surface_position;
        const float distance2 = dot(delta, delta);
        if (distance2 <= 1.0e-10f)
            return false;
        distance = sqrtf(distance2);
        wi = delta / distance;
        const float light_cosine = fabsf(dot(light_normal, -wi));
        if (light_cosine <= 1.0e-7f)
            return false;

        const float area_normalization = 2.0f * M_PIf * fabsf(scale.x * scale.y) * gaussian_z;
        const float area_pdf = expf(-0.5f * radius * radius) /
                               fmaxf(area_normalization, 1.0e-12f);
        solid_angle_pdf = selection_pdf * area_pdf * distance2 / light_cosine;
        const float emission_alpha = fminf(fmaxf(
            optixLaunchParams.surfels_emissives[surfel_id] *
                optixLaunchParams.surfels_opacities[surfel_id] *
                expf(-0.5f * radius * radius),
            0.0f), 1.0f);
        emitted_radiance = optixLaunchParams.surfels_radiances[surfel_id] * emission_alpha;
        return solid_angle_pdf > 1.0e-12f && isfinite(solid_angle_pdf);
    }

    __device__ float emitter_hit_solid_angle_pdf(
        int surfel_id, const float2 &uv, float distance, const float3 &ray_direction)
    {
        if (surfel_id < 0 || optixLaunchParams.surfels_emissive_selection_pdfs == nullptr)
            return 0.0f;
        const float selection_pdf = optixLaunchParams.surfels_emissive_selection_pdfs[surfel_id];
        if (selection_pdf <= 0.0f)
            return 0.0f;
        float3 tangent, bitangent, light_normal;
        surfel_frame(surfel_id, tangent, bitangent, light_normal);
        const float light_cosine = fabsf(dot(light_normal, -ray_direction));
        if (light_cosine <= 1.0e-7f)
            return 0.0f;
        const float2 scale = optixLaunchParams.surfels_scales[surfel_id];
        const float gaussian_z = 1.0f - expf(-4.5f);
        const float area_normalization = 2.0f * M_PIf * fabsf(scale.x * scale.y) * gaussian_z;
        const float area_pdf = expf(-0.5f * (uv.x * uv.x + uv.y * uv.y)) /
                               fmaxf(area_normalization, 1.0e-12f);
        return selection_pdf * area_pdf * distance * distance / light_cosine;
    }

    extern "C" __global__ void __raygen__materialpass()
    {
        const uint3 launchIndex = optixGetLaunchIndex();
        const int i_pixel = launchIndex.y * optixLaunchParams.WIDTH + launchIndex.x;

        PerRayData per_ray_data;
        IntersectionInfo buffer[ANYHIT_CHUNK_BUFFER_SIZE];
        per_ray_data.buffer = buffer;
        uint32_t u0, u1;
        packPointer(&per_ray_data, u0, u1);

        int hitcount;
        float alpha, distance, emissive_dummy, roughness, metallic;
        float3 normal, radiance, basecolor;
        trace_forth_with_material(
            optixLaunchParams.rays_origins[i_pixel],
            optixLaunchParams.rays_directions[i_pixel],
            per_ray_data, u0, u1,
            hitcount, alpha, distance, normal, radiance, emissive_dummy,
            basecolor, roughness, metallic);

        optixLaunchParams.pixels_basecolors[i_pixel] = basecolor;
        optixLaunchParams.pixels_roughnesses[i_pixel] = roughness;
        optixLaunchParams.pixels_metallics[i_pixel] = metallic;
    }

    extern "C" __global__ void __raygen__singlebounce()
    {
        // [get ids of the current thread]

        const uint3 launchIndex = optixGetLaunchIndex();

        const int i_pixel = launchIndex.y * optixLaunchParams.WIDTH + launchIndex.x;

        // if (i_pixel == 0)
        // {
        //     printf("[DEBUG] calling __raygen__1bounce()\n");
        // }

        // [prepare and share the random generator through a pixel; each pixel has a different generator]

        // https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
        curandState curand_state;
        curand_init(0, i_pixel, 0, &curand_state);

        // [prepare and share a single chunk buffer for anyhit through a pixel]

        PerRayData per_ray_data;
        IntersectionInfo buffer[ANYHIT_CHUNK_BUFFER_SIZE];
        per_ray_data.buffer = buffer;

        uint32_t per_ray_data_u0, per_ray_data_u1;
        packPointer(&per_ray_data, per_ray_data_u0, per_ray_data_u1);

        // [get camera origin and direction]

        const float3 camera_ray_origin = optixLaunchParams.rays_origins[i_pixel];
        const float3 camera_ray_direction = optixLaunchParams.rays_directions[i_pixel];

        // [shoot rays from camera and use trace_forth() once for all paths]

        int camera_Hitcount;
        float camera_Alpha;

        float camera_Distance;
        float3 camera_Normal;

        float3 camera_Radiance;
        float camera_Emissive;
        float3 camera_Albedo;
        float camera_Roughness;
        float camera_Metallic;
        float3 camera_emission_radiance;

        trace_forth_with_material(
            camera_ray_origin, camera_ray_direction,
            per_ray_data, per_ray_data_u0, per_ray_data_u1,
            camera_Hitcount, camera_Alpha, camera_Distance, camera_Normal, camera_Radiance, camera_Emissive, camera_Albedo,
            camera_Roughness, camera_Metallic,
            nullptr, nullptr, nullptr, &camera_emission_radiance);

        // [save pixels_albedos for backward pass]

        optixLaunchParams.pixels_albedos[i_pixel] = camera_Albedo;

        // [path tracing]

        // [start path tracing]

        // Emission and reflection are independent material lobes.

        // to accumulate all spp
        float3 path_tracing_radiance = make_float3(0.0f, 0.0f, 0.0f);

        // avoid nan
        if (camera_Hitcount != 0)
        {
            // add radiances from all valid paths
            for (int i_spp = 0; i_spp < optixLaunchParams.SPP; ++i_spp)
            {
                // [always start from the camera rays intersections, that not hit the emissives]

                // start from the 0-bounce intersection point
                float3 ray_origin = camera_ray_origin + camera_Distance * camera_ray_direction;
                // TODO currently, ray_direction_outgoing is not used for material samplng since diffuse material is used
                float3 ray_direction_outgoing = -camera_ray_direction;
                float3 ray_direction_incident;

                int intersection_Hitcount = camera_Hitcount;
                float intersection_Alpha = camera_Alpha;

                float intersection_Distance = camera_Distance;
                float3 intersection_Normal = normalize(camera_Normal);

                float3 intersection_Radiance = camera_Radiance;
                float intersection_Emissive = camera_Emissive;
                float3 intersection_Albedo = camera_Albedo;
                float intersection_Roughness = camera_Roughness;
                float intersection_Metallic = camera_Metallic;

                // multiplication of albedos along the path
                // calculate the currect intersected material
                float3 path_throughput = make_float3(1.0f, 1.0f, 1.0f);

                // re-trace the ray from non-emissive intersection point (camera intersection or after bouncing)
                // 1-bounce into radiant scene
                {
                    float probability;
                    if (!sample_disney_brdf(
                            intersection_Normal, ray_direction_outgoing,
                            intersection_Albedo, intersection_Metallic, intersection_Roughness,
                            curand_state, ray_direction_incident, probability))
                        continue;

                    // [calculate throughput at previous intersection point]

                    // previous: Albedo, Normal
                    // new: ray_direction_incident, probability
                    path_throughput *= eval_disney_brdf(
                        intersection_Normal, ray_direction_incident, ray_direction_outgoing,
                        intersection_Albedo, intersection_Metallic, intersection_Roughness)
                        * dot(intersection_Normal, ray_direction_incident) / fmaxf(probability, 1.0e-6f);

                    // directly update the intersection values
                    trace_forth_with_material(
                        ray_origin, ray_direction_incident,
                        per_ray_data, per_ray_data_u0, per_ray_data_u1,
                        intersection_Hitcount, intersection_Alpha, intersection_Distance, intersection_Normal, intersection_Radiance, intersection_Emissive, intersection_Albedo,
                        intersection_Roughness, intersection_Metallic);

                    // new: Emissive, Radiance
                    path_tracing_radiance += path_throughput * intersection_Radiance;
                }
            }

            // divide spp
            path_tracing_radiance /= (float)optixLaunchParams.SPP;
        }

        path_tracing_radiance += camera_emission_radiance;

        // return path tracing result
        optixLaunchParams.pixels_rendering_radiances[i_pixel] = path_tracing_radiance;
        optixLaunchParams.pixels_d_rendering_radiances_d_P[i_pixel] = path_tracing_radiance;
    }

    extern "C" __global__ void __raygen__pathtracing()
    {
        const uint3 launchIndex = optixGetLaunchIndex();
        const int i_pixel = launchIndex.y * optixLaunchParams.WIDTH + launchIndex.x;
        curandState curand_state;
        curand_init(0, i_pixel, 0, &curand_state);
        PerRayData per_ray_data;
        IntersectionInfo buffer[ANYHIT_CHUNK_BUFFER_SIZE];
        per_ray_data.buffer = buffer;
        uint32_t per_ray_data_u0, per_ray_data_u1;
        packPointer(&per_ray_data, per_ray_data_u0, per_ray_data_u1);
        const float3 camera_ray_origin = optixLaunchParams.rays_origins[i_pixel];
        const float3 camera_ray_direction = optixLaunchParams.rays_directions[i_pixel];
        int camera_Hitcount;
        float camera_Alpha;
        float camera_Distance;
        float3 camera_Normal;
        float3 camera_Radiance;
        float camera_Emissive;
        float3 camera_Albedo;
        float camera_Roughness;
        float camera_Metallic;
        int camera_dominant_id;
        float2 camera_dominant_uv;
        float camera_dominant_distance;
        float3 camera_emission_radiance;
        trace_forth_with_material(
            camera_ray_origin, camera_ray_direction,
            per_ray_data, per_ray_data_u0, per_ray_data_u1,
            camera_Hitcount, camera_Alpha, camera_Distance, camera_Normal, camera_Radiance, camera_Emissive, camera_Albedo,
            camera_Roughness, camera_Metallic,
            &camera_dominant_id, &camera_dominant_uv, &camera_dominant_distance,
            &camera_emission_radiance);

        float3 emission_radiance = make_float3(0.0f);
        float3 direct_radiance = make_float3(0.0f);
        float3 indirect_radiance = make_float3(0.0f);
        float shadow_visibility = 0.0f;

        if (camera_Hitcount > 0)
        {
            emission_radiance = camera_emission_radiance;
            const float camera_surface_distance = camera_dominant_distance > 0.0f
                                                      ? camera_dominant_distance
                                                      : camera_Distance / fmaxf(camera_Alpha, 1.0e-6f);
            const int light_samples = max(optixLaunchParams.pbr_pt_light_samples, 1);

            for (int i_spp = 0; i_spp < optixLaunchParams.SPP; ++i_spp)
            {
                float3 surface_position = camera_ray_origin + camera_surface_distance * camera_ray_direction;
                float3 outgoing = -camera_ray_direction;
                float3 normal = normalize(camera_Normal);
                float3 basecolor = camera_Albedo;
                float roughness = camera_Roughness;
                float metallic = camera_Metallic;
                float3 throughput = make_float3(1.0f);

                for (int bounce = 0; bounce < optixLaunchParams.BOUNCE_LIMIT; ++bounce)
                {
                    if (optixLaunchParams.pbr_pt_use_nee && optixLaunchParams.emissive_surfels_count > 0)
                    {
                        for (int light_sample = 0; light_sample < light_samples; ++light_sample)
                        {
                            float3 light_direction, emitted;
                            float light_distance, light_pdf;
                            float visibility = 0.0f;
                            if (sample_emitter(
                                    surface_position, curand_state, light_direction, emitted,
                                    light_distance, light_pdf))
                            {
                                const float NoL = fmaxf(dot(normal, light_direction), 0.0f);
                                if (NoL > 0.0f)
                                {
                                    visibility = trace_shadow_transmittance(
                                        surface_position, light_direction, light_distance,
                                        per_ray_data, per_ray_data_u0, per_ray_data_u1);
                                    const float3 brdf = eval_disney_brdf(
                                        normal, light_direction, outgoing,
                                        basecolor, metallic, roughness);
                                    const float bsdf_pdf = disney_mixture_pdf(
                                        normal, light_direction, outgoing,
                                        basecolor, metallic, roughness);
                                    const float mis_weight = optixLaunchParams.pbr_pt_use_mis
                                                                 ? power_heuristic(light_pdf, bsdf_pdf)
                                                                 : 1.0f;
                                    const float3 contribution = throughput * brdf * emitted *
                                                                (NoL * visibility * mis_weight /
                                                                 fmaxf(light_pdf * light_samples, 1.0e-12f));
                                    if (bounce == 0)
                                        direct_radiance += contribution;
                                    else
                                        indirect_radiance += contribution;
                                }
                            }
                            if (bounce == 0)
                                shadow_visibility += visibility / (float)light_samples;
                        }
                    }

                    float3 incident;
                    float bsdf_pdf;
                    bool valid_sample;
                    if (optixLaunchParams.pbr_pt_use_disney_sampling)
                    {
                        valid_sample = sample_disney_brdf(
                            normal, outgoing, basecolor, metallic, roughness,
                            curand_state, incident, bsdf_pdf);
                    }
                    else
                    {
                        sample_upper_hemisphere_direction(normal, curand_state, incident, bsdf_pdf);
                        valid_sample = bsdf_pdf > 1.0e-7f;
                    }
                    if (!valid_sample)
                        break;

                    const float NoL = fmaxf(dot(normal, incident), 0.0f);
                    const float3 brdf = eval_disney_brdf(
                        normal, incident, outgoing, basecolor, metallic, roughness);
                    throughput *= brdf * (NoL / fmaxf(bsdf_pdf, 1.0e-7f));
                    if (!isfinite(throughput.x) || !isfinite(throughput.y) ||
                        !isfinite(throughput.z) || max_component(throughput) <= 0.0f)
                        break;

                    int hitcount;
                    float alpha, distance, emissive;
                    float3 hit_normal, radiance, hit_basecolor;
                    float hit_roughness, hit_metallic;
                    int dominant_id;
                    float2 dominant_uv;
                    float dominant_distance;
                    float3 hit_emission;
                    trace_forth_with_material(
                        surface_position, incident,
                        per_ray_data, per_ray_data_u0, per_ray_data_u1,
                        hitcount, alpha, distance, hit_normal, radiance, emissive, hit_basecolor,
                        hit_roughness, hit_metallic,
                        &dominant_id, &dominant_uv, &dominant_distance,
                        &hit_emission);
                    if (hitcount == 0)
                        break;

                    if (max_component(hit_emission) > 0.0f)
                    {
                        float mis_weight = 1.0f;
                        if (optixLaunchParams.pbr_pt_use_nee && optixLaunchParams.pbr_pt_use_mis)
                        {
                            const float light_pdf = emitter_hit_solid_angle_pdf(
                                dominant_id, dominant_uv, dominant_distance, incident);
                            mis_weight = power_heuristic(bsdf_pdf, light_pdf);
                        }
                        indirect_radiance += throughput * hit_emission * mis_weight;
                    }

                    const float surface_distance = dominant_distance > 0.0f
                                                       ? dominant_distance
                                                       : distance / fmaxf(alpha, 1.0e-6f);
                    surface_position += surface_distance * incident;
                    outgoing = -incident;
                    normal = normalize(hit_normal);
                    basecolor = hit_basecolor;
                    roughness = hit_roughness;
                    metallic = hit_metallic;

                    if (optixLaunchParams.pbr_pt_use_russian_roulette &&
                        bounce >= optixLaunchParams.pbr_pt_rr_start_bounce)
                    {
                        const float survival = fminf(fmaxf(max_component(throughput), 0.05f), 0.95f);
                        if (curand_uniform(&curand_state) > survival)
                            break;
                        throughput /= survival;
                    }
                }
            }

            const float inv_spp = 1.0f / fmaxf((float)optixLaunchParams.SPP, 1.0f);
            direct_radiance *= inv_spp;
            indirect_radiance *= inv_spp;
            shadow_visibility *= inv_spp;
        }

        const float3 total_radiance = emission_radiance + direct_radiance + indirect_radiance;
        optixLaunchParams.pixels_rendering_radiances[i_pixel] = total_radiance;
        optixLaunchParams.pixels_direct_radiances[i_pixel] = direct_radiance;
        optixLaunchParams.pixels_indirect_radiances[i_pixel] = indirect_radiance;
        optixLaunchParams.pixels_emission_radiances[i_pixel] = emission_radiance;
        optixLaunchParams.pixels_shadow_visibilities[i_pixel] =
            fminf(fmaxf(shadow_visibility, 0.0f), 1.0f);
    }

    __device__ void trace_forth_backward_nobounce(
        uint3 launchIndex,

        // [input]

        const float3 &ray_origin, const float3 &ray_direction,

        PerRayData per_ray_data,
        uint32_t &per_ray_data_u0,
        uint32_t &per_ray_data_u1,

        // [output]

        int &Hitcount,
        float &Alpha,

        float &Distance,
        float3 &Normal,

        float3 &Radiance,
        float &Emissive,
        float3 &Albedo,

        // [backward input]
        const float3 &backward_pixel_radiance,
        const float &backward_pixel_emissive,
        const float &backward_pixel_alpha,
        const float3 &backward_pixel_normal,
        const float &backward_pixel_distance,

        const float3 &d_L_d_pixel_radiance,
        const float &d_L_d_pixel_emissive,
        const float &d_L_d_pixel_alpha,
        const float3 &d_L_d_pixel_normal,
        const float &d_L_d_pixel_distance)
    {
        // avoid surfels near camera
        float distance_to_start_tracing_ray = NEAREST_DISTANCE_TO_AVOID_FRONT_SURFELS;

        Hitcount = 0;

        float Transmittance = 1.0f;
        float Transmittance_tobe = 1.0f;

        Distance = 0.0f;
        Normal = make_float3(0.0f, 0.0f, 0.0f);

        Radiance = make_float3(0.0f, 0.0f, 0.0f);
        Emissive = 0.0f;
        Albedo = make_float3(0.0f, 0.0f, 0.0f);

        // tracing (0-bounce) along the ray
        while (1)
        {
            // initialized chunk buffer
            for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; i++)
                per_ray_data.buffer[i].distance = SCENE_MAX_DISTANCE;

            // trace a chunk and collect next k closest intersected surfels along the ray
            optixTrace(optixLaunchParams.traversable,
                       ray_origin,
                       ray_direction,
                       distance_to_start_tracing_ray, // tmin
                       SCENE_MAX_DISTANCE,            // tmax
                       0.0f,                          // rayTime
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_NONE,
                       0,                   // SBT offset
                       0,                   // SBT stride
                       0,                   // missSBTIndex
                       per_ray_data_u0, per_ray_data_u1);

            // accumulate the chunk
            for (int i_surfel_in_chunk = 0; i_surfel_in_chunk < ANYHIT_CHUNK_BUFFER_SIZE; ++i_surfel_in_chunk)
            {
                // [get per_ray_data from buffer]

                const float ray_distance = per_ray_data.buffer[i_surfel_in_chunk].distance;
                if (ray_distance == SCENE_MAX_DISTANCE)
                    break;

                const int surfel_id = per_ray_data.buffer[i_surfel_in_chunk].surfel_id;
                const float2 surfel_uv = per_ray_data.buffer[i_surfel_in_chunk].surfel_uv;

                // [get surfel properties by surfel_id]

                const float surfel_opacity = optixLaunchParams.surfels_opacities[surfel_id];
                const float4 surfel_quaternion = optixLaunchParams.surfels_quaternions[surfel_id];
                const float2 surfel_scale = optixLaunchParams.surfels_scales[surfel_id];
                const float3 surfel_position = optixLaunchParams.surfels_positions[surfel_id];

                // normal

                const float surfel_quaternion_r = surfel_quaternion.x;
                const float surfel_quaternion_x = surfel_quaternion.y;
                const float surfel_quaternion_y = surfel_quaternion.z;
                const float surfel_quaternion_z = surfel_quaternion.w;

                // the first column, tu
                float3 surfel_tu = make_float3(1.0f - 2.0f * (surfel_quaternion_y * surfel_quaternion_y + surfel_quaternion_z * surfel_quaternion_z),
                                               2.0f * (surfel_quaternion_x * surfel_quaternion_y + surfel_quaternion_r * surfel_quaternion_z),
                                               2.0f * (surfel_quaternion_x * surfel_quaternion_z - surfel_quaternion_r * surfel_quaternion_y));
                // the third column, tv
                float3 surfel_tv = make_float3(2.0f * (surfel_quaternion_x * surfel_quaternion_y - surfel_quaternion_r * surfel_quaternion_z),
                                               1.0f - 2.0f * (surfel_quaternion_x * surfel_quaternion_x + surfel_quaternion_z * surfel_quaternion_z),
                                               2.0f * (surfel_quaternion_y * surfel_quaternion_z + surfel_quaternion_r * surfel_quaternion_x));
                // the third column, n
                float3 surfel_normal = make_float3(2.0f * (surfel_quaternion_x * surfel_quaternion_z + surfel_quaternion_r * surfel_quaternion_y),
                                                   2.0f * (surfel_quaternion_y * surfel_quaternion_z - surfel_quaternion_r * surfel_quaternion_x),
                                                   1.0f - 2.0f * (surfel_quaternion_x * surfel_quaternion_x + surfel_quaternion_y * surfel_quaternion_y));

                // if (abs(dot(surfel_normal, ray_direction)) < 1e-6f)
                // {
                //     break;
                // }

                // [accumulate start]

                const float gaussian_value = exp(-0.5f * (surfel_uv.x * surfel_uv.x + surfel_uv.y * surfel_uv.y));
                const float eta = surfel_opacity * gaussian_value;
                const float weight = Transmittance * eta;

                if (weight < 1.0e-6f)
                {
                    continue;
                }

                Transmittance_tobe = Transmittance * (1.0f - eta);
                if (Transmittance_tobe < TRANSMITTANCE_THRESHOLD)
                {
                    break;
                }

                // this is the distance from the ray_origin_safe
                Distance += weight * ray_distance;
                Normal += weight * ((dot(surfel_normal, ray_direction) > 0.0f) ? -surfel_normal : surfel_normal);

                Radiance += weight * optixLaunchParams.surfels_radiances[surfel_id];
                Emissive += weight * optixLaunchParams.surfels_emissives[surfel_id];
                Albedo += weight * optixLaunchParams.surfels_albedos[surfel_id];

                // [backward start]

                // radiances
                atomicAdd(&optixLaunchParams.d_L_d_surfels_radiances[surfel_id].x, d_L_d_pixel_radiance.x * weight);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_radiances[surfel_id].y, d_L_d_pixel_radiance.y * weight);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_radiances[surfel_id].z, d_L_d_pixel_radiance.z * weight);

                // emissives
                atomicAdd(&optixLaunchParams.d_L_d_surfels_emissives[surfel_id], d_L_d_pixel_emissive * weight);

                const float d_V_Emissive_d_eta = Transmittance * optixLaunchParams.surfels_emissives[surfel_id] - (backward_pixel_emissive - Emissive) / (1.0f - eta);
                const float3 d_V_Radiance_d_eta = Transmittance * optixLaunchParams.surfels_radiances[surfel_id] - (backward_pixel_radiance - Radiance) / (1.0f - eta);
                const float d_V_alpha_d_eta = (1.0f - backward_pixel_alpha) / (1.0f - eta);
                const float3 d_V_Normal_d_eta = Transmittance * ((dot(surfel_normal, ray_direction) > 0.0f) ? -surfel_normal : surfel_normal) - (backward_pixel_normal - Normal) / (1.0f - eta);
                const float d_V_Distance_d_eta = Transmittance * ray_distance - (backward_pixel_distance - Distance) / (1.0f - eta);

                // CHOICE: propagate to all properties
                const float d_L_d_eta = (d_L_d_pixel_emissive * d_V_Emissive_d_eta + dot(d_L_d_pixel_radiance, d_V_Radiance_d_eta) + d_L_d_pixel_alpha * d_V_alpha_d_eta + dot(d_L_d_pixel_normal, d_V_Normal_d_eta) + d_L_d_pixel_distance * d_V_Distance_d_eta);

                if (isnan(d_L_d_eta))
                {
                    printf("d_L_d_eta isnan\n");
                    printf("launchIndex=(%d,%d,%d)\n", launchIndex.x, launchIndex.y, launchIndex.z);

                    printf("Transmittance=%.20f, ray_distance=%f, backward_pixel_distance=%f, backward_pixel_alpha=%.20f, Distance=%f, eta=%.20f\n", Transmittance, ray_distance, backward_pixel_distance, backward_pixel_alpha, Distance, eta);
                    printf("surfel_opacity=%f, gaussian_value=%f, weight=%f, i_surfel_in_chunk=%d\n", surfel_opacity, gaussian_value, weight, i_surfel_in_chunk);
                }

                if (isnan(d_L_d_pixel_distance))
                {
                    printf("d_L_d_pixel_distance isnan\n");
                }
                if (isnan(d_V_Distance_d_eta))
                {
                    printf("d_V_Distance_d_eta isnan\n");
                }

                // opacities
                // two values (V) and A
                const float d_L_d_surfel_opacity = d_L_d_eta * gaussian_value;
                atomicAdd(&optixLaunchParams.d_L_d_surfels_opacities[surfel_id], d_L_d_surfel_opacity);

                // general
                const float d_eta_d_g = surfel_opacity;
                const float d_g_d_uo = -surfel_uv.x * gaussian_value;
                const float d_g_d_vo = -surfel_uv.y * gaussian_value;

                // scales
                const float d_uo_d_su = -surfel_uv.x / surfel_scale.x;
                const float d_vo_d_sv = -surfel_uv.y / surfel_scale.y;
                const float d_L_d_surfel_scale_u = d_L_d_eta * d_eta_d_g * d_g_d_uo * d_uo_d_su;
                const float d_L_d_surfel_scale_v = d_L_d_eta * d_eta_d_g * d_g_d_vo * d_vo_d_sv;
                atomicAdd(&optixLaunchParams.d_L_d_surfels_scales[surfel_id].x, d_L_d_surfel_scale_u);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_scales[surfel_id].y, d_L_d_surfel_scale_v);

                // positions
                const float3 d_uo_d_p = (dot(ray_direction, surfel_tu) / dot(ray_direction, surfel_normal) * surfel_normal - surfel_tu) / surfel_scale.x;
                const float3 d_vo_d_p = (dot(ray_direction, surfel_tv) / dot(ray_direction, surfel_normal) * surfel_normal - surfel_tv) / surfel_scale.y;
                const float3 d_L_d_surfel_position = d_L_d_eta * d_eta_d_g * (d_g_d_uo * d_uo_d_p + d_g_d_vo * d_vo_d_p) + /* distance backward */ d_L_d_pixel_distance * weight * surfel_normal / dot(ray_direction, surfel_normal);

                // if (isnan(d_L_d_surfel_position.x) || isnan(d_L_d_surfel_position.z) || isnan(d_L_d_surfel_position.z))
                // {
                //     printf("d_L_d_surfel_position (%f,%f,%f) is nan, dot=%.20f\n", d_L_d_surfel_position.x, d_L_d_surfel_position.y, d_L_d_surfel_position.z), dot(ray_direction, surfel_normal);
                // }

                atomicAdd(&optixLaunchParams.d_L_d_surfels_positions[surfel_id].x, d_L_d_surfel_position.x);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_positions[surfel_id].y, d_L_d_surfel_position.y);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_positions[surfel_id].z, d_L_d_surfel_position.z);

                // quaternions
                const float3 op = surfel_position - ray_origin;

                // []
                const float3 d_Bu_d_tu = dot(op, surfel_normal) / dot(ray_direction, surfel_normal) * ray_direction - op;
                const float3 d_Bu_d_tv = d_Bu_d_tu;
                const float3 d_Bu_d_n__general = op / dot(ray_direction, surfel_normal) - dot(op, surfel_normal) / (dot(ray_direction, surfel_normal) * dot(ray_direction, surfel_normal)) * ray_direction;
                const float3 d_Bu_d_n__tu = dot(ray_direction, surfel_tu) * d_Bu_d_n__general;
                const float3 d_Bu_d_n__tv = dot(ray_direction, surfel_tv) * d_Bu_d_n__general;

                // []
                const float3 d_tu_d_q0 = make_float3(0.0f, 2.0f * surfel_quaternion_z, -2.0f * surfel_quaternion_y);
                const float3 d_tv_d_q0 = make_float3(-2.0f * surfel_quaternion_z, 0, 2.0f * surfel_quaternion_x);
                const float3 d_n_d_q0 = make_float3(2.0f * surfel_quaternion_y, -2.0f * surfel_quaternion_x, 0.0f);

                const float3 d_tu_d_q1 = make_float3(0.0f, 2.0f * surfel_quaternion_y, 2.0f * surfel_quaternion_z);
                const float3 d_tv_d_q1 = make_float3(2.0f * surfel_quaternion_y, -4.0f * surfel_quaternion_x, 2.0f * surfel_quaternion_r);
                const float3 d_n_d_q1 = make_float3(2.0f * surfel_quaternion_z, -2.0f * surfel_quaternion_r, -4.0f * surfel_quaternion_x);

                const float3 d_tu_d_q2 = make_float3(-4.0f * surfel_quaternion_y, 2.0f * surfel_quaternion_x, -2.0f * surfel_quaternion_r);
                const float3 d_tv_d_q2 = make_float3(2.0f * surfel_quaternion_x, 0.0f, 2.0f * surfel_quaternion_z);
                const float3 d_n_d_q2 = make_float3(2.0f * surfel_quaternion_r, 2.0f * surfel_quaternion_z, -4.0f * surfel_quaternion_y);

                const float3 d_tu_d_q3 = make_float3(-4.0f * surfel_quaternion_z, 2.0f * surfel_quaternion_r, 2.0f * surfel_quaternion_x);
                const float3 d_tv_d_q3 = make_float3(-2.0f * surfel_quaternion_r, -4.0f * surfel_quaternion_z, 2.0f * surfel_quaternion_y);
                const float3 d_n_d_q3 = make_float3(2.0f * surfel_quaternion_x, 2.0f * surfel_quaternion_y, 0.0f);

                /* distance backward */
                const float3 d_distance_d_n_k = (op - ray_distance * ray_direction) / dot(ray_direction, surfel_normal);

                // []
                const float d_uo_d_q0 = (dot(d_Bu_d_tu, d_tu_d_q0) + dot(d_Bu_d_n__tu, d_n_d_q0)) / surfel_scale.x;
                const float d_vo_d_q0 = (dot(d_Bu_d_tv, d_tv_d_q0) + dot(d_Bu_d_n__tv, d_n_d_q0)) / surfel_scale.y;
                const float d_L_d_q0 = d_L_d_eta * d_eta_d_g * (d_g_d_uo * d_uo_d_q0 + d_g_d_vo * d_vo_d_q0) + dot(d_L_d_pixel_normal * weight, d_n_d_q0) * ((dot(surfel_normal, ray_direction) > 0.0f) ? -1.0f : 1.0f) + /* distance backward */ (d_L_d_pixel_distance * weight * dot(d_distance_d_n_k, d_n_d_q0));

                const float d_uo_d_q1 = (dot(d_Bu_d_tu, d_tu_d_q1) + dot(d_Bu_d_n__tu, d_n_d_q1)) / surfel_scale.x;
                const float d_vo_d_q1 = (dot(d_Bu_d_tv, d_tv_d_q1) + dot(d_Bu_d_n__tv, d_n_d_q1)) / surfel_scale.y;
                const float d_L_d_q1 = d_L_d_eta * d_eta_d_g * (d_g_d_uo * d_uo_d_q1 + d_g_d_vo * d_vo_d_q1) + dot(d_L_d_pixel_normal * weight, d_n_d_q1) * ((dot(surfel_normal, ray_direction) > 0.0f) ? -1.0f : 1.0f) + /* distance backward */ d_L_d_pixel_distance * weight * dot(d_distance_d_n_k, d_n_d_q1);

                const float d_uo_d_q2 = (dot(d_Bu_d_tu, d_tu_d_q2) + dot(d_Bu_d_n__tu, d_n_d_q2)) / surfel_scale.x;
                const float d_vo_d_q2 = (dot(d_Bu_d_tv, d_tv_d_q2) + dot(d_Bu_d_n__tv, d_n_d_q2)) / surfel_scale.y;
                const float d_L_d_q2 = d_L_d_eta * d_eta_d_g * (d_g_d_uo * d_uo_d_q2 + d_g_d_vo * d_vo_d_q2) + dot(d_L_d_pixel_normal * weight, d_n_d_q2) * ((dot(surfel_normal, ray_direction) > 0.0f) ? -1.0f : 1.0f) + /* distance backward */ d_L_d_pixel_distance * weight * dot(d_distance_d_n_k, d_n_d_q2);

                const float d_uo_d_q3 = (dot(d_Bu_d_tu, d_tu_d_q3) + dot(d_Bu_d_n__tu, d_n_d_q3)) / surfel_scale.x;
                const float d_vo_d_q3 = (dot(d_Bu_d_tv, d_tv_d_q3) + dot(d_Bu_d_n__tv, d_n_d_q3)) / surfel_scale.y;
                const float d_L_d_q3 = d_L_d_eta * d_eta_d_g * (d_g_d_uo * d_uo_d_q3 + d_g_d_vo * d_vo_d_q3) + dot(d_L_d_pixel_normal * weight, d_n_d_q3) * ((dot(surfel_normal, ray_direction) > 0.0f) ? -1.0f : 1.0f) + /* distance backward */ d_L_d_pixel_distance * weight * dot(d_distance_d_n_k, d_n_d_q3);

                // []
                const float4 d_L_d_surfel_quaternion = make_float4(d_L_d_q0, d_L_d_q1, d_L_d_q2, d_L_d_q3);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_quaternions[surfel_id].x, d_L_d_surfel_quaternion.x);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_quaternions[surfel_id].y, d_L_d_surfel_quaternion.y);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_quaternions[surfel_id].z, d_L_d_surfel_quaternion.z);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_quaternions[surfel_id].w, d_L_d_surfel_quaternion.w);

                // [accumulate end]

                Hitcount++;
                if (Hitcount >= MAXIMAL_HITCOUNT_ALONG_THE_RAY)
                    break;

                Transmittance = Transmittance_tobe;
            }

            // break again to end the trace_forth(); previous break only end the chunk
            if ((per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance == SCENE_MAX_DISTANCE) || (Transmittance_tobe < TRANSMITTANCE_THRESHOLD) || (Hitcount >= MAXIMAL_HITCOUNT_ALONG_THE_RAY))
                break;

            // update tmin for tracing next chunk
            distance_to_start_tracing_ray = per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance + MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION;
        }

        Alpha = 1.0f - Transmittance;
    }

    extern "C" __global__ void __raygen__nobounce_backward()
    {
        // [get ids of the current thread]

        const uint3 launchIndex = optixGetLaunchIndex();

        const int i_pixel = launchIndex.y * optixLaunchParams.WIDTH + launchIndex.x;

        // if (i_pixel == 0)
        // {
        //     printf("[DEBUG] calling __raygen__0bounce()\n");
        // }

        // [prepare and share a single chunk buffer for anyhit through a pixel]

        PerRayData per_ray_data;
        IntersectionInfo buffer[ANYHIT_CHUNK_BUFFER_SIZE];
        per_ray_data.buffer = buffer;

        uint32_t per_ray_data_u0, per_ray_data_u1;
        packPointer(&per_ray_data, per_ray_data_u0, per_ray_data_u1);

        // [get camera origin and direction]

        const float3 camera_ray_origin = optixLaunchParams.rays_origins[i_pixel];
        const float3 camera_ray_direction = optixLaunchParams.rays_directions[i_pixel];

        // [backward input]

        const float3 backward_pixel_radiance = optixLaunchParams.backward_pixels_radiances[i_pixel];
        const float backward_pixel_emissive = optixLaunchParams.backward_pixels_emissives[i_pixel];
        const float backward_pixel_alpha = optixLaunchParams.backward_pixels_alphas[i_pixel];
        const float3 backward_pixel_normal = optixLaunchParams.backward_pixels_normals[i_pixel];
        const float backward_pixel_distance = optixLaunchParams.backward_pixels_distances[i_pixel];

        const float3 d_L_d_pixel_radiance = optixLaunchParams.d_L_d_pixels_radiances[i_pixel];
        const float d_L_d_pixel_emissive = optixLaunchParams.d_L_d_pixels_emissives[i_pixel];
        const float d_L_d_pixel_alpha = optixLaunchParams.d_L_d_pixels_alphas[i_pixel];
        const float3 d_L_d_pixel_normal = optixLaunchParams.d_L_d_pixels_normals[i_pixel];
        const float d_L_d_pixel_distance = optixLaunchParams.d_L_d_pixels_distances[i_pixel];

        // [shoot rays from camera and use trace_forth() once for all paths]

        int camera_Hitcount;
        float camera_Alpha;

        float camera_Distance;
        float3 camera_Normal;

        float3 camera_Radiance;
        float camera_Emissive;
        float3 camera_Albedo;

        trace_forth_backward_nobounce(
            launchIndex,
            camera_ray_origin, camera_ray_direction,
            per_ray_data, per_ray_data_u0, per_ray_data_u1,
            camera_Hitcount, camera_Alpha, camera_Distance, camera_Normal, camera_Radiance, camera_Emissive, camera_Albedo,
            // [backward]
            backward_pixel_radiance, backward_pixel_emissive, backward_pixel_alpha, backward_pixel_normal, backward_pixel_distance,
            d_L_d_pixel_radiance, d_L_d_pixel_emissive, d_L_d_pixel_alpha, d_L_d_pixel_normal, d_L_d_pixel_distance);
    }

    __device__ void trace_forth_backward_singlebounce(
        // [input]

        const float3 &ray_origin, const float3 &ray_direction,

        PerRayData per_ray_data,
        uint32_t &per_ray_data_u0,
        uint32_t &per_ray_data_u1,

        // [output]

        int &Hitcount,
        float &Alpha,

        float &Distance,
        float3 &Normal,

        float3 &Radiance,
        float &Emissive,
        float3 &Albedo,

        // [backward input]
        const float3 &backward_pixel_albedo,
        const float3 &backward_pixel_rendering_radiance,
        const float3 &backward_d_pixel_rendering_radiance_d_P,

        const float3 &d_L_d_pixel_rendering_radiance)
    {
        // avoid surfels near camera
        float distance_to_start_tracing_ray = NEAREST_DISTANCE_TO_AVOID_FRONT_SURFELS;

        Hitcount = 0;
        float Transmittance = 1.0f;
        float Transmittance_tobe = 1.0f;

        Distance = 0.0f;
        Normal = make_float3(0.0f, 0.0f, 0.0f);

        Radiance = make_float3(0.0f, 0.0f, 0.0f);
        Emissive = 0.0f;
        Albedo = make_float3(0.0f, 0.0f, 0.0f);

        // tracing (0-bounce) along the ray
        while (1)
        {
            // initialized chunk buffer
            for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; i++)
                per_ray_data.buffer[i].distance = SCENE_MAX_DISTANCE;

            // trace a chunk and collect next k closest intersected surfels along the ray
            optixTrace(optixLaunchParams.traversable,
                       ray_origin,
                       ray_direction,
                       distance_to_start_tracing_ray, // tmin
                       SCENE_MAX_DISTANCE,            // tmax
                       0.0f,                          // rayTime
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_NONE,
                       0,                   // SBT offset
                       0,                   // SBT stride
                       0,                   // missSBTIndex
                       per_ray_data_u0, per_ray_data_u1);

            // accumulate the chunk
            for (int i_surfel_in_chunk = 0; i_surfel_in_chunk < ANYHIT_CHUNK_BUFFER_SIZE; ++i_surfel_in_chunk)
            {
                // [get per_ray_data from buffer]

                const float ray_distance = per_ray_data.buffer[i_surfel_in_chunk].distance;
                if (ray_distance == SCENE_MAX_DISTANCE)
                    break;
                // this already includes the condition of hitcount == 0

                const int surfel_id = per_ray_data.buffer[i_surfel_in_chunk].surfel_id;
                const float2 surfel_uv = per_ray_data.buffer[i_surfel_in_chunk].surfel_uv;

                // [get surfel properties by surfel_id]

                const float surfel_opacity = optixLaunchParams.surfels_opacities[surfel_id];
                const float4 surfel_quaternion = optixLaunchParams.surfels_quaternions[surfel_id];
                const float2 surfel_scale = optixLaunchParams.surfels_scales[surfel_id];
                const float3 surfel_position = optixLaunchParams.surfels_positions[surfel_id];

                // normal

                const float surfel_quaternion_r = surfel_quaternion.x;
                const float surfel_quaternion_x = surfel_quaternion.y;
                const float surfel_quaternion_y = surfel_quaternion.z;
                const float surfel_quaternion_z = surfel_quaternion.w;

                // the first column, tu
                float3 surfel_tu = make_float3(1.0f - 2.0f * (surfel_quaternion_y * surfel_quaternion_y + surfel_quaternion_z * surfel_quaternion_z),
                                               2.0f * (surfel_quaternion_x * surfel_quaternion_y + surfel_quaternion_r * surfel_quaternion_z),
                                               2.0f * (surfel_quaternion_x * surfel_quaternion_z - surfel_quaternion_r * surfel_quaternion_y));
                // the third column, tv
                float3 surfel_tv = make_float3(2.0f * (surfel_quaternion_x * surfel_quaternion_y - surfel_quaternion_r * surfel_quaternion_z),
                                               1.0f - 2.0f * (surfel_quaternion_x * surfel_quaternion_x + surfel_quaternion_z * surfel_quaternion_z),
                                               2.0f * (surfel_quaternion_y * surfel_quaternion_z + surfel_quaternion_r * surfel_quaternion_x));
                // the third column, n
                float3 surfel_normal = make_float3(2.0f * (surfel_quaternion_x * surfel_quaternion_z + surfel_quaternion_r * surfel_quaternion_y),
                                                   2.0f * (surfel_quaternion_y * surfel_quaternion_z - surfel_quaternion_r * surfel_quaternion_x),
                                                   1.0f - 2.0f * (surfel_quaternion_x * surfel_quaternion_x + surfel_quaternion_y * surfel_quaternion_y));

                // [accumulate start]

                const float gaussian_value = exp(-0.5f * (surfel_uv.x * surfel_uv.x + surfel_uv.y * surfel_uv.y));
                const float eta = surfel_opacity * gaussian_value;
                const float weight = Transmittance * eta;

                Transmittance_tobe = Transmittance * (1.0f - eta);
                if (Transmittance_tobe < TRANSMITTANCE_THRESHOLD)
                    break;

                // this is the distance from the ray_origin_safe
                Distance += weight * ray_distance;
                Normal += weight * ((dot(surfel_normal, ray_direction) > 0.0f) ? -surfel_normal : surfel_normal);

                Radiance += weight * optixLaunchParams.surfels_radiances[surfel_id];
                Emissive += weight * optixLaunchParams.surfels_emissives[surfel_id];
                Albedo += weight * optixLaunchParams.surfels_albedos[surfel_id];

                // [backward start]

                const float d_P_d_albedo = weight;
                const float3 d_L_d_albedo = d_L_d_pixel_rendering_radiance * backward_d_pixel_rendering_radiance_d_P * d_P_d_albedo;

                atomicAdd(&optixLaunchParams.d_L_d_surfels_albedos[surfel_id].x, d_L_d_albedo.x);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_albedos[surfel_id].y, d_L_d_albedo.y);
                atomicAdd(&optixLaunchParams.d_L_d_surfels_albedos[surfel_id].z, d_L_d_albedo.z);

                // [accumulate end]

                Hitcount++;
                if (Hitcount >= MAXIMAL_HITCOUNT_ALONG_THE_RAY)
                    break;

                Transmittance = Transmittance_tobe;
            }

            // break again to end the trace_forth(); previous break only end the chunk
            if ((per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance == SCENE_MAX_DISTANCE) || (Transmittance_tobe < TRANSMITTANCE_THRESHOLD) || (Hitcount >= MAXIMAL_HITCOUNT_ALONG_THE_RAY))
                break;

            // update tmin for tracing next chunk
            distance_to_start_tracing_ray = per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance + MINIMAL_DISTANCE_TO_AVOID_SELF_INTERSECTION;
        }

        Alpha = 1.0f - Transmittance;
    }

    extern "C" __global__ void __raygen__singlebounce_backward()
    {
        // [get ids of the current thread]

        const uint3 launchIndex = optixGetLaunchIndex();

        const int i_pixel = launchIndex.y * optixLaunchParams.WIDTH + launchIndex.x;

        // if (i_pixel == 0)
        // {
        //     printf("[DEBUG] calling __raygen__0bounce()\n");
        // }

        // [prepare and share a single chunk buffer for anyhit through a pixel]

        PerRayData per_ray_data;
        IntersectionInfo buffer[ANYHIT_CHUNK_BUFFER_SIZE];
        per_ray_data.buffer = buffer;

        uint32_t per_ray_data_u0, per_ray_data_u1;
        packPointer(&per_ray_data, per_ray_data_u0, per_ray_data_u1);

        // [get camera origin and direction]

        const float3 camera_ray_origin = optixLaunchParams.rays_origins[i_pixel];
        const float3 camera_ray_direction = optixLaunchParams.rays_directions[i_pixel];

        // [backward input]

        const float3 backward_pixel_albedo = optixLaunchParams.backward_pixels_albedos[i_pixel];
        const float3 backward_pixel_rendering_radiance = optixLaunchParams.backward_pixels_rendering_radiances[i_pixel];
        const float3 backward_d_pixel_rendering_radiance_d_P = optixLaunchParams.backward_pixels_d_rendering_radiances_d_P[i_pixel];

        const float3 d_L_d_pixel_rendering_radiance = optixLaunchParams.d_L_d_pixels_rendering_radiances[i_pixel];

        // [shoot rays from camera and use trace_forth() once for all paths]

        int camera_Hitcount;
        float camera_Alpha;

        float camera_Distance;
        float3 camera_Normal;

        float3 camera_Radiance;
        float camera_Emissive;
        float3 camera_Albedo;

        trace_forth_backward_singlebounce(
            camera_ray_origin, camera_ray_direction,
            per_ray_data, per_ray_data_u0, per_ray_data_u1,
            camera_Hitcount, camera_Alpha, camera_Distance, camera_Normal, camera_Radiance, camera_Emissive, camera_Albedo,
            // [backward]
            backward_pixel_albedo, backward_pixel_rendering_radiance, backward_d_pixel_rendering_radiance_d_P, d_L_d_pixel_rendering_radiance);
    }

    extern "C" __global__ void __anyhit__ah()
    {
        PerRayData &per_ray_data = *(PerRayData *)getPerRayData<PerRayData>();

        float distance = optixGetRayTmax();

        if (distance < per_ray_data.buffer[ANYHIT_CHUNK_BUFFER_SIZE - 1].distance)
        {
            const int triangle_id = optixGetPrimitiveIndex();
            const float2 triangle_uv = optixGetTriangleBarycentrics();

            const int surfel_id = optixGetPrimitiveIndex() / 2;
            float2 surfel_uv;
            if (triangle_id % 2 == 0)
            {
                surfel_uv.x = 3.0f * (2.0f * triangle_uv.x - 1.0f);
                surfel_uv.y = 3.0f * (2.0f * triangle_uv.y - 1.0f);
            }
            else
            {
                surfel_uv.x = 3.0f * (1.0f - 2.0f * triangle_uv.x);
                surfel_uv.y = 3.0f * (1.0f - 2.0f * triangle_uv.y);
            }

            // IntersectionInfo temp_intersection_info;
            // IntersectionInfo current_intersection_info;

            // current_intersection_info.distance = distance;
            // current_intersection_info.surfel_id = surfel_id;
            // current_intersection_info.surfel_uv = surfel_uv;

            // // Insert the new primitive into the ascending t sorted list
            // for (int i = 0; i < ANYHIT_CHUNK_BUFFER_SIZE; ++i)
            // {
            //     // Swap if the new intersection is closer
            //     if (per_ray_data.buffer[i].distance > current_intersection_info.distance)
            //     {
            //         // Store the original buffer info
            //         temp_intersection_info = per_ray_data.buffer[i];
            //         // Update the current intersection info
            //         per_ray_data.buffer[i] = current_intersection_info;
            //         // Swap
            //         current_intersection_info = temp_intersection_info;
            //     }
            // }

            int position_in_chunk_to_insert = -1;

            for (int i_position = ANYHIT_CHUNK_BUFFER_SIZE - 1; i_position > -1; --i_position)
            {
                if (distance < per_ray_data.buffer[i_position].distance)
                {
                    position_in_chunk_to_insert = i_position;
                }
            }

            for (int i_position = ANYHIT_CHUNK_BUFFER_SIZE - 1; i_position > position_in_chunk_to_insert; --i_position)
            {
                per_ray_data.buffer[i_position] = per_ray_data.buffer[i_position - 1];
            }

            per_ray_data.buffer[position_in_chunk_to_insert].distance = distance;
            per_ray_data.buffer[position_in_chunk_to_insert].surfel_id = surfel_id;
            per_ray_data.buffer[position_in_chunk_to_insert].surfel_uv = surfel_uv;

            // Ignore the intersection to continue traversal
            optixIgnoreIntersection();
        }
    }

    extern "C" __global__ void __miss__ms()
    {
    }

    extern "C" __global__ void __closesthit__ch()
    {
    }
} // ::osc
