#pragma once

#include <optix_types.h>
#include <vector_types.h>

namespace osc
{
    struct LaunchParams
    {
        // [for acceleration structure build]

        OptixTraversableHandle traversable;

        // [optixLaunch]

        int HEIGHT;
        int WIDTH;

        int SPP;          // for reflection and emission
        int BOUNCE_LIMIT; // for emission

        // [input - surfels]

        const float3 *surfels_positions;
        const float2 *surfels_scales;
        const float4 *surfels_quaternions;
        const float *surfels_opacities;
        // values
        const float3 *surfels_radiances;
        const float *surfels_emissives;
        const float3 *surfels_albedos;
        const float *surfels_roughnesses;
        const float *surfels_metallics;
        // values end

        // [input - emissive sampling]

        int emissive_surfels_count;
        const int *emissive_surfels_ids;
        const float *emissive_surfels_proportions_pdfs;
        const float *emissive_surfels_proportions_cdfs;

        // [input - rays]

        const float3 *rays_origins;
        const float3 *rays_directions;

        // [output - results]

        int *pixels_hitcounts;
        float *pixels_alphas;
        float *pixels_distances;
        float3 *pixels_normals;
        // values
        float3 *pixels_radiances;
        float *pixels_emissives;
        float3 *pixels_albedos;
        float3 *pixels_basecolors;
        float *pixels_roughnesses;
        float *pixels_metallics;
        // values end
        float3 *pixels_rendering_radiances;
        float3 *pixels_d_rendering_radiances_d_P;

        // [backward input - no bounce]

        const float3 *backward_pixels_radiances;
        const float *backward_pixels_emissives;
        const float *backward_pixels_alphas;
        const float3 *backward_pixels_normals;
        const float *backward_pixels_distances;

        const float3 *d_L_d_pixels_radiances;
        const float *d_L_d_pixels_emissives;
        const float *d_L_d_pixels_alphas;
        const float3 *d_L_d_pixels_normals;
        const float *d_L_d_pixels_distances;

        // [backward output - no bounce]

        float3 *d_L_d_surfels_radiances;
        float *d_L_d_surfels_emissives;
        float *d_L_d_surfels_opacities;
        float2 *d_L_d_surfels_scales;
        float3 *d_L_d_surfels_positions;
        float4 *d_L_d_surfels_quaternions;

        // [backward input - single bounce]

        const float3 *backward_pixels_albedos;
        const float3 *backward_pixels_rendering_radiances;
        const float3 *backward_pixels_d_rendering_radiances_d_P;

        const float3 *d_L_d_pixels_rendering_radiances;

        // [backward output - single bounce]

        float3 *d_L_d_surfels_albedos;
        const float3 *d_L_d_pixels_basecolors;
        const float *d_L_d_pixels_roughnesses;
        const float *d_L_d_pixels_metallics;
        float *d_L_d_surfels_roughnesses;
        float *d_L_d_surfels_metallics;
    };
} // ::osc
