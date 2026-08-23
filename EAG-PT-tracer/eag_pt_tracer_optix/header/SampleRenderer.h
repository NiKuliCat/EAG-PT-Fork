#pragma once

// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include "LaunchParams.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    /*! a sample OptiX-7 renderer that demonstrates how to set up
        context, module, programs, pipeline, SBT, etc, and perform a
        valid launch that renders some pixel (using a simple test
        pattern, in this case */
    class SampleRenderer
    {
        // ------------------------------------------------------------------
        // publicly accessible interface
        // ------------------------------------------------------------------
    public:
        /*! constructor - performs all setup, including initializing
          optix, creates module, pipeline, programs, SBT, etc. */
        SampleRenderer();

        ~SampleRenderer();

        /*! build an acceleration structure for the given triangle mesh */
        void buildAccel(int triangles_vertices_count, uintptr_t triangles_vertices, int triangle_indices_count, uintptr_t triangles_indices);

        void nobounce(
            // [numbers]
            int HEIGHT,
            int WIDTH,
            // [input - surfels]
            uintptr_t ptr_surfels_positions,
            uintptr_t ptr_surfels_scales,
            uintptr_t ptr_surfels_quaternions,
            uintptr_t ptr_surfels_opacities,
            // values
            uintptr_t ptr_surfels_radiances,
            uintptr_t ptr_surfels_emissives,
            uintptr_t ptr_surfels_albedos,
            // [input - rays]
            uintptr_t ptr_rays_origins,
            uintptr_t ptr_rays_directions,
            // [output - results]
            uintptr_t ptr_pixels_hitcounts,
            uintptr_t ptr_pixels_alphas,
            uintptr_t ptr_pixels_distances,
            uintptr_t ptr_pixels_normals,
            // [output - values]
            uintptr_t ptr_pixels_radiances,
            uintptr_t ptr_pixels_emissives,
            uintptr_t ptr_pixels_albedos);

        void singlebounce(
            // [numbers]
            int HEIGHT,
            int WIDTH,
            int SPP,
            // [input - surfels]
            uintptr_t ptr_surfels_positions,
            uintptr_t ptr_surfels_scales,
            uintptr_t ptr_surfels_quaternions,
            uintptr_t ptr_surfels_opacities,
            // values
            uintptr_t ptr_surfels_radiances,
            uintptr_t ptr_surfels_emissives,
            uintptr_t ptr_surfels_albedos,
            // values end
            // [input - rays]
            uintptr_t ptr_rays_origins,
            uintptr_t ptr_rays_directions,
            // [output - results]
            uintptr_t ptr_pixels_albedos,
            uintptr_t ptr_pixels_rendering_radiances,
            uintptr_t ptr_d_pixels_rendering_radiances_d_P);

        void pathtracing(
            // [numbers]
            int HEIGHT,
            int WIDTH,
            int BOUNCE_LIMIT,
            int SPP,
            // [input - surfels]
            uintptr_t ptr_surfels_positions,
            uintptr_t ptr_surfels_scales,
            uintptr_t ptr_surfels_quaternions,
            uintptr_t ptr_surfels_opacities,
            // [input - surfels values]
            uintptr_t ptr_surfels_radiances,
            uintptr_t ptr_surfels_emissives,
            uintptr_t ptr_surfels_albedos,
            // [input - rays]
            uintptr_t ptr_rays_origins,
            uintptr_t ptr_rays_directions,
            // [output - results]
            uintptr_t ptr_rendering_radiances);

        void nobounce_backward(
            // [numbers]
            int HEIGHT,
            int WIDTH,
            // [input - surfels]
            uintptr_t ptr_surfels_positions,
            uintptr_t ptr_surfels_scales,
            uintptr_t ptr_surfels_quaternions,
            uintptr_t ptr_surfels_opacities,
            // values
            uintptr_t ptr_surfels_radiances,
            uintptr_t ptr_surfels_emissives,
            uintptr_t ptr_surfels_albedos,
            // [input - rays]
            uintptr_t ptr_rays_origins,
            uintptr_t ptr_rays_directions,
            // [backward - input - forward results]
            uintptr_t ptr_pixels_radiances,
            uintptr_t ptr_pixels_emissives,
            uintptr_t ptr_pixels_alphas,
            uintptr_t ptr_pixels_normals,
            uintptr_t ptr_pixels_distances,
            // [backward - input - pytorch gradients]
            uintptr_t ptr_d_L_d_pixels_radiances,
            uintptr_t ptr_d_L_d_pixels_emissives,
            uintptr_t ptr_d_L_d_pixels_alphas,
            uintptr_t ptr_d_L_d_pixels_normals,
            uintptr_t ptr_d_L_d_pixels_distances,
            // [backward: output]
            uintptr_t ptr_d_L_d_surfels_radiances,
            uintptr_t ptr_d_L_d_surfels_emissives,
            uintptr_t ptr_d_L_d_surfels_opacities,
            uintptr_t ptr_d_L_d_surfels_scales,
            uintptr_t ptr_d_L_d_surfels_positions,
            uintptr_t ptr_d_L_d_surfels_quaternions);

        void singlebounce_backward(
            // [numbers]
            int HEIGHT,
            int WIDTH,
            // [input - surfels]
            uintptr_t ptr_surfels_positions,
            uintptr_t ptr_surfels_scales,
            uintptr_t ptr_surfels_quaternions,
            uintptr_t ptr_surfels_opacities,
            // values
            uintptr_t ptr_surfels_radiances,
            uintptr_t ptr_surfels_emissives,
            uintptr_t ptr_surfels_albedos,
            // [input - rays]
            uintptr_t ptr_rays_origins,
            uintptr_t ptr_rays_directions,
            // [backward - input - forward results]
            uintptr_t ptr_pixels_albedos,
            uintptr_t ptr_pixels_rendering_radiances,
            // [backward - input - pytorch gradients]
            uintptr_t ptr_d_L_d_pixels_rendering_radiances,
            // [backward: output]
            uintptr_t ptr_d_L_d_surfels_albedos,
            uintptr_t ptr_d_pixels_rendering_radiances_d_P);

    protected:
        // ------------------------------------------------------------------
        // internal helper functions
        // ------------------------------------------------------------------

        /*! helper function that initializes optix and checks for errors */
        void initOptix();

        /*! creates and configures a optix device context (in this simple
          example, only for the primary GPU device) */
        void createContext();

        /*! creates the module that contains all the programs we are going
          to use. in this simple example, we use a single module from a
          single .cu file, using a single embedded ptx string */
        void createModule();

        /*! does all setup for the raygen program(s) we are going to use */
        void createRaygenPrograms();
        /*! does all setup for the miss program(s) we are going to use */
        void createMissPrograms();
        /*! does all setup for the hitgroup program(s) we are going to use */
        void createHitgroupPrograms();

        /*! assembles the full pipeline of all programs */
        void createPipeline();

        /*! constructs the shader binding table */
        void buildSBT();

    protected:
        /*! @{ CUDA device context and stream that optix pipeline will run
            on, as well as device properties for this device */
        CUcontext cudaContext;
        CUstream stream;
        cudaDeviceProp deviceProps;
        /*! @} */

        //! the optix context that our pipeline will run in.
        OptixDeviceContext optixContext;

        /*! @{ the pipeline we're building */
        OptixPipeline pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        /*! @} */

        /*! @{ the module that contains out device programs */
        OptixModule module;
        OptixModuleCompileOptions moduleCompileOptions = {};
        /* @} */

        /*! vector of all our program(group)s, and the SBT built around
            them */
        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        /*! @{ our launch parameters, on the host, and the buffer to store
            them on the device */
        LaunchParams launchParams;
        CUDABuffer launchParamsBuffer;
        /*! @} */

        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;
    };
} // ::osc
