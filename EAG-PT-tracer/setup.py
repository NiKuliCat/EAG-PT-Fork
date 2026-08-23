import pathlib, os

# https://setuptools.pypa.io/en/latest/userguide/quickstart.html
import setuptools

# https://docs.pytorch.org/tutorials/advanced/cpp_extension.html
import torch.utils.cpp_extension

MY_PACKAGE_NAME = "eag_pt_tracer"

MY_TORCH_EXTENSION_NAME = f"{MY_PACKAGE_NAME}_optix"
MY_TORCH_EXTENSION_PATH = pathlib.Path(__file__).parent / MY_TORCH_EXTENSION_NAME


# Custom build extension to build the OptiX tracing kernel
class CustomBuildExtension(torch.utils.cpp_extension.BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        MY_TORCH_EXTENSION_PROGRAM_PATH = MY_TORCH_EXTENSION_PATH / "program"
        MY_TORCH_EXTENSION_PTX_PATH = MY_TORCH_EXTENSION_PATH / "ptx"
        MY_TORCH_EXTENSION_PTX_PATH.mkdir(parents=True, exist_ok=True)

        INCLUDE_DIR = MY_TORCH_EXTENSION_PATH / "external/optix/"

        PROGRAM_NAMES = ["devicePrograms"]

        # convert each cuda program into a ptx header, which can be directly included in source file

        for program_name in PROGRAM_NAMES:
            cuda_file = MY_TORCH_EXTENSION_PROGRAM_PATH / f"{program_name}.cu"
            ptx_file = MY_TORCH_EXTENSION_PTX_PATH / f"{program_name}.ptx"
            ptx_header_file = MY_TORCH_EXTENSION_PTX_PATH / f"{program_name}.h"

            # should change arch according to the GPU

            # Warning: Program is doing double precision computations. No source location available. The input PTX may not contain debug information (nvcc option: -lineinfo), OptixModuleCompileOptions::debugLevel set to OPTIX_COMPILE_DEBUG_LEVEL_NONE, or no useful information is present for the current block.
            # however it still does not work

            command_nvcc = f"nvcc -ptx -arch=compute_89 -lineinfo {cuda_file} -o {ptx_file} -I{INCLUDE_DIR}/"
            print(f"[DEBUG] {command_nvcc=}")
            ret = os.system(command_nvcc)
            assert ret == 0

            command_bin2c = f"bin2c {ptx_file} --const --name {program_name}_ptx > {ptx_header_file}"
            print(f"[DEBUG] {command_bin2c=}")
            ret = os.system(command_bin2c)
            assert ret == 0

            print()

        # Run the original build_extensions
        super().build_extensions()


# https://setuptools.pypa.io/en/latest/references/keywords.html
setuptools.setup(
    # A string specifying the name of the package.
    name=f"{MY_PACKAGE_NAME}",
    # A string specifying the version number of the package.
    version="1.0.0",
    # A list of strings specifying the packages that setuptools will manipulate.
    packages=[f"{MY_PACKAGE_NAME}_python"],
    # https://setuptools.pypa.io/en/stable/userguide/ext_modules.html
    ext_modules=[
        # https://docs.pytorch.org/docs/stable/cpp_extension.html
        torch.utils.cpp_extension.CUDAExtension(
            name=MY_TORCH_EXTENSION_NAME,
            sources=[
                f"{MY_TORCH_EXTENSION_NAME}/source/SampleRenderer.cpp",
                f"{MY_TORCH_EXTENSION_NAME}/ext.cpp",
            ],
            include_dirs=[
                MY_TORCH_EXTENSION_PATH / f"external/optix",
            ],
            extra_compile_args={
                # https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
                "cxx": ["-fvisibility=hidden"],
            },
            # to use cuCtxGetCurrent
            # https://stackoverflow.com/questions/66273536/undefined-reference-to-cuctxgetcurrent-while-getting-cuda-context-for-optix
            # ImportError: ...cpython-311-x86_64-linux-gnu.so: undefined symbol: cuCtxGetCurrent
            extra_link_args=["-lcuda"],
        )
    ],
    # https://docs.pytorch.org/tutorials/advanced/cpp_extension.html
    # `BuildExtension` is neccessary for pytorch to pass `TORCH_EXTENSION_NAME` into ext.cpp
    cmdclass={"build_ext": CustomBuildExtension},
)
