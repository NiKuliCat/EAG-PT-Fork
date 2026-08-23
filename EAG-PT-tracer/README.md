# EAG-PT-tracer

This is the backend tracer used in [EAG-PT](https://github.com/InternRobotics/EAG-PT).

For running EAG-PT, please follow the installation process in [EAG-PT](https://github.com/InternRobotics/EAG-PT).

## Introduction

- This is a tracer written in CUDA and OptiX, and wrapped with PyTorch Extension to support calls from PyTorch side.
- [`eag_pt_tracer_optix/ext.cpp`](eag_pt_tracer_optix/ext.cpp) defines the interface of calls.
- [`eag_pt_tracer_optix/header/SampleRenderer.h`](eag_pt_tracer_optix/header/SampleRenderer.h) and [`eag_pt_tracer_optix/source/SampleRenderer.cpp`](eag_pt_tracer_optix/source/SampleRenderer.cpp) implements the calls through class `SampleRenderer`.
- OptiX programs are implemented in [`eag_pt_tracer_optix/program/devicePrograms.cu`](eag_pt_tracer_optix/program/devicePrograms.cu). This file is compiled using [`eag_pt_tracer_optix/program/generate-ptx.sh`](eag_pt_tracer_optix/program/generate-ptx.sh), which is already consolidated into [`setup.py`](setup.py), to generate ptx in `eag_pt_tracer_optix/ptx`. [`eag_pt_tracer_optix/source/SampleRenderer.cpp`](eag_pt_tracer_optix/source/SampleRenderer.cpp) uses these programs through `#include "../ptx/devicePrograms.h"`.
- [`eag_pt_tracer_optix/external/optix`](eag_pt_tracer_optix/external/optix): Header files of OptiX v7.7.0 are directly downloaded from https://github.com/NVIDIA/optix-dev/tree/v7.7.0 and included in this repo.

## Development

### Installation

```sh
# check PyTorch versions at https://pytorch.org/get-started/locally/ and https://pytorch.org/get-started/previous-versions/
pip install torch torchvision
# for fast compilation
pip install ninja
# nvcc should be installed in advance
# arch 8.9 is only for RTX 4090, check https://developer.nvidia.com/cuda-gpus. arch should also be changed in `setup.py` `command_nvcc` and `eag_pt_tracer_optix/program/generate-ptx.sh`. It should also be okay to leave the arch blank, in this case all archs will be compiled, which is inefficient.
rm -rf *_optix/ptx/ build/ *.egg-info/; TORCH_CUDA_ARCH_LIST=8.9 pip install . -v --no-build-isolation
```

### IntelliSense

- Accordingly change paths in [`.vscode/c_cpp_properties.json`](.vscode/c_cpp_properties.json) to get IntelliSense in VS Code.

## References

- [optix7course](https://github.com/ingowald/optix7course): the framework that this tracer is based on
- [torchoptix](https://github.com/eliphatfs/torchoptix): insight on passing raw pointers into C++
- [3DGRT](https://github.com/nv-tlabs/3dgrut), [gtracer](https://github.com/fudan-zvg/gtracer): ray tracer on 3D Gaussians
- [EnvGS tracer](https://github.com/xbillowy/diff-surfel-tracing), [IRGS tracer](https://github.com/fudan-zvg/IRGS/tree/main/submodules/surfel_tracer): single-bounce ray tracer on 2D Gaussians
- [PyTorch C++ Extension](https://docs.pytorch.org/tutorials/advanced/cpp_extension.html): link between Python and C++ OptiX
    - [pybind-python](https://github.com/Xijie-Yang/pybind11-mypackage), [pybind11-tensor](https://github.com/Xijie-Yang/pybind11-tensor-mypackage), [pytorch-cpp-extension](https://github.com/Xijie-Yang/pytorch-cpp-extension-mypackage), [pytorch-cuda-extension](https://github.com/Xijie-Yang/pytorch-cuda-extension-mypackage): templates
