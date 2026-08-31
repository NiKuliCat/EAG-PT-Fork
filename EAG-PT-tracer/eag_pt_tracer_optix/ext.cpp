// VS Code include fix: https://discuss.pytorch.org/t/how-to-include-torch-extension-h-correctly/182882/3
#include <torch/extension.h>

#include "header/SampleRenderer.h"

// https://docs.pytorch.org/tutorials/advanced/cpp_extension.html
// The torch extension build will define `TORCH_EXTENSION_NAME` as the name you
// give your extension in the `setup.py` script.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  pybind11::class_<osc::SampleRenderer>(m, "SampleRenderer")
      .def(pybind11::init<>())
      .def("buildAccel", &osc::SampleRenderer::buildAccel)
      .def("nobounce", &osc::SampleRenderer::nobounce)
      .def("materialpass", &osc::SampleRenderer::materialpass)
      .def("materialpass_backward", &osc::SampleRenderer::materialpass_backward)
      .def("singlebounce", &osc::SampleRenderer::singlebounce)
      .def("pathtracing", &osc::SampleRenderer::pathtracing)
      .def("nobounce_backward", &osc::SampleRenderer::nobounce_backward)
      .def("singlebounce_backward", &osc::SampleRenderer::singlebounce_backward);
}
