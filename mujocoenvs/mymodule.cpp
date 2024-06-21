#include <mujoco/mujoco.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

int add(int i, int j) { return i + j; }
int add1(int i, int j) { return i + j; }
int add2(int i, int j) { return i + j; }

int my_mj_func() { return mj_version(); }

PYBIND11_MODULE(example, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("add", &add, "A function that adds two numbers");
  m.def("add1", &add, py::arg("i"), py::arg("j"));
  m.def("add2", &add, "i"_a, "j"_a);
  m.def("my_mj_func", &my_mj_func, "returns the my_mujoco version");
  py::object world = py::cast("World");
  m.attr("meow") = world;
}
