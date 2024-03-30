#include "fitting.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(bs_fit_cpp, m) {
  m.doc() = "pybind11 bspline surface fitting plugin";

  m.def("linspace", &linspace, "fitting.linspace");

  m.def("toMaxValues", &toMaxValues, "fitting.toMaxValues");
}
