#include "value.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(bs_fit_cpp, m) {
  m.doc() = "pybind11 bspline surface fitting plugin";

  m.def("linspace", &linspace, "value.linspace");
  m.def("find_span_linear", &find_span_linear, "value.find_span_linear");
  m.def("find_spans", &find_spans, "value.find_spans");
  m.def("basis_function", &basis_function, "value.basis_function");
  m.def("basis_functions", &basis_functions, "value.basis_functions");
  m.def("toPoints", &toPoints, "value.toPoints");
}
