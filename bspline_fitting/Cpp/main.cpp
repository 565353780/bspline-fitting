#include "value_torch.h"
#include <pybind11/embed.h>

namespace py = pybind11;
using namespace std;
using namespace pybind11::literals;

int main() {
  const string root_path = "../../bspline-fitting/";
  const int degree_u = 2;
  const int degree_v = 2;
  const int size_u = 7;
  const int size_v = 7;
  const int sample_num_u = 20;
  const int sample_num_v = 20;
  const float start_u = 0.0;
  const float start_v = 0.0;
  const float stop_u = 1.0;
  const float stop_v = 1.0;
  const string device = "cpu";

  const int warm_epoch_step_num = 20;
  const int warm_epoch_num = 4;
  const int finetune_step_num = 400;
  const float lr = 5e-2;
  const float weight_decay = 1e-4;
  const float factor = 0.9;
  const int patience = 1;
  const float min_lr = 1e-3;

  const string save_result_folder_path = "auto";
  const string save_log_folder_path = "auto";

  cout << "Hello PyBind World" << endl;

  py::scoped_interpreter guard{};

  py::object sys = py::module_::import("sys");

  sys.attr("path").attr("append")(root_path);

  py::object Trainer = py::module_::import("bspline_fitting.Module.trainer");

  py::object trainer = Trainer.attr("Trainer")(
      "degree_u"_a = degree_u, "degree_v"_a = degree_v, "size_u"_a = size_u,
      "size_v"_a = size_v, "sample_num_u"_a = sample_num_u,
      "sample_num_v"_a = sample_num_v, "start_u"_a = start_u,
      "start_v"_a = start_v, "stop_u"_a = stop_u, "stop_v"_a = stop_v,
      "device"_a = device, "warm_epoch_step_num"_a = warm_epoch_step_num,
      "warm_epoch_num"_a = warm_epoch_num,
      "finetune_step_num"_a = finetune_step_num, "lr"_a = lr,
      "weight_decay"_a = weight_decay, "factor"_a = factor,
      "patience"_a = patience, "min_lr"_a = min_lr,
      "save_result_folder_path"_a = save_result_folder_path,
      "save_log_folder_path"_a = save_log_folder_path);

  return 1;
}
