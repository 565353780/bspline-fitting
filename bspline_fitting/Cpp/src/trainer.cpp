#include "trainer.h"

using namespace pybind11::literals;

Trainer::Trainer(const std::string root_path, const int degree_u,
                 const int degree_v, const int size_u, const int size_v,
                 const int sample_num_u, const int sample_num_v,
                 const float start_u, const float start_v, const float stop_u,
                 const float stop_v, const std::string device,
                 const int warm_epoch_step_num, const int warm_epoch_num,
                 const int finetune_step_num, const float lr,
                 const float weight_decay, const float factor,
                 const int patience, const float min_lr,
                 const std::string save_result_folder_path,
                 const std::string save_log_folder_path) {
  py::gil_scoped_acquire acquire;

  py::object sys = py::module_::import("sys");

  sys.attr("path").attr("append")(root_path);

  py::object Trainer = py::module_::import("bspline_fitting.Module.trainer");

  trainer_ = Trainer.attr("Trainer")(
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

  return;
}

Trainer::~Trainer() {}

const bool Trainer::toBSplineSurface(const std::vector<float> &sample_points) {
  py::gil_scoped_acquire acquire;

  py::list sample_point_list;
  for (int i = 0; i < sample_points.size(); ++i) {
    sample_point_list.append(sample_points[i]);
  }

  py::print("start autoTrainBSplineSurface");
  const bool success =
      trainer_
          .attr("autoTrainBSplineSurface")("gt_points"_a = sample_point_list)
          .cast<bool>();

  return success;
}

const std::vector<float> Trainer::getKNotsU() {
  py::list list = trainer_.attr("bspline_surface").attr("toKNotsUList")();

  std::vector<float> knots_u;
  knots_u.reserve(list.size());

  for (int i = 0; i < list.size(); ++i) {
    knots_u.emplace_back(list[i].cast<float>());
  }

  return knots_u;
}

const std::vector<float> Trainer::getKNotsV() {
  py::list list = trainer_.attr("bspline_surface").attr("toKNotsVList")();

  std::vector<float> knots_v;
  knots_v.reserve(list.size());

  for (int i = 0; i < list.size(); ++i) {
    knots_v.emplace_back(list[i].cast<float>());
  }

  return knots_v;
}
const std::vector<float> Trainer::getCtrlPts() {
  py::list list = trainer_.attr("bspline_surface").attr("toCtrlPtsList")();

  std::vector<float> ctrl_pts;
  ctrl_pts.reserve(list.size());

  for (int i = 0; i < list.size(); ++i) {
    ctrl_pts.emplace_back(list[i].cast<float>());
  }

  return ctrl_pts;
}
