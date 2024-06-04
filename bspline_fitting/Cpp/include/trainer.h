#pragma once

#include <pybind11/embed.h>
#include <string>

namespace py = pybind11;

class __attribute__((visibility("default"))) Trainer {
public:
  Trainer(const std::string root_path = "../../bspline-fitting/",
          const int degree_u = 2, const int degree_v = 2, const int size_u = 7,
          const int size_v = 7, const int sample_num_u = 20,
          const int sample_num_v = 20, const float start_u = 0.0,
          const float start_v = 0.0, const float stop_u = 1.0,
          const float stop_v = 1.0, const std::string device = "cpu",
          const int warm_epoch_step_num = 20, const int warm_epoch_num = 4,
          const int finetune_step_num = 400, const float lr = 5e-2,
          const float weight_decay = 1e-4, const float factor = 0.9,
          const int patience = 1, const float min_lr = 1e-3,
          const std::string save_result_folder_path = "auto",
          const std::string save_log_folder_path = "auto");
  ~Trainer();

  const bool toBSplineSurface(const std::vector<float> &sample_points);

  const std::vector<float> getKNotsU();
  const std::vector<float> getKNotsV();
  const std::vector<float> getCtrlPts();

private:
  py::scoped_interpreter guard_{};

  py::object trainer_;
};
