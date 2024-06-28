#include "bspline_surface.h"
#include "trainer.h"
#include <iostream>

int main() {
  torch::set_num_threads(16);

  BSplineSurface bsf = BSplineSurface();

  // input super params
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
  const std::string idx_dtype = "int64";
  const std::string dtype = "float32";
  const std::string device = "cpu";
  const int warm_epoch_step_num = 20;
  const int warm_epoch_num = 4;
  const int finetune_step_num = 400;
  const float lr = 5e-2;
  const float weight_decay = 1e-4;
  const float factor = 0.8;
  const int patience = 4;
  const float min_lr = 1e-3;
  const std::string save_result_folder_path = "auto";
  const std::string save_log_folder_path = "auto";

  std::string real_device = device;
  if (!torch::cuda::is_available()) {
    std::cout << "[WARN][main::main]" << std::endl;
    std::cout << "\t cuda is not available! will use cpu!" << std::endl;
    real_device = "cpu";
  }

  // input point cloud [x1, y1, z1, x2, y2, z2, ...]
  std::vector<float> sample_points;
  sample_points.resize(900);
  for (int i = 0; i < int(sample_points.size() / 3); ++i) {
    sample_points[3 * i] = 0.0 * i;
    sample_points[3 * i + 1] = 1.0 * i;
    sample_points[3 * i + 2] = 2.0 * i;
  }

  // construct Trainer class
  Trainer trainer(degree_u, degree_v, size_u, size_v, sample_num_u,
                  sample_num_v, start_u, start_v, stop_u, stop_v, idx_dtype,
                  dtype, real_device, warm_epoch_step_num, warm_epoch_num,
                  finetune_step_num, lr, weight_decay, factor, patience,
                  min_lr);

  trainer.autoTrainBSplineSurface(sample_points);

  // get bspline surface params
  const std::vector<float> knots_u = trainer.toKNotsU();
  const std::vector<float> knots_v = trainer.toKNotsV();
  const std::vector<float> ctrl_pts = trainer.toCtrlPts();

  // output bspline surface params
  std::cout << "knots_u:" << std::endl;
  for (int i = 0; i < knots_u.size(); ++i) {
    std::cout << knots_u[i] << ',';
  }
  std::cout << std::endl;

  std::cout << "knots_v:" << std::endl;
  for (int i = 0; i < knots_v.size(); ++i) {
    std::cout << knots_v[i] << ',';
  }
  std::cout << std::endl;

  std::cout << "ctrl_pts:" << std::endl;
  for (int i = 0; i < ctrl_pts.size(); ++i) {
    std::cout << ctrl_pts[i] << ',';
    if (i % 3 == 2) {
      std::cout << std::endl;
    }
  }

  std::cout << "finished running main!" << std::endl;
  return 1;
}
