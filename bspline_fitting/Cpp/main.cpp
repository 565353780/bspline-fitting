#include "trainer.h"
#include <iostream>

int main() {
  // input super params
  const std::string root_path = "../../bspline-fitting/";
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
  const std::string device = "cpu";
  const int warm_epoch_step_num = 20;
  const int warm_epoch_num = 4;
  const int finetune_step_num = 400;
  const float lr = 5e-2;
  const float weight_decay = 1e-4;
  const float factor = 0.9;
  const int patience = 1;
  const float min_lr = 1e-3;
  const std::string save_result_folder_path = "auto";
  const std::string save_log_folder_path = "auto";

  // input point cloud [x1, y1, z1, x2, y2, z2, ...]
  std::vector<float> sample_points;
  sample_points.resize(900);
  for (int i = 0; i < int(sample_points.size() / 3); ++i) {
    sample_points[3 * i] = 0.0 * i;
    sample_points[3 * i + 1] = 1.0 * i;
    sample_points[3 * i + 2] = 2.0 * i;
  }

  // construct Trainer class
  Trainer trainer(root_path, degree_u, degree_v, size_u, size_v, sample_num_u,
                  sample_num_v, start_u, start_v, stop_u, stop_v, device,
                  warm_epoch_step_num, warm_epoch_num, finetune_step_num, lr,
                  weight_decay, factor, patience, min_lr);

  // auto fit bspline surface
  const bool success = trainer.toBSplineSurface(sample_points);
  if (!success) {
    std::cout << "toBSplineSurface failed!" << std::endl;
    return -1;
  }

  return 1;
}
