#include "trainer.h"
#include <iostream>

int main() {
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

  std::vector<float> sample_points;
  sample_points.resize(900);
  for (int i = 0; i < int(sample_points.size() / 3); ++i) {
    sample_points[3 * i] = 0.0;
    sample_points[3 * i + 1] = 1.0;
    sample_points[3 * i + 2] = 2.0;
  }

  std::cout << "Hello PyBind World" << std::endl;

  Trainer trainer(root_path, degree_u, degree_v, size_u, size_v, sample_num_u,
                  sample_num_v, start_u, start_v, stop_u, stop_v, device,
                  warm_epoch_step_num, warm_epoch_num, finetune_step_num, lr,
                  weight_decay, factor, patience, min_lr);

  trainer.toBSplineSurface(sample_points);

  std::cout << "finish run main!" << std::endl;
  return 1;
}
