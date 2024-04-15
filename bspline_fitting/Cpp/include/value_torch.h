#pragma once

#include <torch/extension.h>

const torch::Tensor basis_function_torch(const int &degree,
                                         const torch::Tensor &knot_vector,
                                         const int &span, const float &knot);

const std::vector<torch::Tensor>
basis_functions_torch(const int &degree, const torch::Tensor &knot_vector,
                      const std::vector<int> &spans,
                      const std::vector<float> &knots);

const torch::Tensor
toTorchPoints(const int &degree_u, const int &degree_v, const int &size_u,
              const int &size_v, const int &sample_num_u,
              const int &sample_num_v, const float &start_u,
              const float &start_v, const float &stop_u, const float &stop_v,
              const torch::Tensor &knotvector_u,
              const torch::Tensor &knotvector_v, const torch::Tensor &ctrlpts);
