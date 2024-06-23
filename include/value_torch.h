#pragma once

#include <torch/extension.h>

const std::vector<torch::Tensor>
basis_function_torch(const int &degree, const torch::Tensor &knot_vector,
                     const int &span, const double &knot);

const std::vector<std::vector<torch::Tensor>>
basis_functions_torch(const int &degree, const torch::Tensor &knot_vector,
                      const std::vector<int> &spans,
                      const std::vector<double> &knots);

const torch::Tensor
toTorchPoints(const int &degree_u, const int &degree_v, const int &size_u,
              const int &size_v, const int &sample_num_u,
              const int &sample_num_v, const double &start_u,
              const double &start_v, const double &stop_u, const double &stop_v,
              const torch::Tensor &knotvector_u,
              const torch::Tensor &knotvector_v, const torch::Tensor &ctrlpts);
