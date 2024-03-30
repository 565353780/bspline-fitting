#pragma once

#include <torch/extension.h>

const torch::Tensor toTorchPoints(
    const int &degree_u, const int &degree_v, const int &size_u,
    const int &size_v, const int &sample_num_u, const int &sample_num_v,
    const float &start_u, const float &start_v, const float &stop_u,
    const float &stop_v, const std::vector<float> &knotvector_u,
    const std::vector<float> &knotvector_v, const torch::Tensor &ctrlpts);
