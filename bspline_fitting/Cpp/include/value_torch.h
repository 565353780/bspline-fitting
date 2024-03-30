#pragma once

#include <torch/extension.h>

const torch::Tensor toTorchPoints(
    const std::vector<int> &degree, const std::vector<float> &u_knotvector,
    const std::vector<float> &v_knotvector, const torch::Tensor &ctrlpts,
    const std::vector<int> &size, const std::vector<float> &start,
    const std::vector<float> &stop, const std::vector<int> &sample_size);
