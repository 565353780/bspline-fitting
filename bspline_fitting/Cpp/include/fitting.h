#pragma once

#include <torch/extension.h>

const std::vector<float> linspace(const float &start, const float &stop, const int &num);

const torch::Tensor toMaxValues(const int &unique_idx_num,
                                const torch::Tensor &data,
                                const torch::Tensor &data_idxs);
