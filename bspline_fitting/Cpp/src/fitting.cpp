#include "fitting.h"
#include <cmath>

const std::vector<float> linspace(const float &start, const float &stop, const int &num){
  std::vector<float> ret_vec;
  ret_vec.reserve(1);

    const float delta = stop - start;

    if(std::fabs(delta) <= 10e-8) {
      ret_vec.emplace_back(start);
      return ret_vec;
    }

    if (num < 2) {
      ret_vec.emplace_back(start);
      return ret_vec;
    }

  ret_vec.reserve(num);

    const int div = num - 1;

    for (int i = 0; i < num; ++i) {
      const float current_value = start + i * delta / div;
      ret_vec.emplace_back(current_value);
    }

    return ret_vec;
}

const torch::Tensor toMaxValues(const int &unique_idx_num,
                                const torch::Tensor &data,
                                const torch::Tensor &data_idxs) {
  return data;
}
