#include "value_torch.h"
#include "value.h"

const torch::Tensor toTorchPoints(
    const std::vector<int> &degree, const std::vector<float> &u_knotvector,
    const std::vector<float> &v_knotvector, const torch::Tensor &ctrlpts,
    const std::vector<int> &size, const std::vector<float> &start,
    const std::vector<float> &stop, const std::vector<int> &sample_size) {
  std::vector<std::vector<int>> spans(2);
  std::vector<std::vector<std::vector<float>>> basis(2);

  for (int idx = 0; idx < 2; ++idx) {
    const std::vector<float> knots =
        linspace(start[idx], stop[idx], sample_size[idx]);

    std::vector<float> knotvector;
    if (idx == 0) {
      knotvector = u_knotvector;
    } else {
      knotvector = v_knotvector;
    }

    const std::vector<int> current_spans =
        find_spans(degree[idx], knotvector, size[idx], knots);

    spans[idx] = current_spans;

    const std::vector<std::vector<float>> current_basis =
        basis_functions(degree[idx], knotvector, spans[idx], knots);

    basis[idx] = current_basis;
  }

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(ctrlpts.dtype()).device(ctrlpts.device());

  torch::Tensor eval_points =
      torch::zeros({sample_size[0] * sample_size[1], 3}, opts);

  for (size_t i = 0; i < spans[0].size(); ++i) {
    const int idx_u = spans[0][i] - degree[0];

    for (size_t j = 0; j < spans[1].size(); ++j) {
      const int idx_v = spans[1][j] - degree[1];

      torch::Tensor spt = torch::zeros({3}, opts);

      for (int k = 0; k < degree[0] + 1; ++k) {
        torch::Tensor temp = torch::zeros({3}, opts);

        for (int l = 0; l < degree[1] + 1; ++l) {
          temp = temp +
                 basis[1][j][l] * ctrlpts[idx_v + l + (size[1] * (idx_u + k))];
        }

        spt = spt + basis[0][i][k] * temp;
      }

      eval_points[i * spans[0].size() + j] = spt;
    }
  }

  return eval_points;
}
