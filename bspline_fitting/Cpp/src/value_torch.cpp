#include "value_torch.h"
#include "value.h"

const torch::Tensor toTorchPoints(
    const int &degree_u, const int &degree_v, const int &size_u,
    const int &size_v, const int &sample_num_u, const int &sample_num_v,
    const float &start_u, const float &start_v, const float &stop_u,
    const float &stop_v, const std::vector<float> &knotvector_u,
    const std::vector<float> &knotvector_v, const torch::Tensor &ctrlpts) {
  const std::vector<float> knots_u = linspace(start_u, stop_u, sample_num_u);
  const std::vector<float> knots_v = linspace(start_v, stop_v, sample_num_v);

  const std::vector<int> spans_u =
      find_spans(degree_u, knotvector_u, size_u, knots_u);
  const std::vector<int> spans_v =
      find_spans(degree_v, knotvector_v, size_v, knots_v);

  const std::vector<std::vector<float>> basis_u =
      basis_functions(degree_u, knotvector_u, spans_u, knots_u);

  const std::vector<std::vector<float>> basis_v =
      basis_functions(degree_v, knotvector_v, spans_v, knots_v);

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(ctrlpts.dtype()).device(ctrlpts.device());

  torch::Tensor eval_points =
      torch::zeros({sample_num_u * sample_num_v, 3}, opts);

  for (size_t i = 0; i < spans_u.size(); ++i) {
    const int idx_u = spans_u[i] - degree_u;

    for (size_t j = 0; j < spans_v.size(); ++j) {
      const int idx_v = spans_v[j] - degree_v;

      torch::Tensor spt = torch::zeros({3}, opts);

      for (int k = 0; k < degree_u + 1; ++k) {
        torch::Tensor temp = torch::zeros({3}, opts);

        for (int l = 0; l < degree_v + 1; ++l) {
          temp = temp +
                 basis_v[j][l] * ctrlpts[idx_v + l + (size_v * (idx_u + k))];
        }

        spt = spt + basis_u[i][k] * temp;
      }

      eval_points[i * spans_u.size() + j] = spt;
    }
  }

  return eval_points;
}
