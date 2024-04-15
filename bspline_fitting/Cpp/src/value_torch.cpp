#include "value_torch.h"
#include "value.h"

const std::vector<torch::Tensor>
basis_function_torch(const int &degree, const torch::Tensor &knot_vector,
                     const int &span, const double &knot) {
  const torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(knot_vector.dtype())
                                        .device(knot_vector.device());

  std::vector<torch::Tensor> left(degree + 1, torch::zeros({1}, opts));
  std::vector<torch::Tensor> right(degree + 1, torch::zeros({1}, opts));
  std::vector<torch::Tensor> N(degree + 1, torch::zeros({1}, opts));

  for (int j = 1; j < degree + 1; ++j) {
    left[j] = knot - knot_vector[span + 1 - j];
    right[j] = knot_vector[span + j] - knot;

    torch::Tensor saved = torch::zeros({1}, opts);

    for (int r = 0; r < j; ++r) {
      const torch::Tensor temp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }

    N[j] = saved;
  }

  return N;
}

const std::vector<std::vector<torch::Tensor>>
basis_functions_torch(const int &degree, const torch::Tensor &knot_vector,
                      const std::vector<int> &spans,
                      const std::vector<double> &knots) {
  std::vector<std::vector<torch::Tensor>> basis;
  basis.reserve(spans.size());

  for (size_t i = 0; i < spans.size(); ++i) {
    const int &span = spans[i];
    const double &knot = knots[i];

    const std::vector<torch::Tensor> current_basis_function =
        basis_function_torch(degree, knot_vector, span, knot);

    basis.emplace_back(current_basis_function);
  }

  return basis;
}

const torch::Tensor
toTorchPoints(const int &degree_u, const int &degree_v, const int &size_u,
              const int &size_v, const int &sample_num_u,
              const int &sample_num_v, const double &start_u,
              const double &start_v, const double &stop_u, const double &stop_v,
              const torch::Tensor &knotvector_u,
              const torch::Tensor &knotvector_v, const torch::Tensor &ctrlpts) {
  const std::vector<double> knots_u = linspace(start_u, stop_u, sample_num_u);
  const std::vector<double> knots_v = linspace(start_v, stop_v, sample_num_v);

  const std::vector<double> knotvector_u_vec(knotvector_u.data_ptr<double>(),
                                             knotvector_u.data_ptr<double>() +
                                                 knotvector_u.numel());
  const std::vector<double> knotvector_v_vec(knotvector_v.data_ptr<double>(),
                                             knotvector_v.data_ptr<double>() +
                                                 knotvector_v.numel());

  const std::vector<int> spans_u =
      find_spans(degree_u, knotvector_u_vec, size_u, knots_u);
  const std::vector<int> spans_v =
      find_spans(degree_v, knotvector_v_vec, size_v, knots_v);

  const std::vector<std::vector<torch::Tensor>> basis_u =
      basis_functions_torch(degree_u, knotvector_u, spans_u, knots_u);

  const std::vector<std::vector<torch::Tensor>> basis_v =
      basis_functions_torch(degree_v, knotvector_v, spans_v, knots_v);

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
