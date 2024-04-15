#include "value.h"
#include <cmath>

const std::vector<double> linspace(const double &start, const double &stop,
                                   const int &num) {
  std::vector<double> ret_vec(1);

  const double delta = stop - start;

  if (std::fabs(delta) <= 10e-8) {
    ret_vec[0] = start;
    return ret_vec;
  }

  if (num < 2) {
    ret_vec[0] = start;
    return ret_vec;
  }

  ret_vec.resize(num);

  const int div = num - 1;

  for (int i = 0; i < num; ++i) {
    const double current_value = start + i * delta / div;
    ret_vec[i] = current_value;
  }

  return ret_vec;
}

const int find_span_linear(const int &degree,
                           const std::vector<double> &knot_vector,
                           const int &num_ctrlpts, const double &knot) {
  int span = degree + 1;

  while (span < num_ctrlpts && knot_vector[span] <= knot) {
    span += 1;
  }

  return span - 1;
}

const std::vector<int> find_spans(const int &degree,
                                  const std::vector<double> &knot_vector,
                                  const int &num_ctrlpts,
                                  const std::vector<double> &knots) {
  std::vector<int> spans(knots.size());

  for (size_t i = 0; i < knots.size(); ++i) {
    const double &knot = knots[i];

    const int current_span =
        find_span_linear(degree, knot_vector, num_ctrlpts, knot);

    spans[i] = current_span;
  }

  return spans;
}

const std::vector<double> basis_function(const int &degree,
                                         const std::vector<double> &knot_vector,
                                         const int &span, const double &knot) {
  std::vector<double> left(degree + 1, 0.0);
  std::vector<double> right(degree + 1, 0.0);
  std::vector<double> N(degree + 1, 1.0);

  for (int j = 1; j < degree + 1; ++j) {
    left[j] = knot - knot_vector[span + 1 - j];
    right[j] = knot_vector[span + j] - knot;

    double saved = 0.0;

    for (int r = 0; r < j; ++r) {
      const double temp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }

    N[j] = saved;
  }

  return N;
}

const std::vector<std::vector<double>>
basis_functions(const int &degree, const std::vector<double> &knot_vector,
                const std::vector<int> &spans,
                const std::vector<double> &knots) {
  std::vector<std::vector<double>> basis(spans.size());

  for (size_t i = 0; i < spans.size(); ++i) {
    const int &span = spans[i];
    const double &knot = knots[i];

    const std::vector<double> current_basis_function =
        basis_function(degree, knot_vector, span, knot);

    basis[i] = current_basis_function;
  }

  return basis;
}

const std::vector<std::vector<double>>
toPoints(const int &degree_u, const int &degree_v, const int &size_u,
         const int &size_v, const int &sample_num_u, const int &sample_num_v,
         const double &start_u, const double &start_v, const double &stop_u,
         const double &stop_v, const std::vector<double> &knotvector_u,
         const std::vector<double> &knotvector_v,
         const std::vector<std::vector<double>> &ctrlpts) {
  const std::vector<double> knots_u = linspace(start_u, stop_u, sample_num_u);
  const std::vector<double> knots_v = linspace(start_v, stop_v, sample_num_v);

  const std::vector<int> spans_u =
      find_spans(degree_u, knotvector_u, size_u, knots_u);
  const std::vector<int> spans_v =
      find_spans(degree_v, knotvector_v, size_v, knots_v);

  const std::vector<std::vector<double>> basis_u =
      basis_functions(degree_u, knotvector_u, spans_u, knots_u);
  const std::vector<std::vector<double>> basis_v =
      basis_functions(degree_v, knotvector_v, spans_v, knots_v);

  std::vector<std::vector<double>> eval_points;

  for (size_t i = 0; i < spans_u.size(); ++i) {
    const int idx_u = spans_u[i] - degree_u;

    for (size_t j = 0; j < spans_v.size(); ++j) {
      const int idx_v = spans_v[j] - degree_v;

      std::vector<double> spt(3, 0.0);

      for (int k = 0; k < degree_u + 1; ++k) {
        std::vector<double> temp(3, 0.0);

        for (int l = 0; l < degree_v + 1; ++l) {
          for (size_t m = 0; m < temp.size(); ++m) {
            const double cp = ctrlpts[idx_v + l + (size_v * (idx_u + k))][m];

            temp[m] += basis_v[j][l] * cp;
          }
        }

        for (size_t l = 0; l < spt.size(); ++l) {
          spt[l] += basis_u[i][k] * temp[l];
        }
      }

      eval_points.emplace_back(spt);
    }
  }

  return eval_points;
}
