#include "value.h"

const std::vector<float> linspace(const float &start, const float &stop,
                                  const int &num) {
  std::vector<float> ret_vec(1);

  const float delta = stop - start;

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
    const float current_value = start + i * delta / div;
    ret_vec[i] = current_value;
  }

  return ret_vec;
}

const int find_span_linear(const int &degree,
                           const std::vector<float> &knot_vector,
                           const int &num_ctrlpts, const float &knot) {
  int span = degree + 1;

  while (span < num_ctrlpts && knot_vector[span] <= knot) {
    span += 1;
  }

  return span - 1;
}

const std::vector<int> find_spans(const int &degree,
                                  const std::vector<float> &knot_vector,
                                  const int &num_ctrlpts,
                                  const std::vector<float> &knots) {
  std::vector<int> spans(knots.size());

  for (size_t i = 0; i < knots.size(); ++i) {
    const float &knot = knots[i];

    const int current_span =
        find_span_linear(degree, knot_vector, num_ctrlpts, knot);

    spans[i] = current_span;
  }

  return spans;
}

const std::vector<float> basis_function(const int &degree,
                                        const std::vector<float> &knot_vector,
                                        const int &span, const float &knot) {
  std::vector<float> left(degree + 1, 0.0);
  std::vector<float> right(degree + 1, 0.0);
  std::vector<float> N(degree + 1, 1.0);

  for (int j = 1; j < degree + 1; ++j) {
    left[j] = knot - knot_vector[span + 1 - j];
    right[j] = knot_vector[span + j] - knot;

    float saved = 0.0;

    for (int r = 0; r < j; ++r) {
      const float temp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }

    N[j] = saved;
  }

  return N;
}

const std::vector<std::vector<float>>
basis_functions(const int &degree, const std::vector<float> &knot_vector,
                const std::vector<int> &spans,
                const std::vector<float> &knots) {
  std::vector<std::vector<float>> basis(spans.size());

  for (size_t i = 0; i < spans.size(); ++i) {
    const int &span = spans[i];
    const float &knot = knots[i];

    const std::vector<float> current_basis_function =
        basis_function(degree, knot_vector, span, knot);

    basis[i] = current_basis_function;
  }

  return basis;
}

const std::vector<std::vector<int>>
toSpans(const std::vector<int> &degree, const std::vector<float> &u_knotvector,
        const std::vector<float> &v_knotvector, const std::vector<int> &size,
        const std::vector<float> &start, const std::vector<float> &stop,
        const std::vector<int> &sample_size) {
  std::vector<std::vector<int>> spans(2);

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
  }

  return spans;
}

const std::vector<std::vector<std::vector<float>>>
toBasis(const std::vector<int> &degree, const std::vector<float> &u_knotvector,
        const std::vector<float> &v_knotvector, const std::vector<float> &start,
        const std::vector<float> &stop, const std::vector<int> &sample_size,
        const std::vector<std::vector<int>> spans) {
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

    const std::vector<std::vector<float>> current_basis =
        basis_functions(degree[idx], knotvector, spans[idx], knots);

    basis[idx] = current_basis;
  }

  return basis;
}

const std::vector<std::vector<float>>
toPoints(const std::vector<int> &degree, const std::vector<float> &u_knotvector,
         const std::vector<float> &v_knotvector,
         const std::vector<std::vector<float>> &ctrlpts,
         const std::vector<int> &size, const std::vector<float> &start,
         const std::vector<float> &stop, const std::vector<int> &sample_size) {
  const std::vector<std::vector<int>> spans = toSpans(
      degree, u_knotvector, v_knotvector, size, start, stop, sample_size);

  const std::vector<std::vector<std::vector<float>>> basis = toBasis(
      degree, u_knotvector, v_knotvector, start, stop, sample_size, spans);

  std::vector<std::vector<float>> eval_points;

  for (size_t i = 0; i < spans[0].size(); ++i) {
    const int idx_u = spans[0][i] - degree[0];

    for (size_t j = 0; j < spans[1].size(); ++j) {
      const int idx_v = spans[1][j] - degree[1];

      std::vector<float> spt(3, 0.0);

      for (int k = 0; k < degree[0] + 1; ++k) {
        std::vector<float> temp(3, 0.0);

        for (int l = 0; l < degree[1] + 1; ++l) {
          for (size_t m = 0; m < temp.size(); ++m) {
            const float cp = ctrlpts[idx_v + l + (size[1] * (idx_u + k))][m];

            temp[m] += basis[1][j][l] * cp;
          }
        }

        for (size_t l = 0; l < spt.size(); ++l) {
          spt[l] += basis[0][i][k] * temp[l];
        }
      }

      eval_points.emplace_back(spt);
    }
  }

  return eval_points;
}
