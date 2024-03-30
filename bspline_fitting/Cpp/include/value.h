#pragma once

#include <torch/extension.h>

const std::vector<float> linspace(const float &start, const float &stop,
                                  const int &num);

const int find_span_linear(const int &degree,
                           const std::vector<float> &knot_vector,
                           const int &num_ctrlpts, const float &knot);

const std::vector<int> find_spans(const int &degree,
                                  const std::vector<float> &knot_vector,
                                  const int &num_ctrlpts,
                                  const std::vector<float> &knots);

const std::vector<float> basis_function(const int &degree,
                                        const std::vector<float> &knot_vector,
                                        const int &span, const float &knot);

const std::vector<std::vector<float>>
basis_functions(const int &degree, const std::vector<float> &knot_vector,
                const std::vector<int> &spans, const std::vector<float> &knots);

const std::vector<std::vector<float>>
toPoints(const std::vector<int> &degree, const std::vector<float> &u_knotvector,
         const std::vector<float> &v_knotvector,
         const std::vector<std::vector<float>> &ctrlpts,
         const std::vector<int> &size, const std::vector<float> &start,
         const std::vector<float> &stop, const std::vector<int> &sample_size);
