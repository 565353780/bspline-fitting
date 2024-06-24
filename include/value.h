#pragma once
#include <vector>

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
toPoints(const int &degree_u, const int &degree_v, const int &size_u,
         const int &size_v, const int &sample_num_u, const int &sample_num_v,
         const float &start_u, const float &start_v, const float &stop_u,
         const float &stop_v, const std::vector<float> &knotvector_u,
         const std::vector<float> &knotvector_v,
         const std::vector<std::vector<float>> &ctrlpts);
