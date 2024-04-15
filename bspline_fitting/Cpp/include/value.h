#pragma once
#include <vector>

const std::vector<double> linspace(const double &start, const double &stop,
                                   const int &num);

const int find_span_linear(const int &degree,
                           const std::vector<double> &knot_vector,
                           const int &num_ctrlpts, const double &knot);

const std::vector<int> find_spans(const int &degree,
                                  const std::vector<double> &knot_vector,
                                  const int &num_ctrlpts,
                                  const std::vector<double> &knots);

const std::vector<double> basis_function(const int &degree,
                                         const std::vector<double> &knot_vector,
                                         const int &span, const double &knot);

const std::vector<std::vector<double>>
basis_functions(const int &degree, const std::vector<double> &knot_vector,
                const std::vector<int> &spans,
                const std::vector<double> &knots);

const std::vector<std::vector<double>>
toPoints(const int &degree_u, const int &degree_v, const int &size_u,
         const int &size_v, const int &sample_num_u, const int &sample_num_v,
         const double &start_u, const double &start_v, const double &stop_u,
         const double &stop_v, const std::vector<double> &knotvector_u,
         const std::vector<double> &knotvector_v,
         const std::vector<std::vector<double>> &ctrlpts);
