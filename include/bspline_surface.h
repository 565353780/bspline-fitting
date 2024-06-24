#pragma once

#include <torch/extension.h>

class BSplineSurface {
public:
  BSplineSurface(const int &degree_u = 3, const int &degree_v = 3,
                 const int &size_u = 7, const int &size_v = 7,
                 const int &sample_num_u = 20, const int &sample_num_v = 20,
                 const float &start_u = 0.0, const float &start_v = 0.0,
                 const float &stop_u = 1.0, const float &stop_v = 1.0,
                 const std::string &idx_dtype = "int64",
                 const std::string &dtype = "float32",
                 const std::string &device = "cpu");

  const bool updateSetting(
      const int &degree_u = 3, const int &degree_v = 3, const int &size_u = 7,
      const int &size_v = 7, const int &sample_num_u = 20,
      const int &sample_num_v = 20, const float &start_u = 0.0,
      const float &start_v = 0.0, const float &stop_u = 1.0,
      const float &stop_v = 1.0, const std::string &idx_dtype = "int64",
      const std::string &dtype = "float32", const std::string &device = "cpu");

  const bool setGradState(const bool &need_grad);

  const bool loadCtrlPts(const torch::Tensor &ctrlpts);

  const bool loadParams(const torch::Tensor &knotvector_u,
                        const torch::Tensor &knotvector_v,
                        const torch::Tensor &ctrlpts);

  const torch::Tensor toSigmoidKnotvectorU();
  const torch::Tensor toSigmoidKnotvectorV();

  const torch::Tensor toSamplePoints();

  const bool isValid() { return is_valid_; }

public:
  // supar params
  int degree_u_ = 3;
  int degree_v_ = 3;
  int size_u_ = 7;
  int size_v_ = 7;
  int sample_num_u_ = 20;
  int sample_num_v_ = 20;
  float start_u_ = 0.0;
  float start_v_ = 0.0;
  float stop_u_ = 1.0;
  float stop_v_ = 1.0;
  torch::Dtype idx_dtype_ = torch::kInt64;
  torch::Dtype dtype_ = torch::kFloat32;
  torch::Device device_ = torch::kCPU;

  torch::TensorOptions idx_opts_;
  torch::TensorOptions opts_;

  // diff params
  torch::Tensor knotvector_u_;
  torch::Tensor knotvector_v_;
  torch::Tensor ctrlpts_;

  bool is_valid_ = true;
};
