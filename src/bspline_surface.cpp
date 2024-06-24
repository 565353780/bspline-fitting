#include "bspline_surface.h"
#include "value_torch.h"

using namespace torch::indexing;

BSplineSurface::BSplineSurface(const int &degree_u, const int &degree_v,
                               const int &size_u, const int &size_v,
                               const int &sample_num_u, const int &sample_num_v,
                               const float &start_u, const float &start_v,
                               const float &stop_u, const float &stop_v,
                               const std::string &idx_dtype,
                               const std::string &dtype,
                               const std::string &device) {
  is_valid_ = updateSetting(degree_u, degree_v, size_u, size_v, sample_num_u,
                            sample_num_v, start_u, start_v, stop_u, stop_v,
                            idx_dtype, dtype, device);

  if (!is_valid_) {
    std::cout << "[ERROR][BSplineSurface::BSplineSurface]" << std::endl;
    std::cout << "\t updateSetting failed!" << std::endl;

    return;
  }

  return;
}

const bool BSplineSurface::updateSetting(
    const int &degree_u, const int &degree_v, const int &size_u,
    const int &size_v, const int &sample_num_u, const int &sample_num_v,
    const float &start_u, const float &start_v, const float &stop_u,
    const float &stop_v, const std::string &idx_dtype, const std::string &dtype,
    const std::string &device) {
  degree_u_ = degree_u;
  degree_v_ = degree_v;
  size_u_ = size_u;
  size_v_ = size_v;
  sample_num_u_ = sample_num_u;
  sample_num_v_ = sample_num_v;
  start_u_ = start_u;
  start_v_ = start_v;
  stop_u_ = stop_u;
  stop_v_ = stop_v;

  if (idx_dtype == "int64") {
    idx_dtype_ = torch::kInt64;
  } else if (idx_dtype == "int32") {
    idx_dtype_ = torch::kInt32;
  } else {
    std::cout << "[ERROR][BSplineSurface::updateSetting]" << std::endl;
    std::cout << "\t idx dtype not valid!" << std::endl;

    return false;
  }

  if (dtype == "float64") {
    dtype_ = torch::kFloat64;
  } else if (dtype == "float32") {
    dtype_ = torch::kFloat32;
  } else {
    std::cout << "[ERROR][BSplineSurface::updateSetting]" << std::endl;
    std::cout << "\t dtype not valid!" << std::endl;

    return false;
  }

  if (device == "cpu") {
    device_ = torch::kCPU;
  } else if (device == "cuda") {
    device_ = torch::kCUDA;
  } else {
    std::cout << "[ERROR][BSplineSurface::updateSetting]" << std::endl;
    std::cout << "\t device not valid!" << std::endl;

    return false;
  }

  idx_opts_ = torch::TensorOptions().dtype(idx_dtype_).device(device_);
  opts_ = torch::TensorOptions().dtype(dtype_).device(device_);

  knotvector_u_ = torch::zeros({size_u_ - degree_u_ - 1}, opts_);
  knotvector_v_ = torch::zeros({size_v_ - degree_v_ - 1}, opts_);
  ctrlpts_ = torch::zeros({(size_u_ - 1) * (size_v_ - 1), 3}, opts_);

  return true;
}

const bool BSplineSurface::setGradState(const bool &need_grad) {
  knotvector_u_.set_requires_grad(need_grad);
  knotvector_v_.set_requires_grad(need_grad);
  ctrlpts_.set_requires_grad(need_grad);
  return true;
}

const bool BSplineSurface::loadCtrlPts(const torch::Tensor &ctrlpts) {
  ctrlpts_ = ctrlpts.detach().clone().toType(idx_dtype_).to(device_);
  return true;
}

const bool BSplineSurface::loadParams(const torch::Tensor &knotvector_u,
                                      const torch::Tensor &knotvector_v,
                                      const torch::Tensor &ctrlpts) {
  knotvector_u_ = knotvector_u.detach().clone().toType(idx_dtype_).to(device_);
  knotvector_v_ = knotvector_v.detach().clone().toType(idx_dtype_).to(device_);
  loadCtrlPts(ctrlpts);
  return true;
}

const torch::Tensor BSplineSurface::toSigmoidKnotvectorU() {
  torch::Tensor full_knotvector_u = torch::zeros({degree_u_ + size_u_}, opts_);

  full_knotvector_u.index_put_({Slice(size_u_ - 1, degree_u_ + size_u_)}, 1.0);

  const torch::Tensor sigmoid_knotvector_u = torch::sigmoid(knotvector_u_);

  const torch::Tensor sigmoid_knotvector_u_sum =
      torch::sum(sigmoid_knotvector_u);

  const torch::Tensor normed_sigmoid_knotvector_u =
      sigmoid_knotvector_u / sigmoid_knotvector_u_sum;

  for (int i = 0; i < normed_sigmoid_knotvector_u.size(0) - 1; ++i) {
    full_knotvector_u[degree_u_ + 1 + i] =
        full_knotvector_u[degree_u_ + i] + normed_sigmoid_knotvector_u[i];
  }

  return full_knotvector_u;
}

const torch::Tensor BSplineSurface::toSigmoidKnotvectorV() {
  torch::Tensor full_knotvector_v = torch::zeros({degree_v_ + size_v_}, opts_);

  full_knotvector_v.index_put_({Slice(size_v_ - 1, degree_v_ + size_v_)}, 1.0);

  const torch::Tensor sigmoid_knotvector_v = torch::sigmoid(knotvector_v_);

  const torch::Tensor sigmoid_knotvector_v_sum =
      torch::sum(sigmoid_knotvector_v);

  const torch::Tensor normed_sigmoid_knotvector_v =
      sigmoid_knotvector_v / sigmoid_knotvector_v_sum;

  for (int i = 0; i < normed_sigmoid_knotvector_v.size(0) - 1; ++i) {
    full_knotvector_v[degree_v_ + 1 + i] =
        full_knotvector_v[degree_v_ + i] + normed_sigmoid_knotvector_v[i];
  }

  return full_knotvector_v;
}

const torch::Tensor BSplineSurface::toSamplePoints() {
  const torch::Tensor sigmoid_knotvector_u = toSigmoidKnotvectorU();
  const torch::Tensor sigmoid_knotvector_v = toSigmoidKnotvectorV();

  const torch::Tensor sample_points = toTorchPoints(
      degree_u_, degree_v_, size_u_ - 1, size_v_ - 1, sample_num_u_,
      sample_num_v_, start_u_, start_v_, stop_u_, stop_v_, sigmoid_knotvector_u,
      sigmoid_knotvector_v, ctrlpts_);

  return sample_points;
}
