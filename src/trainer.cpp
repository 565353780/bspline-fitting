#include "trainer.h"
#include "chamfer.h"
#include "constant.h"
#include <torch/optim/adamw.h>

using namespace torch::indexing;

Trainer::Trainer(const int &degree_u, const int &degree_v, const int &size_u,
                 const int &size_v, const int &sample_num_u,
                 const int &sample_num_v, const float &start_u,
                 const float &start_v, const float &stop_u, const float &stop_v,
                 const std::string &idx_dtype, const std::string &dtype,
                 const std::string &device, const int &warm_epoch_step_num,
                 const int &warm_epoch_num, const int &finetune_step_num,
                 const float &lr, const float &weight_decay,
                 const float &factor, const int &patience,
                 const float &min_lr) {
  bspline_surface_.updateSetting(degree_u, degree_v, size_u, size_v,
                                 sample_num_u, sample_num_v, start_u, start_v,
                                 stop_u, stop_v, idx_dtype, dtype, device);

  if (!isValid()) {
    std::cout << "[ERROR][Trainer::Trainer]" << std::endl;
    std::cout << "\t updateSetting for bspline_surface_ failed!" << std::endl;

    return;
  }

  warm_epoch_step_num_ = warm_epoch_step_num;
  warm_epoch_num_ = warm_epoch_num;
  finetune_step_num_ = finetune_step_num;
  lr_ = lr;
  weight_decay_ = weight_decay;
  factor_ = factor;
  patience_ = patience;
  min_lr_ = min_lr;

  return;
}

const bool Trainer::updateBestParams(const float &loss, const bool &force) {
  if (!force) {
    if (loss < 0) {
      return false;
    }

    if (loss >= loss_min_) {
      return false;
    }

    loss_min_ = loss;
  }

  best_knots_u_ = bspline_surface_.toSigmoidKnotvectorU().detach().clone();
  best_knots_v_ = bspline_surface_.toSigmoidKnotvectorV().detach().clone();
  best_ctrlpts_ = bspline_surface_.ctrlpts_.detach().clone();

  return true;
}

const bool Trainer::loadParams(std::vector<float> &knotvector_u,
                               std::vector<float> &knotvector_v,
                               std::vector<float> &ctrlpts) {
  const torch::Tensor knotvector_u_ =
      torch::from_blob(knotvector_u.data(), {long(knotvector_u.size())},
                       bspline_surface_.opts_)
          .clone();
  const torch::Tensor knotvector_v_ =
      torch::from_blob(knotvector_v.data(), {long(knotvector_v.size())},
                       bspline_surface_.opts_)
          .clone();
  const torch::Tensor ctrlpts_ =
      torch::from_blob(ctrlpts.data(), {long(ctrlpts.size())},
                       bspline_surface_.opts_)
          .clone()
          .reshape({-1, 3});

  const bool load_params_success =
      bspline_surface_.loadParams(knotvector_u_, knotvector_v_, ctrlpts_);

  if (!load_params_success) {
    std::cout << "[ERROR][Trainer::loadParams]" << std::endl;
    std::cout << "\t loadParams for bspline_surface_ failed!" << std::endl;

    return false;
  }

  const bool update_best_params_success = updateBestParams(-1.0, true);

  if (!update_best_params_success) {
    std::cout << "[ERROR][Trainer::loadParams]" << std::endl;
    std::cout << "\t updateBestParams failed!" << std::endl;

    return false;
  }

  return true;
}

const float Trainer::getLr(const torch::optim::AdamW &optimizer) {
  return optimizer.param_groups()[0].options().get_lr();
}

const float Trainer::trainStep(torch::optim::AdamW &optimizer,
                               const torch::Tensor gt_points) {
  optimizer.zero_grad();

  const torch::Tensor detect_points =
      bspline_surface_.toSamplePoints().unsqueeze(0);

  const std::vector<torch::Tensor> chamfer_distances =
      toChamferDistance(detect_points, gt_points);

  const torch::Tensor fit_dists2 = chamfer_distances[0].squeeze(0);
  const torch::Tensor coverage_dists2 = chamfer_distances[1].squeeze(0);

  const torch::Tensor fit_dists = torch::sqrt(fit_dists2 + EPSILON);
  const torch::Tensor coverage_dists = torch::sqrt(coverage_dists2 + EPSILON);

  const torch::Tensor fit_loss = torch::mean(fit_dists);
  const torch::Tensor coverage_loss = torch::mean(coverage_dists);

  const torch::Tensor loss = fit_loss + coverage_loss;

  loss.backward();

  optimizer.step();

  return loss.detach().clone().cpu().data_ptr<float>()[0];
}

const bool Trainer::checkStop(const torch::optim::AdamW &optimizer,
                              ReduceLROnPlateauScheduler &scheduler,
                              const float &loss) {
  scheduler.step(loss);

  if (getLr(optimizer) == min_lr_) {
    min_lr_reach_time_ += 1;
  }

  return min_lr_reach_time_ > patience_;
}

const bool Trainer::trainBSplineSurface(torch::optim::AdamW &optimizer,
                                        ReduceLROnPlateauScheduler &scheduler,
                                        const torch::Tensor &gt_points) {
  bspline_surface_.setGradState(true);

  std::cout << "[INFO][Trainer::trainBSplineSurface]" << std::endl;
  std::cout << "\t start optimizing bspline surface params..." << std::endl;

  for (int i = 0; i < finetune_step_num_; ++i) {
    const float loss = trainStep(optimizer, gt_points);

    updateBestParams(loss);

    if (checkStop(optimizer, scheduler, loss)) {
      break;
    }

    std::cout << "\t\t optimizing at step " << i + 1 << "/"
              << finetune_step_num_ << "\tLoss=" << loss
              << "\tLr=" << getLr(optimizer) << std::endl;
  }

  std::cout << std::endl;

  return true;
}

const bool Trainer::autoTrainBSplineSurface(std::vector<float> &gt_points_vec) {
  torch::Tensor gt_points =
      torch::from_blob(gt_points_vec.data(), {long(gt_points_vec.size())},
                       torch::TensorOptions()
                           .dtype(bspline_surface_.dtype_)
                           .device(torch::kCPU))
          .clone()
          .reshape({-1, 3});

  if (bspline_surface_.device_ == torch::kCUDA) {
    gt_points = gt_points.cuda();
  }

  const torch::Tensor max_point = std::get<0>(torch::max(gt_points, 0));
  const torch::Tensor min_point = std::get<0>(torch::min(gt_points, 0));
  const torch::Tensor center = (max_point + min_point) / 2.0;
  const float scale = torch::max(max_point - min_point).item<float>();

  if (scale == 0.0) {
    std::cout << "[ERROR][Trainer::autoTrainBSplineSurface]" << std::endl;
    std::cout << "\t the input points are all the same point!" << std::endl;

    best_ctrlpts_ = bspline_surface_.ctrlpts_.detach().clone();
    best_ctrlpts_.index_put_({Slice(None), 0}, gt_points[0][0]);
    best_ctrlpts_.index_put_({Slice(None), 1}, gt_points[0][1]);
    best_ctrlpts_.index_put_({Slice(None), 2}, gt_points[0][2]);

    return true;
  }

  gt_points = (gt_points - center) / scale;

  torch::Tensor ctrlpts = torch::zeros(
      {bspline_surface_.size_u_ - 1, bspline_surface_.size_v_ - 1, 3},
      bspline_surface_.opts_);

  torch::Tensor u_values = torch::arange(bspline_surface_.size_u_ - 1) /
                               (bspline_surface_.size_u_ - 2) -
                           0.5;

  torch::Tensor v_values = torch::arange(bspline_surface_.size_v_ - 1) /
                               (bspline_surface_.size_v_ - 2) -
                           0.5;

  for (int i = 0; i < bspline_surface_.size_u_ - 1; ++i) {
    ctrlpts.index_put_(
        {Slice(None), i, 0},
        u_values.toType(bspline_surface_.dtype_).to(bspline_surface_.device_));
  }

  for (int i = 0; i < bspline_surface_.size_v_ - 1; ++i) {
    ctrlpts.index_put_(
        {i, Slice(None), 1},
        v_values.toType(bspline_surface_.dtype_).to(bspline_surface_.device_));
  }

  ctrlpts = ctrlpts.reshape({-1, 3}) * 0.4;

  bspline_surface_.loadCtrlPts(ctrlpts);

  gt_points = gt_points.unsqueeze(0);

  torch::optim::AdamW optimizer(
      std::vector<torch::Tensor>({bspline_surface_.ctrlpts_}),
      torch::optim::AdamWOptions().lr(lr_).weight_decay(weight_decay_));

  ReduceLROnPlateauScheduler scheduler(optimizer, factor_, patience_, min_lr_);

  trainBSplineSurface(optimizer, scheduler, gt_points);

  bspline_surface_.setGradState(false);

  bspline_surface_.ctrlpts_ = bspline_surface_.ctrlpts_ * scale + center;

  return true;
}

const std::vector<float> Trainer::toKNotsU() {
  const torch::Tensor sigmoid_knotvector_u =
      bspline_surface_.toSigmoidKnotvectorU().detach().clone().cpu();

  return std::vector<float>(sigmoid_knotvector_u.data_ptr<float>(),
                            sigmoid_knotvector_u.data_ptr<float>() +
                                sigmoid_knotvector_u.numel());
}

const std::vector<float> Trainer::toKNotsV() {
  const torch::Tensor sigmoid_knotvector_v =
      bspline_surface_.toSigmoidKnotvectorV().detach().clone().cpu();

  return std::vector<float>(sigmoid_knotvector_v.data_ptr<float>(),
                            sigmoid_knotvector_v.data_ptr<float>() +
                                sigmoid_knotvector_v.numel());
}

const std::vector<float> Trainer::toCtrlPts() {
  const torch::Tensor flatten_ctrlpts =
      bspline_surface_.ctrlpts_.detach().clone().cpu().reshape({-1});

  return std::vector<float>(flatten_ctrlpts.data_ptr<float>(),
                            flatten_ctrlpts.data_ptr<float>() +
                                flatten_ctrlpts.numel());
}
