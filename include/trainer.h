#pragma once

#include "bspline_surface.h"
#include "scheduler.h"
#include <limits>
#include <string>
#include <torch/extension.h>
#include <torch/optim/schedulers/lr_scheduler.h>

class Trainer {
public:
  Trainer(const int &degree_u = 3, const int &degree_v = 3,
          const int &size_u = 7, const int &size_v = 7,
          const int &sample_num_u = 20, const int &sample_num_v = 20,
          const float &start_u = 0.0, const float &start_v = 0.0,
          const float &stop_u = 1.0, const float &stop_v = 1.0,
          const std::string &idx_dtype = "int64",
          const std::string &dtype = "float32",
          const std::string &device = "cpu",
          const int &warm_epoch_step_num = 20, const int &warm_epoch_num = 4,
          const int &finetune_step_num = 400, const float &lr = 5e-2,
          const float &weight_decay = 1e-4, const float &factor = 0.9,
          const int &patience = 1, const float &min_lr = 1e-3);

  const bool toBSplineSurface(const std::vector<float> &sample_points);

  const bool updateBestParams(const float &loss = -1.0,
                              const bool &force = false);

  const bool loadParams(std::vector<float> &knotvector_u,
                        std::vector<float> &knotvector_v,
                        std::vector<float> &ctrlpts);

  const float getLr(const torch::optim::AdamW &optimizer);

  const float trainStep(torch::optim::AdamW &optimizer,
                        const torch::Tensor gt_points);

  const bool checkStop(const torch::optim::AdamW &optimizer,
                       ReduceLROnPlateauScheduler &scheduler,
                       const float &loss);

  const bool trainBSplineSurface(torch::optim::AdamW &optimizer,
                                 ReduceLROnPlateauScheduler &scheduler,
                                 const torch::Tensor &gt_points);

  const bool autoTrainBSplineSurface(std::vector<float> &gt_points_vec);

  const std::vector<float> toKNotsU();
  const std::vector<float> toKNotsV();
  const std::vector<float> toCtrlPts();

  const bool isValid() { return bspline_surface_.isValid(); }

public:
  BSplineSurface bspline_surface_;

  // super params
  int warm_epoch_step_num_ = 20;
  int warm_epoch_num_ = 4;
  int finetune_step_num_ = 400;
  float lr_ = 5e-2;
  float weight_decay_ = 1e-4;
  float factor_ = 0.9;
  int patience_ = 1;
  float min_lr_ = 1e-3;

  // training state params
  int min_lr_reach_time_ = 0;
  float loss_min_ = std::numeric_limits<float>().max();
  torch::Tensor best_knots_u_;
  torch::Tensor best_knots_v_;
  torch::Tensor best_ctrlpts_;
};
