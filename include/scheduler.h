#pragma once

#include <torch/extension.h>

class ReduceLROnPlateauScheduler {
public:
  ReduceLROnPlateauScheduler(torch::optim::Optimizer &optimizer, float factor,
                             int patience, float min_lr)
      : optimizer_(optimizer), factor_(factor), patience_(patience),
        min_lr_(min_lr), wait_(0),
        best_loss_(std::numeric_limits<float>::max()) {}

  void step(float current_loss) {
    if (current_loss < best_loss_) {
      best_loss_ = current_loss;
      wait_ = 0;
    } else {
      wait_++;
      if (wait_ >= patience_) {
        adjust_lr();
        wait_ = 0;
      }
    }
  }

  void adjust_lr() {
    for (auto &param_group : optimizer_.param_groups()) {
      float old_lr = param_group.options().get_lr();
      float new_lr = std::max(old_lr * factor_, min_lr_);
      param_group.options().set_lr(new_lr);
      // std::cout << "Adjusting learning rate to: " << new_lr << std::endl;
    }
  }

private:
  torch::optim::Optimizer &optimizer_;
  float factor_;
  int patience_;
  float min_lr_;
  int wait_;
  float best_loss_;
};
