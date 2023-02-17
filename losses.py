#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains PyTorch losses, which can be used for training the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicL1MultiscaleGradLoss(nn.Module):
  """
  The traditional L1 loss, with the addition of a multi-scale gradient matching term, as used in the
  "3D Ken Burns Effect from a Single Image" article by Niklaus et al.
  "Dynamic" = weights of the L1 loss and multicale gradient matching loss can be adjusted on the go
  """

  def __init__(self, scales):
    super(DynamicL1MultiscaleGradLoss, self).__init__()
    self.scales = scales

  def forward(self, input, target):
    # The L1 loss is firstly computed
    l1_loss = F.l1_loss(input, target, reduction="sum")

    # We then compute the multiscale gradient matching loss (eq. 2 of the reference paper)
    input_abs = torch.abs(input)
    target_abs = torch.abs(target)
    eps = 1e-6
    ms_grad_match_loss = 0.0
    for scale in range(self.scales):
      shift_px = 2**scale
      shifted_input_x = input.clone()
      shifted_input_y = input.clone()
      shifted_target_x = target.clone()
      shifted_target_y = target.clone()
      shifted_input_x[:, :, :, :-shift_px] = input[:, :, :, shift_px:]
      shifted_input_y[:, :, :-shift_px, :] = input[:, :, shift_px:, :]
      shifted_target_x[:, :, :, :-shift_px] = target[:, :, :, shift_px:]
      shifted_target_y[:, :, :-shift_px, :] = target[:, :, shift_px:, :]
      scale_invar_grad_input_x = (shifted_input_x-input)/(torch.abs(shifted_input_x)+input_abs+eps)
      scale_invar_grad_input_y = (shifted_input_y-input)/(torch.abs(shifted_input_y)+input_abs+eps)
      scale_invar_grad_target_x = (shifted_target_x-target)/(torch.abs(shifted_target_x)+target_abs+eps)
      scale_invar_grad_target_y = (shifted_target_y-target)/(torch.abs(shifted_target_y)+target_abs+eps)
      ms_grad_match_loss += F.mse_loss(scale_invar_grad_input_x, scale_invar_grad_target_x, reduction="sum")
      ms_grad_match_loss += F.mse_loss(scale_invar_grad_input_y, scale_invar_grad_target_y, reduction="sum")

    # Finally, we return the L1 loss and the multiscale gradient matching loss
    return l1_loss, ms_grad_match_loss


class DynamicL1MultiscaleGradLossScaled(nn.Module):
  """
  Similar to DynamicL1MultiscaleGradLoss, but without the use of the scale invariant term
  """

  def __init__(self, scales):
    super(DynamicL1MultiscaleGradLossScaled, self).__init__()
    self.scales = scales

  def forward(self, input, target):
    # The L1 loss is firstly computed
    not_nan_mask = (~torch.isnan(target))
    l1_loss = F.l1_loss(input[not_nan_mask], target[not_nan_mask], reduction="sum")

    # We then compute the multiscale gradient matching loss (eq. 2 of the reference paper), without
    # the use of the scale invariant term
    ms_grad_match_loss = 0.0
    for scale in range(self.scales):
      shift_px = 2**scale
      shifted_input_x = input.clone()
      shifted_input_y = input.clone()
      shifted_target_x = target.clone()
      shifted_target_y = target.clone()
      shifted_input_x[:, :, :, :-shift_px] = input[:, :, :, shift_px:]
      shifted_input_y[:, :, :-shift_px, :] = input[:, :, shift_px:, :]
      shifted_target_x[:, :, :, :-shift_px] = target[:, :, :, shift_px:]
      shifted_target_y[:, :, :-shift_px, :] = target[:, :, shift_px:, :]
      scale_invar_grad_input_x = shifted_input_x-input
      scale_invar_grad_input_y = shifted_input_y-input
      scale_invar_grad_target_x = shifted_target_x-target
      scale_invar_grad_target_y = shifted_target_y-target
      not_nan_mask_x = (~torch.isnan(scale_invar_grad_target_x))
      not_nan_mask_y = (~torch.isnan(scale_invar_grad_target_y))
      ms_grad_match_loss += F.mse_loss(scale_invar_grad_input_x[not_nan_mask_x], scale_invar_grad_target_x[not_nan_mask_x], reduction="sum")
      ms_grad_match_loss += F.mse_loss(scale_invar_grad_input_y[not_nan_mask_y], scale_invar_grad_target_y[not_nan_mask_y], reduction="sum")

    # Finally, we return the L1 loss and the multiscale gradient matching loss
    return l1_loss, ms_grad_match_loss
