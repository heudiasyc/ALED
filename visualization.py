#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains visualization functions, which can be used to convert raw tensors of data (event
volumes, raw depth images, ...) to a humanly understandble image, which can then be saved or
displayed for overview/debug/... purposes.
"""

from math import log

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def lidar_proj_to_img_color(lidar_proj):
  """
  Converts a LiDAR projection Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [B, 4, H, W], where each LiDAR point is colored based on its depth
  """

  color_map = plt.get_cmap("inferno")
  lidar_proj_img_np = color_map(lidar_proj.cpu())
  lidar_proj_img = torch.from_numpy(lidar_proj_img_np).permute((0, 3, 1, 2))
  return lidar_proj_img


def lidar_proj_to_img_color_gray(lidar_proj):
  """
  Converts a LiDAR projection Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [B, 4, H, W], where each LiDAR point is colored based on its depth, and with a gray background
  """

  color_map = plt.get_cmap("inferno")
  lidar_proj_img_np = color_map(lidar_proj.cpu())
  lidar_proj_img = torch.from_numpy(lidar_proj_img_np)
  lidar_proj_img[:, :, :, :3][lidar_proj_img[:, :, :, :3] == lidar_proj_img[:, 0, 0, :3]] = 0.133333
  lidar_proj_img = lidar_proj_img.permute(0, 3, 1, 2)
  return lidar_proj_img


def lidar_proj_to_img_color_bigger_points(lidar_proj):
  """
  Converts a LiDAR projection Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [B, 4, H, W], where each LiDAR point is colored based on its depth, with its size slightly
  increased for a better visualization (careful, this is slower than the simpler
  lidar_proj_to_img_color variant!)
  """

  color_map = plt.get_cmap("inferno")
  lidar_proj_orig = lidar_proj.cpu()
  lidar_proj_bigger_pts = np.zeros(lidar_proj_orig.shape[1:3], np.float32)
  for y in range(lidar_proj_orig.shape[1]):
    for x in range(lidar_proj_orig.shape[2]):
      val = lidar_proj_orig[0, y, x].item()
      if val != 0.0:
        cv2.circle(lidar_proj_bigger_pts, (x, y), 1, val, cv2.FILLED)
  lidar_proj_img_np = color_map(lidar_proj_bigger_pts)
  lidar_proj_img = torch.from_numpy(lidar_proj_img_np).permute((2, 0, 1))
  return lidar_proj_img


def lidar_proj_to_img_color_bigger_points_gray(lidar_proj):
  """
  Converts a LiDAR projection Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [4, H, W], where each LiDAR point is colored based on its depth, with its size slightly
  increased for a better visualization, and with a gray background (careful, this is slower than the
  simpler lidar_proj_to_img_color variant!)
  """

  color_map = plt.get_cmap("inferno")
  lidar_proj_orig = lidar_proj.cpu()
  lidar_proj_bigger_pts = np.zeros(lidar_proj_orig.shape[1:3], np.float32)
  for y in range(lidar_proj_orig.shape[1]):
    for x in range(lidar_proj_orig.shape[2]):
      val = lidar_proj_orig[0, y, x].item()
      if val != 0.0:
        cv2.circle(lidar_proj_bigger_pts, (x, y), 1, val, cv2.FILLED)
  lidar_proj_img_np = color_map(lidar_proj_bigger_pts)
  lidar_proj_img = torch.from_numpy(lidar_proj_img_np)
  lidar_proj_img[:, :, :3][lidar_proj_img[:, :, :3] == lidar_proj_img[0, 0, :3]] = 0.133333
  lidar_proj_img = lidar_proj_img.permute(2, 0, 1)
  return lidar_proj_img


def depth_image_to_img_color(depth_image, events_mask=None):
  """
  Converts a depth image Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [B, 4, H, W], after applying a log transformation and ensuring values are in the range [0, 1]
  """

  color_map = plt.get_cmap("inferno")
  depth_image_log = depth_image.clone()
  depth_image_log[depth_image_log > 1.0] = 1.0
  depth_image_log[torch.isnan(depth_image_log)] = 0.0
  depth_image_log = torch.log(depth_image_log+1) / log(2)
  if events_mask is not None:
    depth_image_log[~events_mask] = float("nan")
  depth_image_img_np = color_map(depth_image_log.cpu())
  depth_image_img = torch.from_numpy(depth_image_img_np).permute((0, 3, 1, 2))
  return depth_image_img


def event_volume_to_img(event_volume):
  """
  Converts an event volume Tensor of shape [B, C, H, W] into a visualizable Tensor of shape
  [B, 3, H, W], where the C/2 temporal bins are squashed, and the negative events are displayed in
  blue, while the positive ones are in red
  """

  batches, bins, height, width = event_volume.size()
  event_volume_binary = (event_volume != 0)
  event_volume_img = torch.zeros((batches, 3, height, width))
  event_volume_img[:, 2, :, :] = torch.sum(event_volume_binary[:, 0:bins//2, :, :], dim=1)
  event_volume_img[:, 0, :, :] = torch.sum(event_volume_binary[:, bins//2:bins, :, :], dim=1)
  return event_volume_img


def predicted_depths_to_img_color(predicted_depths, events_mask=None):
  """
  Converts a predicted depth image Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [B, 4, H, W], after applying a log transformation and ensuring values are in the range [0, 1]
  """

  color_map = plt.get_cmap("inferno")
  predicted_depths_log = predicted_depths.clone()
  predicted_depths_log[predicted_depths_log < 0.0] = 0.0
  predicted_depths_log[predicted_depths_log > 1.0] = 1.0
  predicted_depths_log = torch.log(predicted_depths_log+1) / log(2)
  if events_mask is not None:
    predicted_depths_log[~events_mask] = float("nan")
  predicted_depths_img_np = color_map(predicted_depths_log.cpu())
  predicted_depths_img = torch.from_numpy(predicted_depths_img_np).permute((0, 3, 1, 2))
  return predicted_depths_img


def depth_difference_to_img_color(depth_diff, events_mask=None, diff_thr=1.0):
  """
  Converts a depth difference Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [B, 4, H, W], by transforming values in the range [-1, 1] to the range [0, 1]
  """

  color_map = plt.get_cmap("inferno")
  depth_diff_norm = depth_diff.clone()
  depth_diff_norm[depth_diff_norm < -diff_thr] = -1
  depth_diff_norm[depth_diff_norm > diff_thr] = 1
  depth_diff_norm += 1.0
  depth_diff_norm /= 2.0
  if events_mask is not None:
    depth_diff_norm[~events_mask] = float("nan")
  depth_diff_img_np = color_map(depth_diff_norm.cpu())
  depth_diff_img = torch.from_numpy(depth_diff_img_np).permute((0, 3, 1, 2))
  return depth_diff_img


def prediction_error_to_img_color(prediction_error):
  """
  Converts an error on the prediction Tensor of shape [B, H, W] into a visualizable Tensor of shape
  [B, 4, H, W], normalized between 0 and 1
  """

  color_map = plt.get_cmap("inferno")
  batches = prediction_error.size()[0]
  prediction_error_norm = prediction_error.clone()
  prediction_error_norm[torch.isnan(prediction_error_norm)] = 0.0
  for b in range(batches):
    prediction_error_norm[b, :, :] /= torch.max(prediction_error_norm[b, :, :])
  prediction_error_img_np = color_map(prediction_error_norm.cpu())
  prediction_error_img = torch.from_numpy(prediction_error_img_np).permute((0, 3, 1, 2))
  return prediction_error_img
