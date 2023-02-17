#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains PyTorch submodules, used by the ALED network.
"""

import torch
import torch.nn as nn


class residual_encoder(nn.Module):
  """
  A ResNet Basic encoder, with optional downsampling.
  It is composed of a two convolutions, each followed by a batch normalization and a PReLU
  activation.
  At the end, the input (potentially downsampled through a 1x1 convolution) is added to the output
  of the last convolution, such that the convolutions only compute the residual.
  Note: an optional instance normalization before the last ReLU can also be enabled, as proposed by
  Pan et al. in their "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
  article
  """

  def __init__(self, channels_in, channels_out, kernel_size, stride, padding, use_instance_norm):
    super(residual_encoder, self).__init__()
    self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
    self.bn1 = nn.BatchNorm2d(channels_out)
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size, 1, padding)
    self.bn2 = nn.BatchNorm2d(channels_out)
    self.relu = nn.PReLU()
    self.downsample = (stride != 1)
    if self.downsample:
      self.convd = nn.Conv2d(channels_in, channels_out, 1, stride)
      self.bnd = nn.BatchNorm2d(channels_out)
    self.use_instance_norm = use_instance_norm
    if self.use_instance_norm:
      self.insn = nn.InstanceNorm2d(channels_out)

  def forward(self, x):
    identity = x.clone()
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample:
      identity = self.convd(identity)
      identity = self.bnd(identity)
    out += identity
    if self.use_instance_norm:
      out = self.insn(out)
    out = self.relu(out)
    return out


class event_head(nn.Module):
  """
  The event head, transforming the input event representation to a tensor of fixed size, used as
  part of ALED network.
  It is constituted of a convolution, followed by a PReLU activation function.
  """

  def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
    super(event_head, self).__init__()
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
    self.relu = nn.PReLU()

  def forward(self, x):
    out = self.conv(x)
    out = self.relu(out)
    return out


class lidar_head(nn.Module):
  """
  The LiDAR head, transforming the input LiDAR representation to a tensor of fixed size, used as
  part of ALED network.
  It is constituted of a convolution, followed by a PReLU activation function.
  """

  def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
    super(lidar_head, self).__init__()
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
    self.relu = nn.PReLU()

  def forward(self, x):
    out = self.conv(x)
    out = self.relu(out)
    return out


class event_encoder(nn.Module):
  """
  The event encoder submodule, used as part of ALED network.
  It is composed of a single RESNet Basic block, with optional instance normalization
  """

  def __init__(self, channels_in, channels_out, kernel_size, stride, padding, use_instance_norm):
    super(event_encoder, self).__init__()
    self.res = residual_encoder(channels_in, channels_out, kernel_size, stride, padding, use_instance_norm)

  def forward(self, x):
    out = self.res(x)
    return out


class lidar_encoder(nn.Module):
  """
  The LiDAR encoder submodule, used as part of ALED network.
  It is actually the same as the event_encoder module, meaning that it is composed of a single
  RESNet Basic block, with optional instance normalization
  """

  def __init__(self, channels_in, channels_out, kernel_size, stride, padding, use_instance_norm):
    super(lidar_encoder, self).__init__()
    self.res = residual_encoder(channels_in, channels_out, kernel_size, stride, padding, use_instance_norm)

  def forward(self, x):
    out = self.res(x)
    return out


class conv_gru(nn.Module):
  """
  The Convolutional Gated Recurrent Unit (ConvGRU) submodule, used as part of ALED network.
  Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
  """

  def __init__(self, input_size, hidden_size, kernel_size):
    super().__init__()
    padding = kernel_size // 2
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
    self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
    self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    nn.init.orthogonal_(self.reset_gate.weight)
    nn.init.orthogonal_(self.update_gate.weight)
    nn.init.orthogonal_(self.out_gate.weight)
    nn.init.constant_(self.reset_gate.bias, 0.)
    nn.init.constant_(self.update_gate.bias, 0.)
    nn.init.constant_(self.out_gate.bias, 0.)

  def forward(self, x, prev_state):
    # Get batch and spatial sizes
    batch_size = x.data.size()[0]
    spatial_size = x.data.size()[2:]

    # Generate empty prev_state if None is provided
    if prev_state is None:
      state_size = [batch_size, self.hidden_size] + list(spatial_size)
      prev_state = torch.zeros(state_size, dtype=x.dtype).to(x.device)

    # Data size is [batch, channel, height, width]
    stacked_inputs = torch.cat([x, prev_state], dim=1)
    update = torch.sigmoid(self.update_gate(stacked_inputs))
    reset = torch.sigmoid(self.reset_gate(stacked_inputs))
    out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
    new_state = prev_state * (1 - update) + out_inputs * update

    return new_state


class s_conv_gru(nn.Module):
  """
  The Sparse Convolutional Gated Recurrent Unit (SConvGRU) submodule, as described in the "A Sparse
  Gating Convolutional Recurrent Network for Traffic Flow Prediction" article by Huang et al.
  This module is a slightly simplified version of the ConvGRU, allowing for a reduction of the total
  number of parameters of the network without loss of accuracy.
  """

  def __init__(self, input_size, hidden_size, kernel_size):
    super().__init__()
    padding = kernel_size // 2
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.reset_gate = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
    self.update_gate = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
    self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    nn.init.orthogonal_(self.reset_gate.weight)
    nn.init.orthogonal_(self.update_gate.weight)
    nn.init.orthogonal_(self.out_gate.weight)
    nn.init.constant_(self.reset_gate.bias, 0.)
    nn.init.constant_(self.update_gate.bias, 0.)
    nn.init.constant_(self.out_gate.bias, 0.)

  def forward(self, x, prev_state):
    # Get batch and spatial sizes
    batch_size = x.data.size()[0]
    spatial_size = x.data.size()[2:]

    # Generate empty prev_state if None is provided
    if prev_state is None:
      state_size = [batch_size, self.hidden_size] + list(spatial_size)
      prev_state = torch.zeros(state_size, dtype=x.dtype).to(x.device)

    # Data size is [batch, channel, height, width]
    update = torch.sigmoid(self.update_gate(prev_state))
    reset = torch.sigmoid(self.reset_gate(prev_state))
    out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
    new_state = prev_state * (1 - update) + out_inputs * update

    return new_state


class convex_upsampling(nn.Module):
  """
  The convex upsampling submodule, used as part of ALED network.
  It is a learnt alternative to bilinear upsampling, originally described in the "RAFT: Recurrent
  All-Pairs Field Transforms for Optical Flow" article by Z. Teed and J. Deng.
  """

  def __init__(self, channels_in_guide, upsample_factor):
    super(convex_upsampling, self).__init__()
    self.upsample_factor = upsample_factor
    self.mask_conv1 = nn.Conv2d(4*channels_in_guide, 256, 3, padding=1)
    self.mask_relu = nn.PReLU()
    self.mask_conv2 = nn.Conv2d(256, self.upsample_factor**2 * 9, 1)
    self.unfold = nn.Unfold((3, 3), padding=1)

  def forward(self, x, guide):
    # Guide reshaping
    batch_size_guide, channels_guide, height_guide, width_guide = guide.shape
    mask = torch.empty(batch_size_guide, 4*channels_guide, height_guide//2, width_guide//2).to(guide.device)
    mask[:, :channels_guide, :, :] = guide[:, :, ::2, ::2]
    mask[:, channels_guide:2*channels_guide, :, :] = guide[:, :, ::2, 1::2]
    mask[:, 2*channels_guide:3*channels_guide, :, :] = guide[:, :, 1::2, ::2]
    mask[:, 3*channels_guide:, :, :] = guide[:, :, 1::2, 1::2]

    # Mask computation
    mask = self.mask_conv1(mask)
    mask = self.mask_relu(mask)
    mask = self.mask_conv2(mask)
    mask = 0.25 * mask

    # Mask reshaping and activation function
    batch_size, channels, height, width = x.shape
    mask = mask.view(batch_size, 1, 9, self.upsample_factor, self.upsample_factor, height, width)
    mask = torch.softmax(mask, dim=2)

    # Upsampling
    x_up = self.unfold(x)
    x_up = x_up.view(batch_size, channels, 9, 1, 1, height, width)
    x_up = torch.sum(mask * x_up, dim=2)
    x_up = x_up.permute(0, 1, 4, 2, 5, 3)
    return x_up.reshape(batch_size, channels, self.upsample_factor*height, self.upsample_factor*width)


class decoder(nn.Module):
  """
  The decoder submodule, used as part of ALED network.
  It is constituted of a convex upsampling, followed by a convolution and a PReLU activation
  function
  """

  def __init__(self, channels_in, channels_in_guide, channels_out, upsample_factor, kernel_size, stride, padding):
    super(decoder, self).__init__()
    self.upsample = convex_upsampling(channels_in_guide, upsample_factor)
    self.upsample_factor = upsample_factor
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
    self.relu = nn.PReLU()

  def forward(self, x, guide):
    if self.upsample_factor > 1:
      out = self.upsample(x, guide)
    else:
      out = x
    out = self.conv(out)
    out = self.relu(out)
    return out


class prediction_layer(nn.Module):
  """
  The prediction layer, used as part of ALED network.
  It is constituted of a single convolution step
  """

  def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
    super(prediction_layer, self).__init__()
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)

  def forward(self, x):
    out = self.conv(x)
    return out
