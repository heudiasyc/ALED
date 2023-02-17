#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the PyTorch code for the ALED network, as described in the "Learning to Estimate
Two Dense Depths from LiDAR and Event Data" article.
"""

import torch
import torch.nn as nn

from submodules import event_head, lidar_head, event_encoder, lidar_encoder, conv_gru, s_conv_gru, \
                       residual_encoder, decoder, prediction_layer


class ALED(nn.Module):
  """
  The ALED network, as described in the article.
  It is composed of two branches, one for the events, the other one for the LiDAR scans, and
  uses Convolutional Gated Recurrent Units (convGRU) for fusion and memory purposes.
  """

  def __init__(self, channels_in_events, channels_in_lidar):
    super(ALED, self).__init__()

    # The event head
    self.event_head = event_head(channels_in_events, 32, 5, 1, 2)

    # The LiDAR head
    self.lidar_head = lidar_head(channels_in_lidar, 32, 5, 1, 2)

    # The 3 event encoders
    self.event_encoder1 = event_encoder(32, 64, 5, 2, 2, True)
    self.event_encoder2 = event_encoder(64, 128, 5, 2, 2, True)
    self.event_encoder3 = event_encoder(128, 256, 5, 2, 2, False)

    # The 3 LiDAR encoders
    self.lidar_encoder1 = lidar_encoder(32, 64, 5, 2, 2, True)
    self.lidar_encoder2 = lidar_encoder(64, 128, 5, 2, 2, True)
    self.lidar_encoder3 = lidar_encoder(128, 256, 5, 2, 2, False)

    # The 4 convGRU blocks for the events
    self.conv_gru_events0 = conv_gru(32, 32+32, 3)
    self.conv_gru_events1 = conv_gru(64, 64+64, 3)
    self.conv_gru_events2 = conv_gru(128, 128+128, 3)
    self.conv_gru_events3 = conv_gru(256, 256, 3)

    # The 4 convGRU blocks for the LiDAR
    self.conv_gru_lidar0 = conv_gru(32, 32+32, 3)
    self.conv_gru_lidar1 = conv_gru(64, 64+64, 3)
    self.conv_gru_lidar2 = conv_gru(128, 128+128, 3)
    self.conv_gru_lidar3 = conv_gru(256, 256, 3)

    # The 2 residual blocks
    self.residual_block1 = residual_encoder(256, 256, 3, 1, 1, False)
    self.residual_block2 = residual_encoder(256, 256, 3, 1, 1, False)

    # The 3 decoders
    self.decoder1 = decoder(256, 128, 128, 2, 5, 1, 2)
    self.decoder2 = decoder(128, 64, 64, 2, 5, 1, 2)
    self.decoder3 = decoder(64, 32, 32, 2, 5, 1, 2)

    # The 3 convolutions used to reduce the number of channels after concatenating the decoded state
    # and the hidden state of the corresponding convGRU module
    self.conv_concat1 = nn.Conv2d(256, 128, 1)
    self.conv_concat2 = nn.Conv2d(128, 64, 1)
    self.conv_concat3 = nn.Conv2d(64, 32, 1)

    # The final prediction layer
    self.prediction_layer = prediction_layer(32, 2, 1, 1, 0)


  def forward(self, x, conv_gru_states, forward_type):
    """
    To allow for the use of nn.DataParallel, a single forward() function has to be defined.
    Therefore, this function serves as a wrapper, and can either be called:
    - with events for the `x` input, and `forward_type` set to `events`
    - with lidar data for the `x` input, and `forward_type` set to `lidar`
    - with `forward_type` set to `predict` (and the value of 'x' is ignored)
    """
    if forward_type not in ("events", "lidar", "predict"):
      raise ValueError("'forward_type' must either be 'events', 'lidar', or 'predict'")
    if forward_type == "events":
      conv_gru_states = self.forward_events(x, conv_gru_states)
      return conv_gru_states
    if forward_type == "lidar":
      conv_gru_states = self.forward_lidar(x, conv_gru_states)
      return conv_gru_states
    if forward_type == "predict":
      pred = self.forward_predict(conv_gru_states)
      return pred


  def forward_events(self, x, conv_gru_states):
    # We first apply the head, to go from N layers to 32, and give the result to the top level
    # convGRU to update its state
    out = self.event_head(x)
    conv_gru_states[0] = self.conv_gru_events0(out, conv_gru_states[0])

    # We apply the first encoder and give it to the convGRU to update its state
    out = self.event_encoder1(out)
    conv_gru_states[1] = self.conv_gru_events1(out, conv_gru_states[1])

    # We apply the second encoder and give it to the convGRU to update its state
    out = self.event_encoder2(out)
    conv_gru_states[2] = self.conv_gru_events2(out, conv_gru_states[2])

    # We apply the third encoder and give it to the convGRU to update its state
    out = self.event_encoder3(out)
    conv_gru_states[3] = self.conv_gru_events3(out, conv_gru_states[3])

    # We return the convGRU states so that they can be saved
    return conv_gru_states


  def forward_lidar(self, x, conv_gru_states):
    # We first apply the head, to go from M layers to 32, and give the result to the top level
    # convGRU to update its state
    out = self.lidar_head(x)
    conv_gru_states[0] = self.conv_gru_lidar0(out, conv_gru_states[0])

    # We apply the first encoder and give it to the convGRU to update its state
    out = self.lidar_encoder1(out)
    conv_gru_states[1] = self.conv_gru_lidar1(out, conv_gru_states[1])

    # We apply the second encoder and give it to the convGRU to update its state
    out = self.lidar_encoder2(out)
    conv_gru_states[2] = self.conv_gru_lidar2(out, conv_gru_states[2])

    # We apply the third encoder and give it to the convGRU to update its state
    out = self.lidar_encoder3(out)
    conv_gru_states[3] = self.conv_gru_lidar3(out, conv_gru_states[3])

    # We return the convGRU states so that they can be saved
    return conv_gru_states


  def forward_predict(self, conv_gru_states):
    # The input is the saved state of the last convGRU module
    x = conv_gru_states[3]

    # We apply the two residual blocks
    out = self.residual_block1(x)
    out = self.residual_block2(out)

    # We decompose the third convGRU state in two parts: a "prediction" part and an "upsampling
    # mask" part
    conv_gru_state_2_pred = conv_gru_states[2][:, :128, :, :]
    conv_gru_state_2_mask = conv_gru_states[2][:, 128:, :, :]

    # We apply the first decoder, guided by the upsampling mask from the third convGRU hidden state
    out = self.decoder1(out, conv_gru_state_2_mask)

    # We concatenate the prediction from the third convGRU module, and apply the convolution to go
    # from 256 to 128 channels
    out = torch.concat((out, conv_gru_state_2_pred), dim=1)
    out = self.conv_concat1(out)

    # We decompose the second convGRU state in two parts: a "prediction" part and an "upsampling
    # mask" part
    conv_gru_state_1_pred = conv_gru_states[1][:, :64, :, :]
    conv_gru_state_1_mask = conv_gru_states[1][:, 64:, :, :]

    # We apply the second decoder, guided by the upsampling mask from the second convGRU hidden
    # state
    out = self.decoder2(out, conv_gru_state_1_mask)

    # We concatenate the prediction from the second convGRU module, and apply the convolution to go
    # from 128 to 64 channels
    out = torch.concat((out, conv_gru_state_1_pred), dim=1)
    out = self.conv_concat2(out)

    # We decompose the first convGRU state in two parts: a "prediction" part and an "upsampling
    # mask" part
    conv_gru_state_0_pred = conv_gru_states[0][:, :32, :, :]
    conv_gru_state_0_mask = conv_gru_states[0][:, 32:, :, :]

    # We apply the last decoder, guided by the upsampling mask from the first convGRU hidden state
    out = self.decoder3(out, conv_gru_state_0_mask)

    # We concatenate the prediction from the first convGRU module, and apply the convolution to go
    # from 64 to 32 channels
    out = torch.concat((out, conv_gru_state_0_pred), dim=1)
    out = self.conv_concat3(out)

    # We finish by applying the prediction layer
    out = self.prediction_layer(out)

    # And we return the final prediction
    return out
