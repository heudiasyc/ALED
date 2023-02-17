#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file can be used to test the performances of our ALED network for dense and sparse depth
inference, as used in our "Learning to Estimate Two Dense Depths from LiDAR and Event Data" article.
"""

import argparse
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Pad
from torchvision.utils import save_image
from tqdm import tqdm

from aled import ALED
from preprocessed_dataset_loader import PreprocessedDataset
from raw_dataset_loader_sled import SLEDRawDataset
from visualization import depth_difference_to_img_color, depth_image_to_img_color, \
                          event_volume_to_img, lidar_proj_to_img_color_gray, \
                          predicted_depths_to_img_color


def parse_args():
  """Args parser"""
  parser = argparse.ArgumentParser()
  parser.add_argument("config_file", help="Path to the JSON config file to use for testing")
  parser.add_argument("checkpoint", help="Path to the .pth checkpoint file to use for testing")
  return parser.parse_args()


def main():
  """Main function"""

  with torch.no_grad():
    # Before doing anything, we must change the torch multiprocessing sharing strategy, to avoid
    # having issues with leaking file descriptors.
    # For more informations, see https://github.com/pytorch/pytorch/issues/973
    torch.multiprocessing.set_sharing_strategy("file_system")

    # We start by loading the config file given by the user
    args = parse_args()
    config = json.load(open(args.config_file))

    # We configure the device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We setup the transforms we will perform on the test dataset, i.e. padding (if required)
    if config["transforms"]["pad_test"]:
      pad_size_x = config["transforms"]["pad_size_x"]
      pad_size_y = config["transforms"]["pad_size_y"]
      test_transforms = Pad((pad_size_x, pad_size_y, pad_size_x, pad_size_y))
    else:
      pad_size_x = 0
      pad_size_y = 0
      test_transforms = None

    # We collect the batch_size and num_workers parameters from the config file
    batch_size = config["batch_size_test"]
    num_workers = config["num_workers"]

    # We collect the lidar_max_range parameter from the config file, and compute the cutoff values
    # based on it
    lidar_max_range = config["lidar_max_range"]
    cutoff_dists = (10.0/lidar_max_range, 20.0/lidar_max_range, 30.0/lidar_max_range, 0.5, 1.0)

    # We create/open our txt files in which we will store the results
    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file_seq = open(f"results/{time_prefix}_per_seq.txt", "w")
    txt_file_global = open(f"results/{time_prefix}_global.txt", "w")

    # We load the dataset and create the dataloader
    dataset_path = config["datasets"]["path_test"]
    if config["datasets"]["is_preprocessed_test"]:
      dataset = PreprocessedDataset(dataset_path, False, test_transforms)
    else:
      dataset = SLEDRawDataset(dataset_path, 5, 0, lidar_max_range, 90.0, False, test_transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # We initialize the network
    net = ALED(10, 1)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()
    net.to(device)

    # We initialize the metrics
    # We use an array of 5 elements, as we use 5 cutoff distances: 10m, 20m, 30m, half, no cutoff
    full_bf_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_af_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_bf_loss_rel = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_af_loss_rel = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_1_depth_per_event_bf_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_1_depth_per_event_af_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_depth_difference_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_depth_difference_thr_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
    total_pixels_bf = [0, 0, 0, 0, 0]
    total_pixels_af = [0, 0, 0, 0, 0]
    total_pxs_1_depth_per_event_bf = [0, 0, 0, 0, 0]
    total_pxs_1_depth_per_event_af = [0, 0, 0, 0, 0]
    total_pxs_depth_difference = [0, 0, 0, 0, 0]

    # For each sequence extracted from the dataset...
    for seq_nbr, sequence in enumerate(tqdm(dataloader, "Testing")):
      # We compute some infos about the length of the sequence
      total_items = len(sequence)
      total_elems_per_item = len(sequence[0][1])
      total_elems_in_seq = total_items * total_elems_per_item

      # We initialize the metrics on this sequence
      # As before, we use an array of 5 elements as we use 5 cutoff distances
      seq_bf_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_af_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_bf_loss_rel = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_af_loss_rel = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_1_depth_per_event_bf_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_1_depth_per_event_af_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_depth_difference_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_depth_difference_thr_loss = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_total_pxs_bf = [0, 0, 0, 0, 0]
      seq_total_pxs_af = [0, 0, 0, 0, 0]
      seq_total_pxs_1_depth_per_event_bf = [0, 0, 0, 0, 0]
      seq_total_pxs_1_depth_per_event_af = [0, 0, 0, 0, 0]
      seq_total_pxs_depth_difference = [0, 0, 0, 0, 0]

      # We reset the state of the convGRUs
      conv_gru_states = [None, None, None, None]

      # For each item (1 LiDAR proj, events, depth images) in the sequence...
      for i, item in enumerate(tqdm(sequence, "Sequence", leave=False)):
        # We extract the LiDAR projection and feed it to the network
        lidar_proj = item[0].to(device)
        conv_gru_states = net(lidar_proj, conv_gru_states, "lidar")

        # We compute the limits for padding removal
        min_x = pad_size_x
        max_x = lidar_proj.shape[3] - pad_size_x
        min_y = pad_size_y
        max_y = lidar_proj.shape[2] - pad_size_y

        # If required, we save a preview of the projected LiDAR data
        if config["save_visualization"]:
          lidax_idx = seq_nbr*total_elems_in_seq + i*total_elems_per_item
          unpadded_lidar_proj = lidar_proj[:, :, min_y:max_y, min_x:max_x]
          lidar_proj_disp = lidar_proj_to_img_color_gray(unpadded_lidar_proj[:, 0, :, :])
          save_image(lidar_proj_disp, f"images/lidar{lidax_idx:06d}.png")

        # We compute the y coordinates of the uppermost and lowermost LiDAR scans (this will be
        # useful later on)
        y_values_lidar = (lidar_proj[0, 0, :, :] != 0.0).nonzero(as_tuple=True)[0]
        min_y_lidar = torch.min(y_values_lidar)
        max_y_lidar = torch.max(y_values_lidar)

        # And, for each event volume / each depth image
        for j, (event_volume, bf_depth_image, af_depth_image) in enumerate(zip(item[1], item[2], item[3])):
          # We upload them to the device
          event_volume = event_volume.to(device)
          bf_depth_image = bf_depth_image.to(device)
          af_depth_image = af_depth_image.to(device)

          # We make sure that the depth images are in the range [0, 1]
          bf_depth_image[bf_depth_image > 1.0] = 1.0
          af_depth_image[af_depth_image > 1.0] = 1.0

          # We feed the event volume to the network
          conv_gru_states = net(event_volume, conv_gru_states, "events")

          # We run a prediction
          pred = net(None, conv_gru_states, "predict")

          # We correct the prediction, to force it to be in the [0, 1] range
          pred[pred < 0.0] = 0.0
          pred[pred > 1.0] = 1.0

          # We remove any padding before using the data further on
          unpadded_bf_depth_image = bf_depth_image[:, :, min_y:max_y, min_x:max_x]
          unpadded_af_depth_image = af_depth_image[:, :, min_y:max_y, min_x:max_x]
          unpadded_event_volume = event_volume[:, :, min_y:max_y, min_x:max_x]
          unpadded_pred = pred[:, :, min_y:max_y, min_x:max_x]

          # We compute a mask from the events
          events_mask = (torch.sum(unpadded_event_volume, dim=1) != 0)

          # We compute a second mask from the events, where we only take the events between the
          # uppermost and lowermost LiDAR scans
          events_mask_restr = events_mask.clone()
          events_mask_restr[:, :min_y_lidar, :] = False
          events_mask_restr[:, max_y_lidar+1:, :] = False

          # We compute the D_af-D_bf difference
          gt_diff = unpadded_af_depth_image[:, [0], :, :] - unpadded_bf_depth_image[:, [0], :, :]
          pred_diff = unpadded_pred[:, [1], :, :] - unpadded_pred[:, [0], :, :]

          # We compute the thresholded D_af-D_bf difference, with a threshold of 1 meter
          threshold_diff = 1.0/lidar_max_range
          gt_diff_thr = torch.zeros_like(gt_diff)
          gt_diff_thr[gt_diff < -threshold_diff] = -1
          gt_diff_thr[gt_diff > threshold_diff] = 1
          pred_diff_thr = torch.zeros_like(pred_diff)
          pred_diff_thr[pred_diff < -threshold_diff] = -1
          pred_diff_thr[pred_diff > threshold_diff] = 1

          # If required, we save previews of the inputs and results
          if config["save_visualization"]:
            event_volume_disp = event_volume_to_img(unpadded_event_volume)
            save_image(event_volume_disp, f"images/evts{lidax_idx+j:06d}.png")

            gt_bf_disp = depth_image_to_img_color(unpadded_bf_depth_image[:, 0, :, :])
            save_image(gt_bf_disp, f"images/gtbf{lidax_idx+j:06d}.png")

            gt_af_disp = depth_image_to_img_color(unpadded_af_depth_image[:, 0, :, :])
            save_image(gt_af_disp, f"images/gtaf{lidax_idx+j:06d}.png")

            pred_bf_disp = predicted_depths_to_img_color(unpadded_pred[:, 0, :, :])
            save_image(pred_bf_disp, f"images/predbf{lidax_idx+j:06d}.png")

            pred_af_disp = predicted_depths_to_img_color(unpadded_pred[:, 1, :, :])
            save_image(pred_af_disp, f"images/predaf{lidax_idx+j:06d}.png")

            gt_bf_masked_disp = depth_image_to_img_color(unpadded_bf_depth_image[:, 0, :, :], events_mask_restr)
            save_image(gt_bf_masked_disp, f"images/gtbfmasked{lidax_idx+j:06d}.png")

            pred_bf_masked_disp = predicted_depths_to_img_color(unpadded_pred[:, 0, :, :], events_mask_restr)
            save_image(pred_bf_masked_disp, f"images/predbfmasked{lidax_idx+j:06d}.png")

            gt_diff_disp = depth_difference_to_img_color(gt_diff[:, 0, :, :], events_mask, threshold_diff)
            save_image(gt_diff_disp, f"images/gtdiff{lidax_idx+j:06d}.png")

            pred_diff_disp = depth_difference_to_img_color(pred_diff[:, 0, :, :], events_mask, threshold_diff)
            save_image(pred_diff_disp, f"images/preddiff{lidax_idx+j:06d}.png")

          # We compute the error for each cutoff distance (without forgetting to ignore NaN values)
          val_criterion = nn.L1Loss(reduction="sum")

          # Removing NaN values for D_bf
          not_nan_mask_bf = (~torch.isnan(unpadded_bf_depth_image))
          no_nan_unpadded_pred_bf = unpadded_pred[:, [0], :, :][not_nan_mask_bf]
          no_nan_unpadded_depth_img_bf = unpadded_bf_depth_image[not_nan_mask_bf]

          # Removing NaN values for D_af
          not_nan_mask_af = (~torch.isnan(unpadded_af_depth_image))
          no_nan_unpadded_pred_af = unpadded_pred[:, [1], :, :][not_nan_mask_af]
          no_nan_unpadded_depth_img_af = unpadded_af_depth_image[not_nan_mask_af]

          # Mask to use to remove NaN values and use evts between the uppermost and lowermost LiDAR
          # scans as a mask for D_bf and D_af
          not_nan_and_events_restr_mask_bf = torch.bitwise_and(events_mask_restr, not_nan_mask_bf)
          not_nan_and_events_restr_mask_af = torch.bitwise_and(events_mask_restr, not_nan_mask_af)

          # Mask to use to remove NaN values and using evts as a mask for D_af-D_bf and
          # thresholded D_af-D_bf
          not_nan_and_events_mask_bf_af = torch.bitwise_and(torch.bitwise_and(events_mask, not_nan_mask_bf), not_nan_mask_af)

          for c, cutoff in enumerate(cutoff_dists):
            # Error on dense D_bf (with cutoff)
            cutoff_mask_bf = (no_nan_unpadded_depth_img_bf <= cutoff)
            cutoff_pred_bf = no_nan_unpadded_pred_bf[cutoff_mask_bf]
            cutoff_depth_img_bf = no_nan_unpadded_depth_img_bf[cutoff_mask_bf]
            seq_bf_loss[c] += val_criterion(cutoff_pred_bf, cutoff_depth_img_bf).item()

            # Relative error on dense D_bf (with cutoff)
            seq_bf_loss_rel[c] += torch.sum(torch.abs(cutoff_depth_img_bf-cutoff_pred_bf)/cutoff_depth_img_bf).item()

            # Error on dense D_af (with cutoff)
            cutoff_mask_af = (no_nan_unpadded_depth_img_af <= cutoff)
            cutoff_pred_af = no_nan_unpadded_pred_af[cutoff_mask_af]
            cutoff_depth_img_af = no_nan_unpadded_depth_img_af[cutoff_mask_af]
            seq_af_loss[c] += val_criterion(cutoff_pred_af, cutoff_depth_img_af).item()

            # Relative error on dense D_af (with cutoff)
            seq_af_loss_rel[c] += torch.sum(torch.abs(cutoff_depth_img_af-cutoff_pred_af)/cutoff_depth_img_af).item()

            # Error on sparse D_bf (with cutoff and with the restriction to be between the LiDAR
            # scans)
            cutoff_not_nan_and_events_restr_mask_bf = torch.bitwise_and((unpadded_bf_depth_image <= cutoff), not_nan_and_events_restr_mask_bf)
            cutoff_no_nan_evts_restr_unpadded_pred_bf = unpadded_pred[:, [0], :, :][cutoff_not_nan_and_events_restr_mask_bf]
            cutoff_no_nan_evts_restr_unpadded_depth_img_bf = unpadded_bf_depth_image[cutoff_not_nan_and_events_restr_mask_bf]
            seq_1_depth_per_event_bf_loss[c] += val_criterion(cutoff_no_nan_evts_restr_unpadded_pred_bf, cutoff_no_nan_evts_restr_unpadded_depth_img_bf).item()

            # Error on sparse D_af (with cutoff and with the restriction to be between the LiDAR
            # scans)
            cutoff_not_nan_and_events_restr_mask_af = torch.bitwise_and((unpadded_af_depth_image <= cutoff), not_nan_and_events_restr_mask_af)
            cutoff_no_nan_evts_restr_unpadded_pred_af = unpadded_pred[:, [1], :, :][cutoff_not_nan_and_events_restr_mask_af]
            cutoff_no_nan_evts_restr_unpadded_depth_img_af = unpadded_af_depth_image[cutoff_not_nan_and_events_restr_mask_af]
            seq_1_depth_per_event_af_loss[c] += val_criterion(cutoff_no_nan_evts_restr_unpadded_pred_af, cutoff_no_nan_evts_restr_unpadded_depth_img_af).item()

            # Error on sparse D_af-D_bf (with cutoff)
            cutoff_not_nan_and_events_mask_bf_af = torch.bitwise_and((unpadded_bf_depth_image <= cutoff), not_nan_and_events_mask_bf_af)
            cutoff_no_nan_evts_unpadded_pred_diff = pred_diff[cutoff_not_nan_and_events_mask_bf_af]
            cutoff_no_nan_evts_unpadded_gt_diff = gt_diff[cutoff_not_nan_and_events_mask_bf_af]
            seq_depth_difference_loss[c] += val_criterion(cutoff_no_nan_evts_unpadded_pred_diff, cutoff_no_nan_evts_unpadded_gt_diff).item()

            # Error on thresholded sparse D_af-D_bf (with cutoff)
            cutoff_no_nan_evts_unpadded_pred_diff_thr = pred_diff_thr[cutoff_not_nan_and_events_mask_bf_af]
            cutoff_no_nan_evts_unpadded_gt_diff_thr = gt_diff_thr[cutoff_not_nan_and_events_mask_bf_af]
            seq_depth_difference_thr_loss[c] += torch.sum(cutoff_no_nan_evts_unpadded_pred_diff_thr != cutoff_no_nan_evts_unpadded_gt_diff_thr).item()

            seq_total_pxs_bf[c] += torch.sum(cutoff_mask_bf).item()
            seq_total_pxs_af[c] += torch.sum(cutoff_mask_af).item()
            seq_total_pxs_1_depth_per_event_bf[c] += torch.sum(cutoff_not_nan_and_events_restr_mask_bf).item()
            seq_total_pxs_1_depth_per_event_af[c] += torch.sum(cutoff_not_nan_and_events_restr_mask_af).item()
            seq_total_pxs_depth_difference[c] += torch.sum(cutoff_not_nan_and_events_mask_bf_af).item()

      # Once the sequence is over, we display and save the error for each cutoff distance
      tqdm.write(f"{dataset.sequences_paths[seq_nbr].split('/')[-1]}")
      seq_name_latex = dataset.sequences_paths[seq_nbr].split('/')[-1][:-9].replace('_', '\_')
      txt_file_seq.write(f"\multirow{{5}}{{*}}{{\\added{{{seq_name_latex}}}}} ")
      for c, cutoff in enumerate(cutoff_dists):
        tqdm.write(f"Error bf {cutoff*lidar_max_range}m: {seq_bf_loss[c]/seq_total_pxs_bf[c]*lidar_max_range:.2f}; " +
                   f"Error bf rel {cutoff*lidar_max_range}m: {seq_bf_loss_rel[c]/seq_total_pxs_bf[c]*100:.2f}; " +
                   f"Error af {cutoff*lidar_max_range}m: {seq_af_loss[c]/seq_total_pxs_af[c]*lidar_max_range:.2f}; " +
                   f"Error af rel {cutoff*lidar_max_range}m: {seq_af_loss_rel[c]/seq_total_pxs_af[c]*100:.2f}; " +
                   f"Error bf 1 depth / evt {cutoff*lidar_max_range}m: {seq_1_depth_per_event_bf_loss[c]/seq_total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}; " +
                   f"Error af 1 depth / evt {cutoff*lidar_max_range}m: {seq_1_depth_per_event_af_loss[c]/seq_total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}; " +
                   f"Error depth diff {cutoff*lidar_max_range}m: {seq_depth_difference_loss[c]/seq_total_pxs_depth_difference[c]*lidar_max_range:.2f}; " +
                   f"Correctly classified {cutoff*lidar_max_range}m: {100-seq_depth_difference_thr_loss[c]/seq_total_pxs_depth_difference[c]*100:.2f}")
        txt_file_seq.write(f"& {int(cutoff*lidar_max_range)}m " +
                           f"& {seq_bf_loss[c]/seq_total_pxs_bf[c]*lidar_max_range:.2f}m " +
                           f"& {seq_bf_loss_rel[c]/seq_total_pxs_bf[c]*100:.2f}\% " +
                           f"& {seq_af_loss[c]/seq_total_pxs_af[c]*lidar_max_range:.2f}m " +
                           f"& {seq_af_loss_rel[c]/seq_total_pxs_af[c]*100:.2f}\% " +
                           f"& m "
                           f"& {seq_1_depth_per_event_bf_loss[c]/seq_total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}m " +
                           f"& m "
                           f"& {seq_1_depth_per_event_af_loss[c]/seq_total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}m " +
                           f"& {seq_depth_difference_loss[c]/seq_total_pxs_depth_difference[c]*lidar_max_range:.2f}m " +
                           f"& {100-seq_depth_difference_thr_loss[c]/seq_total_pxs_depth_difference[c]*100:.2f}\% \\\\\n")

        # And we add it to the global error
        full_bf_loss[c] += seq_bf_loss[c]
        full_af_loss[c] += seq_af_loss[c]
        full_bf_loss_rel[c] += seq_bf_loss_rel[c]
        full_af_loss_rel[c] += seq_af_loss_rel[c]
        full_1_depth_per_event_bf_loss[c] += seq_1_depth_per_event_bf_loss[c]
        full_1_depth_per_event_af_loss[c] += seq_1_depth_per_event_af_loss[c]
        full_depth_difference_loss[c] += seq_depth_difference_loss[c]
        full_depth_difference_thr_loss[c] += seq_depth_difference_thr_loss[c]
        total_pixels_bf[c] += seq_total_pxs_bf[c]
        total_pixels_af[c] += seq_total_pxs_af[c]
        total_pxs_1_depth_per_event_bf[c] += seq_total_pxs_1_depth_per_event_bf[c]
        total_pxs_1_depth_per_event_af[c] += seq_total_pxs_1_depth_per_event_af[c]
        total_pxs_depth_difference[c] += seq_total_pxs_depth_difference[c]

      txt_file_seq.write(f"\midrule\n")

    # Once we have gone over all the sequences, we display and save the final error for each cutoff
    # distance
    tqdm.write("FINAL ERROR")
    for c, cutoff in enumerate(cutoff_dists):
      tqdm.write(f"Error bf {cutoff*lidar_max_range}m: {full_bf_loss[c]/total_pixels_bf[c]*lidar_max_range:.2f}; " +
                 f"Error bf rel {cutoff*lidar_max_range}m: {full_bf_loss_rel[c]/total_pixels_bf[c]*100:.2f}; " +
                 f"Error af {cutoff*lidar_max_range}m: {full_af_loss[c]/total_pixels_af[c]*lidar_max_range:.2f}; " +
                 f"Error af rel {cutoff*lidar_max_range}m: {full_af_loss_rel[c]/total_pixels_af[c]*100:.2f}; " +
                 f"Error bf 1 depth / evt {cutoff*lidar_max_range}m: {full_1_depth_per_event_bf_loss[c]/total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}; " +
                 f"Error af 1 depth / evt {cutoff*lidar_max_range}m: {full_1_depth_per_event_af_loss[c]/total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}; " +
                 f"Error depth diff {cutoff*lidar_max_range}m: {full_depth_difference_loss[c]/total_pxs_depth_difference[c]*lidar_max_range:.2f}; " +
                 f"Correctly classified {cutoff*lidar_max_range}m: {100-full_depth_difference_thr_loss[c]/total_pxs_depth_difference[c]*100:.2f}")
      txt_file_global.write(f"FINAL ERROR {cutoff*lidar_max_range}m: " +
                            f"Error bf: {full_bf_loss[c]/total_pixels_bf[c]*lidar_max_range:.2f}; " +
                            f"Error bf rel: {full_bf_loss_rel[c]/total_pixels_bf[c]*100:.2f}; " +
                            f"Error af: {full_af_loss[c]/total_pixels_af[c]*lidar_max_range:.2f}; " +
                            f"Error af rel: {full_af_loss_rel[c]/total_pixels_af[c]*100:.2f}; " +
                            f"Error bf 1 depth / evt: {full_1_depth_per_event_bf_loss[c]/total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}; " +
                            f"Error af 1 depth / evt: {full_1_depth_per_event_af_loss[c]/total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}; " +
                            f"Error depth diff: {full_depth_difference_loss[c]/total_pxs_depth_difference[c]*lidar_max_range:.2f}; " +
                            f"Correctly classified: {100-full_depth_difference_thr_loss[c]/total_pxs_depth_difference[c]*100:.2f}\n")

    # And we don't forget to close the .txt files!
    txt_file_seq.close()
    txt_file_global.close()


if __name__ == "__main__":
  main()
