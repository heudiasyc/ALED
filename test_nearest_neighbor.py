#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file can be used to test the performances of the naive Nearest Neighbor approach for depth
inference, as used in our "Learning to Estimate Two Dense Depths from LiDAR and Event Data" article.
"""

import argparse
from datetime import datetime
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from preprocessed_dataset_loader import PreprocessedDataset
from raw_dataset_loader_sled import SLEDRawDataset
from visualization import predicted_depths_to_img_color


def parse_args():
  """Args parser"""
  parser = argparse.ArgumentParser()
  parser.add_argument("config_file", help="Path to the JSON config file to use for testing")
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

    # We collect the num_workers parameter from the config file
    num_workers = config["num_workers"]

    # We collect the lidar_max_range parameter from the config file, and compute the cutoff values
    # based on it
    lidar_max_range = config["lidar_max_range"]
    cutoff_dists = (10.0/lidar_max_range, 20.0/lidar_max_range, 30.0/lidar_max_range, 0.5, 1.0)

    # We create/open our txt files in which we will store the results
    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_files = []
    for cutoff_dist in cutoff_dists:
      txt_files.append(open(f"results/{time_prefix}_{int(cutoff_dist*lidar_max_range)}m_nn.txt", "w"))

    # We load the dataset and create the dataloader
    dataset_path = config["datasets"]["path_test"]
    if config["datasets"]["is_preprocessed_test"]:
      dataset = PreprocessedDataset(dataset_path, False)
    else:
      dataset = SLEDRawDataset(dataset_path, 5, 0, lidar_max_range, 90.0, False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # We initialize the metrics
    # We use an array of 5 elements, as we use 5 cutoff distances: 10m, 20m, 30m, half, no cutoff
    full_1_depth_per_event_loss_bf = [0.0, 0.0, 0.0, 0.0, 0.0]
    full_1_depth_per_event_loss_af = [0.0, 0.0, 0.0, 0.0, 0.0]
    total_pxs_1_depth_per_event_bf = [0, 0, 0, 0, 0]
    total_pxs_1_depth_per_event_af = [0, 0, 0, 0, 0]

    # For each sequence extracted from the dataset...
    for seq_nbr, sequence in enumerate(tqdm(dataloader, "Testing")):
      # We compute some infos about the length of the sequence
      total_items = len(sequence)
      total_elems_per_item = len(sequence[0][1])
      total_elems_in_seq = total_items * total_elems_per_item

      # We initialize the metrics on this sequence
      # As before, we use an array of 5 elements as we use 5 cutoff distances
      seq_1_depth_per_event_loss_bf = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_1_depth_per_event_loss_af = [0.0, 0.0, 0.0, 0.0, 0.0]
      seq_total_pxs_1_depth_per_event_bf = [0, 0, 0, 0, 0]
      seq_total_pxs_1_depth_per_event_af = [0, 0, 0, 0, 0]

      # For each item (1 LiDAR proj, events, depth images) in the sequence...
      for i, item in enumerate(tqdm(sequence, "Sequence", leave=False)):
        # We extract the LiDAR projection
        lidar_proj = item[0]

        # We compute the LiDAR cloud index
        lidax_idx = seq_nbr*total_elems_in_seq + i*total_elems_per_item

        # We extract all the LiDAR points as a list
        # Note: this part could probably be optimized/revised by using mask operations instead of
        # for loops
        lidar_pts = []
        depths = []
        min_y = None
        max_y = None
        for y in range(lidar_proj.shape[2]):
          for x in range(lidar_proj.shape[3]):
            depth = lidar_proj[0, 0, y, x].item()
            if depth != 0.0:
              lidar_pts.append([x, y])
              depths.append(depth)
              if min_y is None or y < min_y:
                min_y = y
              if max_y is None or y > max_y:
                max_y = y

        # We construct the Delaunay triangles / Voronoï diagram using the LiDAR points
        subdiv = cv2.Subdiv2D([0, 0, lidar_proj.shape[3], lidar_proj.shape[2]])
        subdiv.insert(lidar_pts)

        # And, for each event volume / each depth image
        for j, (event_volume, bf_depth_image, af_depth_image) in enumerate(zip(item[1], item[2], item[3])):
          # We make sure that the depth images are in the range [0, 1]
          bf_depth_image[bf_depth_image > 1.0] = 1.0
          af_depth_image[af_depth_image > 1.0] = 1.0

          # We create empty images, which will contain the depths associated with the events using
          # the naive nearest neighbour method
          events_depths_nn_img = torch.full_like(lidar_proj, float("nan"))

          # For each pixel with at least an event in the event volume...
          binary_flat_event_volume = (torch.sum(event_volume, dim=1) != 0)
          for y in range(event_volume.shape[2]):
            if y < min_y or y > max_y:
              continue

            for x in range(event_volume.shape[3]):
              if not binary_flat_event_volume[0, y, x]:
                continue

              if lidar_proj[0, 0, y, x] != 0:
                # If the event is on a LiDAR point: we use its depth directly
                closest_pt_depth = lidar_proj[0, 0, y, x].item()
              else:
                # Otherwise, we find its nearest LiDAR point using the Voronoi diagram, and get its
                # depth
                _, closest_pt = subdiv.findNearest((x, y))
                closest_pt = np.array(closest_pt)
                closest_pt_depth = lidar_proj[0, 0, int(closest_pt[1]), int(closest_pt[0])].item()

              # We save the results for the nearest neighbour
              events_depths_nn_img[0, 0, y, x] = closest_pt_depth

          # If required, we save a preview of the estimated depths
          if config["save_visualization"]:
            pred_nn_disp = predicted_depths_to_img_color(events_depths_nn_img[:, 0, :, :])
            save_image(pred_nn_disp, f"images/nn{lidax_idx+j:06d}.png")

          # We compute the error for each cutoff distance
          val_criterion = nn.L1Loss(reduction="sum")

          for c, cutoff in enumerate(cutoff_dists):
            cutoff_evts_mask_bf = torch.bitwise_and((~torch.isnan(events_depths_nn_img)), (bf_depth_image <= cutoff))
            cutoff_evts_mask_af = torch.bitwise_and((~torch.isnan(events_depths_nn_img)), (af_depth_image <= cutoff))
            cutoff_masked_pred_bf = events_depths_nn_img[cutoff_evts_mask_bf]
            cutoff_masked_pred_af = events_depths_nn_img[cutoff_evts_mask_af]
            cutoff_masked_depth_img_bf = bf_depth_image[cutoff_evts_mask_bf]
            cutoff_masked_depth_img_af = af_depth_image[cutoff_evts_mask_af]
            seq_1_depth_per_event_loss_bf[c] += val_criterion(cutoff_masked_pred_bf, cutoff_masked_depth_img_bf).item()
            seq_1_depth_per_event_loss_af[c] += val_criterion(cutoff_masked_pred_af, cutoff_masked_depth_img_af).item()
            seq_total_pxs_1_depth_per_event_bf[c] += torch.sum(cutoff_evts_mask_bf).item()
            seq_total_pxs_1_depth_per_event_af[c] += torch.sum(cutoff_evts_mask_af).item()

      # Once the sequence is over, we display and save the error for each cutoff distance
      tqdm.write(f"{dataset.sequences_paths[seq_nbr].split('/')[-1]}")
      for c, (cutoff, txt_file) in enumerate(zip(cutoff_dists, txt_files)):
        tqdm.write(f"Error nn {cutoff*lidar_max_range}m: " +
                   f"{seq_1_depth_per_event_loss_bf[c]/seq_total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}; " +
                   f"{seq_1_depth_per_event_loss_af[c]/seq_total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}")
        txt_file.write(f"{dataset.sequences_paths[seq_nbr].split('/')[-1]}; " +
                       f"Error nn bf: {seq_1_depth_per_event_loss_bf[c]/seq_total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}; " +
                       f"Error nn af: {seq_1_depth_per_event_loss_af[c]/seq_total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}; " +
                       f"Full values bf (loss + nb pixels): {seq_1_depth_per_event_loss_bf[c]} + {seq_total_pxs_1_depth_per_event_bf[c]}; " +
                       f"Full values af (loss + nb pixels): {seq_1_depth_per_event_loss_af[c]} + {seq_total_pxs_1_depth_per_event_af[c]}\n")

        # And we add it to the global error
        full_1_depth_per_event_loss_bf[c] += seq_1_depth_per_event_loss_bf[c]
        full_1_depth_per_event_loss_af[c] += seq_1_depth_per_event_loss_af[c]
        total_pxs_1_depth_per_event_bf[c] += seq_total_pxs_1_depth_per_event_bf[c]
        total_pxs_1_depth_per_event_af[c] += seq_total_pxs_1_depth_per_event_af[c]

    # Once we have gone over all the sequences, we display and save the final error for each cutoff
    # distance
    tqdm.write("FINAL ERROR")
    for c, (cutoff, txt_file) in enumerate(zip(cutoff_dists, txt_files)):
      tqdm.write(f"Error nn bf {cutoff*lidar_max_range}m: {full_1_depth_per_event_loss_bf[c]/total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}")
      tqdm.write(f"Error nn af {cutoff*lidar_max_range}m: {full_1_depth_per_event_loss_af[c]/total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}")
      txt_file.write("FINAL ERROR; " +
                     f"Error nn bf: {full_1_depth_per_event_loss_bf[c]/total_pxs_1_depth_per_event_bf[c]*lidar_max_range:.2f}; " +
                     f"Error nn af: {full_1_depth_per_event_loss_af[c]/total_pxs_1_depth_per_event_af[c]*lidar_max_range:.2f}\n")

    # And we don't forget to close the .txt files!
    for txt_file in txt_files:
      txt_file.close()


if __name__ == "__main__":
  main()
