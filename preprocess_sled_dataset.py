#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script can be called to preprocess the SLED dataset, transforming the raw .npz recordings into
PyTorch .pt files, with all the preprocessing steps (data formatting, normalization, LiDAR
projection, ...) already applied.
Preprocessing the dataset has its pros and cons:
- it reduces greatly the computational power necessary to load the dataset during training /
  validation / testing;
- it greatly increases disk space usage, as the dataset is converted into multiple small sequences;
- if the preprocessing steps are modified, the whole dataset has to be preprocessed again.
"""

import argparse
import csv
from os import mkdir, path

import numpy as np
import torch
from tqdm.contrib.concurrent import process_map

from raw_dataset_loader_sled import compute_event_volume, compute_depth_image, \
  compute_lidar_projection


# We define args as a global variable, so that it can be seen by all parallel processes
args = None


def parse_args():
  """Args parser"""
  parser = argparse.ArgumentParser()
  parser.add_argument("path_raw", help="Path to the folder containing the raw dataset")
  parser.add_argument("path_processed", help="Path to the folder where the dataset will be stored "
    "after preprocessing")
  parser.add_argument("evts_bins", type=int, help="Number of bins B for the event volume "
    "representation")
  parser.add_argument("lidar_clouds_per_seq", type=int, help="Number of LiDAR clouds contained in "
    "each sequence. By setting it to a value <= 0, the full sequence won't be split")
  parser.add_argument("lidar_max_range", type=float, help="Maximum range (in meters) of the LiDAR "
    "sensor")
  parser.add_argument("dvs_fov", type=float, help="FOV (in degrees) of the event camera")
  parser.add_argument("--use_lidar_intensities", action="store_true", help="If used, the LiDAR "
    "intensities will be used")
  parser.add_argument("-j", type=int, default=0, help="Number of parallel processes spawned to "
    "preprocess the dataset. If not specified, `min(32, cpu_count() + 4)` processes are spawned")
  return parser.parse_args()


def preprocess_recording(recording_path):
  """Recording preprocessing function, which can be called in parallel"""

  # We start by getting the args
  global args

  # We open and read data from the file
  recording = np.load(args.path_raw+"/"+recording_path, allow_pickle=True)
  events_with_ts = recording["events"]
  lidar_clouds_with_ts = recording["lidar_clouds"]
  depth_images_with_ts = recording["depth_images"]

  # If args.lidar_clouds_per_seq is set to a value <= 0, we do not split the sequence
  if args.lidar_clouds_per_seq <= 0:
    args.lidar_clouds_per_seq = len(lidar_clouds_with_ts) - 1

  # Each recording allows us to generate N/L sequences ((N/L)-1 if N%L==0), where N is the total
  # number of point clouds in the recording, and L is the total number of successive point clouds in
  # a sequence
  if len(lidar_clouds_with_ts) % args.lidar_clouds_per_seq:
    nb_seq = len(lidar_clouds_with_ts) // args.lidar_clouds_per_seq
  else:
    nb_seq = len(lidar_clouds_with_ts) // args.lidar_clouds_per_seq - 1

  # We generate each of the nb_seq sequences
  for i in range(nb_seq):
    sequence = []

    # Each sequence contains L successive LiDAR clouds from the recording
    for j in range(i*args.lidar_clouds_per_seq, (i+1)*args.lidar_clouds_per_seq):
      # We extract the LiDAR cloud, its timestamp, and project it as an image
      lidar_cloud, start_ts = lidar_clouds_with_ts[j]
      lidar_proj = compute_lidar_projection(lidar_cloud, args.lidar_max_range, args.dvs_fov, args.use_lidar_intensities)

      # Since the LiDAR in CARLA still doesn't see some objects (even though it is supposed to be
      # fixed, see https://github.com/carla-simulator/carla/issues/5732), we replace points with a
      # distance computed from the LiDAR with the distance from the depth map directly.
      # If it is fixed one day in a new release, remove this paragraph of code
      depth_image_raw = depth_images_with_ts[depth_images_with_ts[:, 1] >= start_ts][0, 0]
      depth_image = compute_depth_image(depth_image_raw, args.lidar_max_range)
      mask = torch.bitwise_and(lidar_proj[0, :, :] != 0, depth_image[0, :, :] < 1.0)
      lidar_proj[0, :, :][mask] = depth_image[0, :, :][mask]
      lidar_proj[0, :, :][~mask] = 0.

      # To know which events and depth images should be extracted, we set the end timestamp as the
      # timestamp of the next LiDAR scan
      end_ts = lidar_clouds_with_ts[j+1, 1]

      # We extract the event arrays based on this timestamp range
      events_ts_mask = np.bitwise_and(events_with_ts[:, 1] > start_ts,
                                      events_with_ts[:, 1] <= end_ts)
      events = events_with_ts[events_ts_mask][:, 0]

      # We concatenate them to have 2 event arrays per LiDAR cloud (so, 50ms of events for a 10Hz
      # LiDAR, for instance)
      events_concat = [np.concatenate(events[:events.shape[0]//2], axis=None),
                       np.concatenate(events[events.shape[0]//2:], axis=None)]

      # And for each of them, we compute the corresponding event volume
      event_volumes = []
      for event_array in events_concat:
        event_volume = compute_event_volume(event_array, args.evts_bins)
        event_volumes.append(event_volume)

      # We do the same with the D_bf depth images
      bf_depth_images_ts_mask = np.bitwise_and(depth_images_with_ts[:, 1] >= start_ts,
                                               depth_images_with_ts[:, 1] <= end_ts)
      bf_depth_images_raw = depth_images_with_ts[bf_depth_images_ts_mask][:, 0]
      bf_depth_images_raw_restricted = [bf_depth_images_raw[0],
                                        bf_depth_images_raw[bf_depth_images_raw.shape[0]//2]]
      bf_depth_images = []
      for bf_depth_image_raw in bf_depth_images_raw_restricted:
        bf_depth_image = compute_depth_image(bf_depth_image_raw, args.lidar_max_range)
        bf_depth_images.append(bf_depth_image)

      # And the D_af depth images
      af_depth_images_ts_mask = np.bitwise_and(depth_images_with_ts[:, 1] >= start_ts,
                                               depth_images_with_ts[:, 1] <= end_ts)
      af_depth_images_raw = depth_images_with_ts[af_depth_images_ts_mask][:, 0]
      af_depth_images_raw_restricted = [af_depth_images_raw[af_depth_images_raw.shape[0]//2],
                                        af_depth_images_raw[-1]]
      af_depth_images = []
      for af_depth_image_raw in af_depth_images_raw_restricted:
        af_depth_image = compute_depth_image(af_depth_image_raw, args.lidar_max_range)
        af_depth_images.append(af_depth_image)

      # Finally, we add the projected LiDAR cloud, the event volumes, and the D_bf and D_af depth
      # images to the sequence array
      sequence.append([lidar_proj, event_volumes, bf_depth_images, af_depth_images])

    # Once all the LiDAR point clouds, events and depth images have been added to the sequence, we
    # save it
    torch.save(sequence, f"{args.path_processed}/{recording_path[:-4]}_seq{i:02}.pt")

  # Once the recording has been fully explored, we don't forget to close it
  recording.close()


def main():
  """Main function"""

  # We start by reading the args given by the user
  global args
  args = parse_args()

  # We begin by verifying that the paths given by the user are valid
  if not path.isdir(args.path_raw):
    raise Exception("The path to the dataset should be a folder, containing .npz recordings and "
      "a metadata.csv file")
  if not path.isdir(args.path_processed):
    mkdir(args.path_processed)

  # Based on the metadata.csv file, we list all the recordings in the folder
  recordings_paths = []
  with open(args.path_raw+"/metadata.csv", newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
      recordings_paths.append(row[0])
  if not recordings_paths:
    raise Exception("The provided metadata.csv file is empty!")

  # Then, we process all the recordings in parallel, using the "process_map" function from tqdm
  if args.j > 0:
    process_map(preprocess_recording, recordings_paths, max_workers=args.j)
  else:
    process_map(preprocess_recording, recordings_paths)


if __name__ == "__main__":
  main()
