#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a dataloader and helper functions which can be used to load the raw (unprocessed)
SLED dataset.
"""

from bisect import bisect_right
import csv
from itertools import accumulate
from os import path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def compute_event_volume(events, bins):
  """
  From a numpy array of events, computes an event volume, as described in the "Learning to Detect
  Objects with a 1 Megapixel Event Camera" article by Perot et al.
  This implementation is optimized for fast computation (which is still a bit slow :c), thanks to
  https://stackoverflow.com/a/55739936
  """

  # We create an empty event volume
  event_volume = np.zeros((2*bins, 720, 1280), np.float32)

  # We compute the t_star value for each event
  t_star = (bins-1)*(events["t"]-events[0]["t"])/(events[-1]["t"]-events[0]["t"])

  # We create an index of unique (x, y, pol) events
  idx, u_evts = pd.factorize(events[["x", "y", "pol"]])

  # Then, for each bin...
  for i in range(bins):
    # We compute the sum of the max(0, 1-abs(bin-t_star)) for each pixel
    sums = np.bincount(idx, np.fmax(0, 1-abs(i-t_star)))

    # We set these values inside the event volume
    event_volume[i+bins*u_evts["pol"], u_evts["y"], u_evts["x"]] = sums

  # We finally return the event volume, in the pytorch format
  return torch.from_numpy(event_volume)


def compute_depth_image(depth_image_raw, lidar_max_range):
  """
  From a raw CARLA depth image, computes a Tensor representation, which can be fed to the network.
  Details on how the conversion works can be found here:
  https://carla.readthedocs.io/en/0.9.13/ref_sensors/#depth-camera
  """

  # We convert the raw depth image to a float32 matrix of depth values in meters
  depth_image = depth_image_raw.astype(np.float32)
  depth_image = ((depth_image[:, :, 2] + depth_image[:, :, 1]*256.0 + depth_image[:, :, 0]*256.0*256.0)/(256.0*256.0*256.0 - 1.))
  depth_image *= 1000

  # We normalize these values based on the max range of the LiDAR
  # Note that the depth image contains values > than 1.0, which should probably be filtered out
  # during training
  depth_image /= lidar_max_range
  
  # Finally, we transform the numpy matrix to a PyTorch Tensor
  depth_image = torch.from_numpy(depth_image)
  depth_image = depth_image.unsqueeze(0)

  # And we return it
  return depth_image


def compute_lidar_projection(lidar_cloud, lidar_max_range, camera_fov, use_intensities):
  """
  Creates a projection of the point cloud in a 1- or 2-channel matrix.
  The first channel corresponds to the depth values, normalized between 0 and 1.
  The second channel is optional, and corresponds to the intensity values.
  """

  # We create a false camera, of resolution 1280x720, aligned with the LiDAR sensor
  # R_c_l is the rotation matrix from LiDAR to camera, to correct the axes
  f = 1280/(2*np.tan(camera_fov*np.pi/360))
  cx = 1280/2
  cy = 720/2
  K = np.array([[f, 0, cx],
                [0, f, cy],
                [0, 0, 1 ]])
  R_c_l = np.array([[0, 1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])

  # We then filter the point cloud, to only retain points in front of the camera
  lidar_cloud_filt = lidar_cloud[lidar_cloud[:, 0] > 0]
  pcl_pts_filt = lidar_cloud_filt[:, :3]
  if use_intensities:
    intensities_filt = lidar_cloud_filt[:, 3]

  # We project them to the camera's frame
  pcl_camera_frame = (R_c_l @ pcl_pts_filt.T).T
  depths = pcl_camera_frame[:, 2].copy()
  pcl_camera_frame[:, 0] /= depths
  pcl_camera_frame[:, 1] /= depths
  pcl_camera_frame[:, 2] /= depths

  # We project them in the image
  pcl_camera = (K @ pcl_camera_frame.T).T

  # We create the projection, and add each projected LiDAR point to it
  # The projection is composed of 1 or 2 channels: depth and, if required, intensity of the point
  if use_intensities:
    lidar_proj = torch.zeros(2, 720, 1280)
  else:
    lidar_proj = torch.zeros(1, 720, 1280)
  for i, pt in enumerate(pcl_camera[:, :2]):
    if pt[0] >= 0 and pt[0] < 1280 and pt[1] >= 0 and pt[1] < 720:
      lidar_proj[0, int(pt[1]), int(pt[0])] = min(depths[i]/lidar_max_range, 1.0)
      if use_intensities:
        lidar_proj[1, int(pt[1]), int(pt[0])] = np.float64(intensities_filt[i])

  # We return the projection
  return lidar_proj


class SLEDRawDataset(Dataset):
  """
  A data loader for the SLED dataset
  """

  def __init__(self, path_dataset, evts_bins, lidar_clouds_per_sequence, lidar_max_range, dvs_fov,
               use_lidar_intensities, transform=None):
    # `path_dataset` should point to a folder containing at least one .npz recording, as well as a
    # metadata.csv file indicating the length of each recording

    # We begin by verifying that the path is correct
    if not path.isdir(path_dataset):
      raise Exception("The path to the dataset should be a folder, containing .npz recordings and "
        "a metadata.csv file")

    # Based on the metadata.csv file, we list all the recordings and their length
    self.recordings_paths = []
    recordings_lengths = []
    with open(path_dataset+"/metadata.csv", newline='') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=';')
      for row in csv_reader:
        # We save the recording path
        self.recordings_paths.append(path_dataset+"/"+row[0])

        # We save the number of sequences which can be generated from this recording
        # Note: for each recording, we do not use the last point cloud, as it has less events
        # associated with it than the other point clouds, which can cause issues
        nb_point_clouds = int(row[1])
        if nb_point_clouds % lidar_clouds_per_sequence:
          recordings_lengths.append(nb_point_clouds // lidar_clouds_per_sequence)
        else:
          recordings_lengths.append(nb_point_clouds // lidar_clouds_per_sequence - 1)

    # We verify that we have read at least one entry
    if not self.recordings_paths:
      raise Exception("The provided metadata.csv file is empty!")

    # We save some data, which will be used by the __getitem__ and the __len__ functions of the
    # dataloader
    self.cumulative_recordings_lengths = list(accumulate(recordings_lengths))
    self.bins = evts_bins
    self.lidar_clouds_per_sequence = lidar_clouds_per_sequence
    self.lidar_max_range = lidar_max_range
    self.dvs_fov = dvs_fov
    self.use_lidar_intensities = use_lidar_intensities
    self.transform = transform


  def __getitem__(self, index):
    """
    An item is a sequence of L successive LiDAR scans *from the same recording*, and of all the
    events associated with them. They are returned with the following form:
    ```python
    [[lidar_proj_j,
      [events_j_0, events_j_1, ...],
      [bf_depths_j_0, bf_depths_j_1, ...],
      [af_depths_j_0, af_depths_j_1, ...]],
     [lidar_proj_j+1,
      [events_j+1_0, events_j+1_1, ...],
      [bf_depths_j_0, bf_depths_j_1, ...],
      [af_depths_j_0, af_depths_j_1, ...]]]
    ```

    Note that sequences all contain distinct point clouds, meaning that if
    `lidar_clouds_per_sequence` is set to 3 for instance, that two recordings were loaded with
    respectively 4 and 3 LiDAR clouds, then:
    - the first sequence will contain LiDAR clouds [0_0, 0_1, 0_2] (i.e., clouds 0, 1, and 2 from
      recording 0)
    - LiDAR cloud 0_3 will be dropped, as it cannot be included in a new sequence of length 3
    - the second and final sequence will contain LiDAR clouds [1_0, 1_1, 1_2]
    """
      # 

    # We create the sequence, as an empty array at first
    sequence = []

    # We have to find in which recording the sequences corresponding to the given index are
    recording_index = bisect_right(self.cumulative_recordings_lengths, index)
    if recording_index == 0:
      seq_in_recording_index = index
    else:
      seq_in_recording_index = index - self.cumulative_recordings_lengths[recording_index-1]

    # We open and read data from the correct file
    recording = np.load(self.recordings_paths[recording_index], allow_pickle=True)
    events_with_ts = recording["events"]
    lidar_clouds_with_ts = recording["lidar_clouds"]
    depth_images_with_ts = recording["depth_images"]

    # We save the RNG state for the transform operations, which should be consistent on the whole
    # sequence
    saved_rng_state = torch.get_rng_state()

    # Then, for each LiDAR point cloud that should be considered...
    for j in range(self.lidar_clouds_per_sequence*seq_in_recording_index, self.lidar_clouds_per_sequence*(seq_in_recording_index+1)):
      # We extract the LiDAR cloud, its timestamp, and project it as an image
      lidar_cloud, start_ts = lidar_clouds_with_ts[j]
      lidar_proj = compute_lidar_projection(lidar_cloud, self.lidar_max_range, self.dvs_fov, self.use_lidar_intensities)

      # Since the LiDAR in CARLA still doesn't see some objects (even though it is supposed to be
      # fixed, see https://github.com/carla-simulator/carla/issues/5732), we replace points with a
      # distance computed from the LiDAR with the distance from the depth map directly
      # If it is fixed one day in a new release, remove this paragraph of code
      depth_image_raw = depth_images_with_ts[depth_images_with_ts[:, 1] >= start_ts][0, 0]
      depth_image = compute_depth_image(depth_image_raw, self.lidar_max_range)
      mask = torch.bitwise_and(lidar_proj[0, :, :] != 0, depth_image[0, :, :] < 1.0)
      lidar_proj[0, :, :][mask] = depth_image[0, :, :][mask]
      lidar_proj[0, :, :][~mask] = 0.

      # We apply the transform on the point cloud if required
      if self.transform:
        torch.set_rng_state(saved_rng_state)
        lidar_proj = self.transform(lidar_proj)

      # To know which events and depth images should be extracted, we set the end timestamp as the
      # timestamp of the next LiDAR scan (if available)
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
        event_volume = compute_event_volume(event_array, self.bins)
        if self.transform:
          torch.set_rng_state(saved_rng_state)
          event_volume = self.transform(event_volume)
        event_volumes.append(event_volume)

      # We do the same with the D_bf depth images
      bf_depth_images_ts_mask = np.bitwise_and(depth_images_with_ts[:, 1] >= start_ts,
                                               depth_images_with_ts[:, 1] <= end_ts)
      bf_depth_images_raw = depth_images_with_ts[bf_depth_images_ts_mask][:, 0]
      bf_depth_images_raw_restricted = [bf_depth_images_raw[0],
                                        bf_depth_images_raw[bf_depth_images_raw.shape[0]//2]]
      bf_depth_images = []
      for bf_depth_image_raw in bf_depth_images_raw_restricted:
        bf_depth_image = compute_depth_image(bf_depth_image_raw, self.lidar_max_range)
        if self.transform:
          torch.set_rng_state(saved_rng_state)
          bf_depth_image = self.transform(bf_depth_image)
        bf_depth_images.append(bf_depth_image)

      # And the D_af depth images
      af_depth_images_ts_mask = np.bitwise_and(depth_images_with_ts[:, 1] >= start_ts,
                                               depth_images_with_ts[:, 1] <= end_ts)
      af_depth_images_raw = depth_images_with_ts[af_depth_images_ts_mask][:, 0]
      af_depth_images_raw_restricted = [af_depth_images_raw[af_depth_images_raw.shape[0]//2],
                                        af_depth_images_raw[-1]]
      af_depth_images = []
      for af_depth_image_raw in af_depth_images_raw_restricted:
        af_depth_image = compute_depth_image(af_depth_image_raw, self.lidar_max_range)
        if self.transform:
          torch.set_rng_state(saved_rng_state)
          af_depth_image = self.transform(af_depth_image)
        af_depth_images.append(af_depth_image)

      # Finally, we add the projected LiDAR cloud, the event volumes, and the depth images to the
      # sequence array
      sequence.append([lidar_proj, event_volumes, bf_depth_images, af_depth_images])

    # Once all the LiDAR clouds composing the sequence were explored, we don't forget to close the
    # recording
    recording.close()

    # And we return the sequence
    return sequence 


  def __len__(self):
    """
    Returns the number of sequences that can be generated from the dataset.
    For a better understanding, see the description of the __getitem__ function above
    """
    return self.cumulative_recordings_lengths[-1]
