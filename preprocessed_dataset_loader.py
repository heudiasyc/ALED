#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a dataloader which can be used to load the preprocessed SLED and MVSEC datasets.
"""

from glob import glob
from os import path

import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
  """
  A data loader for both the SLED dataset (already preprocessed with the preprocess_sled_dataset.py
  script) and the MVSEC dataset (preprocessed with the preprocess_mvsec_dataset.py script)
  """

  def __init__(self, path_dataset, is_val, transform=None):
    # The path should point to a folder containing at least one preprocessed .pt sequence
    if not path.isdir(path_dataset):
      raise Exception("The path to the dataset should be a folder, containing at least one "
        "preprocessed .pt sequence")

    # If we are in validation, we only want to load the first sequence of each recording, i.e. the
    # sequence finishing with "_seq0000.pt"
    if is_val:
      self.sequences_paths = sorted(glob(path_dataset+"/*_seq0000.pt"))
    else:
      self.sequences_paths = sorted(glob(path_dataset+"/*.pt"))

    # If the folder doesn't contain any .pt file, and if we are not in validation, we throw an 
    # exception
    if not self.sequences_paths and not is_val:
      raise Exception(f"The given folder ({path_dataset}) doesn't contain any .pt file!")

    # We also save the required transform(s)
    self.transform = transform


  def __getitem__(self, index):
    # As the dataset has already been preprocessed, we only have to load the correct .pt file
    sequence = torch.load(self.sequences_paths[index])

    # If a transform is required...
    if self.transform is not None:
      # We save the RNG state for the transform operations, which should be consistent on the whole
      # sequence
      saved_rng_state = torch.get_rng_state()

      # And we apply the transform for each item in the sequence
      for item in sequence:
        # For the LiDAR cloud
        torch.set_rng_state(saved_rng_state)
        item[0] = self.transform(item[0])

        # For all the event volumes
        for i in range(len(item[1])):
          torch.set_rng_state(saved_rng_state)
          item[1][i] = self.transform(item[1][i])

        # For all the previous depth images
        for i in range(len(item[2])):
          torch.set_rng_state(saved_rng_state)
          item[2][i] = self.transform(item[2][i])

        # For all the current depth images
        for i in range(len(item[2])):
          torch.set_rng_state(saved_rng_state)
          item[3][i] = self.transform(item[3][i])

    # And we return the sequence
    return sequence 


  def __len__(self):
    """
    Returns the number of sequences that were found in the given folder
    """
    return len(self.sequences_paths)
