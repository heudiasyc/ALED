#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file can be used to train or fine-tune the ALED network, on either the SLED or the MVSEC
datasets, as describred in the "Learning to Estimate Two Dense Depths from LiDAR and Event Data"
article.
"""

import argparse
from datetime import datetime
import json
from os import path

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Pad, RandomCrop, RandomHorizontalFlip
from tqdm import tqdm

from aled import ALED
from losses import DynamicL1MultiscaleGradLossScaled
from preprocessed_dataset_loader import PreprocessedDataset
from raw_dataset_loader_sled import SLEDRawDataset
from visualization import depth_image_to_img_color, event_volume_to_img, lidar_proj_to_img_color, \
                          predicted_depths_to_img_color, prediction_error_to_img_color


def parse_args():
  """Args parser"""
  parser = argparse.ArgumentParser()
  parser.add_argument("config_file", help="Path to the JSON config file to use for training")
  parser.add_argument("--cp", default=None, help="Checkpoint to restart from (optional)")
  return parser.parse_args()


def display_count_parameters(model):
  """
  Utility function to display the number of parameters of a network in PyTorch.
  Thanks to https://stackoverflow.com/a/62508086 
  """
  total_params = 0
  for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
      continue
    params = parameter.numel()
    print(name, ":", params)
    total_params += params
  print(f"Total Trainable Params: {total_params}")
  return total_params


def main():
  """Main function"""

  # Before doing anything, we must change the torch multiprocessing sharing strategy, to avoid
  # having issues with leaking file descriptors.
  # For more informations, see https://github.com/pytorch/pytorch/issues/973
  torch.multiprocessing.set_sharing_strategy("file_system")

  # We start by loading the config file given by the user
  args = parse_args()
  config = json.load(open(args.config_file))

  # We configure the device for PyTorch
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # We also configure the tensorboard summary writer
  time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
  writer = SummaryWriter(path.join("runs", time_prefix))

  # We setup the transforms we will perform on the training dataset, i.e. random cropping (if
  # required) and random horizontal flipping
  if config["transforms"]["crop_input"]:
    crop_size = config["transforms"]["crop_size"]
    train_transforms = Compose([RandomCrop(crop_size), RandomHorizontalFlip()])
  else:
    train_transforms = Compose([RandomHorizontalFlip()])

  # We setup the transforms we will perform on the validation dataset, i.e. padding (if required)
  if config["transforms"]["pad_val"]:
    pad_size_x = config["transforms"]["pad_size_x"]
    pad_size_y = config["transforms"]["pad_size_y"]
    val_transforms = Pad((pad_size_x, pad_size_y, pad_size_x, pad_size_y))
  else:
    pad_size_x = 0
    pad_size_y = 0
    val_transforms = None

  # We collect the batch_size and num_workers parameters from the config file
  batch_size_train = config["batch_size_train"]
  batch_size_val = config["batch_size_val"]
  num_workers = config["num_workers"]

  # We load the training dataset, create the dataloader, and collect the number of sequences that
  # were loaded
  train_dataset_path = config["datasets"]["path_train"]
  if config["datasets"]["is_preprocessed_train"]:
    train_dataset = PreprocessedDataset(train_dataset_path, False, train_transforms)
  else:
    train_dataset = SLEDRawDataset(train_dataset_path, 5, 3, 200.0, 90.0, False, train_transforms)
  train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                                num_workers=num_workers)
  total_nb_seq_train = len(train_dataloader)

  # We do the same for the validation dataset
  val_dataset_path = config["datasets"]["path_val"]
  if config["datasets"]["is_preprocessed_val"]:
    val_dataset = PreprocessedDataset(val_dataset_path, True, val_transforms)
  else:
    val_dataset = SLEDRawDataset(val_dataset_path, 5, 20, 200.0, 90.0, False, val_transforms)
  val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=False,
                              num_workers=num_workers)
  total_nb_seq_val = len(val_dataloader)

  # We initialize the network
  # The use of nn.DataParallel allows for the use of multiple GPUs if available
  # Note: according to the documentation, nn.DistributedDataParallel should be used instead (see
  # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html for more details)
  # If a checkpoint has been specified, we use it to set the state of the network
  model = ALED(10, 1)
  model = nn.DataParallel(model)
  if args.cp is not None:
    model.load_state_dict(torch.load(args.cp))
  model.to(device)

  # We display its number of parameters (debug, uncomment if needed)
  #display_count_parameters(model)

  # We initialize the loss criterion, Adam optimizer, and the scheduler
  criterion = DynamicL1MultiscaleGradLossScaled(5)
  learning_rate = config["learning_rate"]
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # We also initialize the GradScaler, necessary for training with mixed precision
  scaler = amp.GradScaler()

  # We set the number of epochs
  num_epochs = config["epochs"]

  # We get the "how often should we display the losses in the terminal" parameter
  losses_display_every_x = config["losses_display_every_x"]

  # Then, for each epoch
  for epoch in tqdm(range(num_epochs), "Epochs"):
    # We set to zero the running losses
    running_bf_loss_l1 = 0.0
    running_bf_loss_ms = 0.0
    running_af_loss_l1 = 0.0
    running_af_loss_ms = 0.0
    running_loss = 0.0

    # We set the model to training mode
    model.train()

    # We compute the weights for the loss
    weight_L1 = 1.0
    if epoch == 0:
      weight_mse_grad_match = config["weight_mse_grad_match_epoch_0"]
    else:
      weight_mse_grad_match = 1.0

    # For each sequence extracted from the dataset...
    for seq_nbr, sequence in enumerate(tqdm(train_dataloader, "Training", leave=False)):
      # We enter the mixed precision mode
      with amp.autocast():
        # We reset the state of the convGRUs
        conv_gru_states = [None, None, None, None, None]

        # We create two arrays, that will hold the ground truth depths and predicted outputs
        ground_truth_depths = []
        pred_depths = []

        # For each item (1 LiDAR proj, events, depth images) in the sequence...
        for item in sequence:
          # We extract the LiDAR projection and feed it to the network
          lidar_proj = item[0].to(device)
          conv_gru_states = model(lidar_proj, conv_gru_states, "lidar")

          # And, for each event volume / each depth image...
          for i, (event_volume, bf_depth_image, af_depth_image) in enumerate(zip(item[1], item[2], item[3])):
            # We upload them to the device
            event_volume = event_volume.to(device)
            bf_depth_image = bf_depth_image.to(device)
            af_depth_image = af_depth_image.to(device)

            # We make sure that the depth images are in the range [0, 1]
            bf_depth_image[bf_depth_image > 1.0] = 1.0
            af_depth_image[af_depth_image > 1.0] = 1.0

            # We feed the event volume to the network
            conv_gru_states = model(event_volume, conv_gru_states, "events")

            # We run a prediction
            pred = model(None, conv_gru_states, "predict")

            # We concatenate the D_bf and D_af depth images as a single 2-channel image, to match
            # the shape of the prediction
            bf_af_depths = torch.cat((bf_depth_image, af_depth_image), dim=1)

            # And we save them
            ground_truth_depths.append(bf_af_depths)
            pred_depths.append(pred)

        # Once the sequence is over, we compute the loss
        bf_loss_l1 = 0.0
        bf_loss_ms = 0.0
        af_loss_l1 = 0.0
        af_loss_ms = 0.0
        for prediction, ground_truth in zip(pred_depths, ground_truth_depths):
          bf_loss_l1_, bf_loss_ms_ = criterion(prediction[:, [0], :, :], ground_truth[:, [0], :, :])
          af_loss_l1_, af_loss_ms_ = criterion(prediction[:, [1], :, :], ground_truth[:, [1], :, :])
          bf_loss_l1 += bf_loss_l1_
          bf_loss_ms += bf_loss_ms_
          af_loss_l1 += af_loss_l1_
          af_loss_ms += af_loss_ms_
        loss = weight_L1*(bf_loss_l1+af_loss_l1) + weight_mse_grad_match*(bf_loss_ms+af_loss_ms)

      # Note: we leave the mixed precision mode here

      # And we apply the backwards pass
      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      # We save the gradients of the parameters of the network (debug, uncomment if needed)
      #for name, param in model.named_parameters():
      #  writer.add_histogram(f'{name}.grad', param.grad, epoch*total_nb_seq_train+seq_nbr)

      # We save the losses for analysis
      running_bf_loss_l1 += bf_loss_l1.item()
      running_bf_loss_ms += bf_loss_ms.item()
      running_af_loss_l1 += af_loss_l1.item()
      running_af_loss_ms += af_loss_ms.item()
      running_loss += loss.item()

      # Finally, we write the losses in tensorboard
      writer.add_scalar("Loss on D_bf depths (L1)", bf_loss_l1.item(), epoch*total_nb_seq_train+seq_nbr)
      writer.add_scalar("Loss on D_bf depths (MS)", bf_loss_ms.item(), epoch*total_nb_seq_train+seq_nbr)
      writer.add_scalar("Loss on D_af depths (L1)", af_loss_l1.item(), epoch*total_nb_seq_train+seq_nbr)
      writer.add_scalar("Loss on D_af depths (MS)", af_loss_ms.item(), epoch*total_nb_seq_train+seq_nbr)
      writer.add_scalar("Total loss", loss.item(), epoch*total_nb_seq_train+seq_nbr)

      # And, if it is needed, we display them
      if (seq_nbr+1)%losses_display_every_x == 0:
        tqdm.write(f"Epoch {epoch+1} / {num_epochs}, seq. {seq_nbr+1}/{total_nb_seq_train}, "
                   f"loss bf l1 = {running_bf_loss_l1/losses_display_every_x:.2f}, "
                   f"loss bf ms = {running_bf_loss_ms/losses_display_every_x:.2f}, "
                   f"loss af l1 = {running_af_loss_l1/losses_display_every_x:.2f}, "
                   f"loss af ms = {running_af_loss_ms/losses_display_every_x:.2f}, "
                   f"total loss = {running_loss/losses_display_every_x:.2f}")
        running_bf_loss_l1 = 0.0
        running_bf_loss_ms = 0.0
        running_af_loss_l1 = 0.0
        running_af_loss_ms = 0.0
        running_loss = 0.0


    # At the end of the epoch, we run a short evaluation on the test dataset, to monitor the
    # progress of the training
    # Before doing so, we must not forget to set the model to evaluation mode
    model.eval()

    # The code run here is nearly the same as the training one (only parts added are the images
    # being pushed to tensorboard and the final metric evaluation part), hence the few comments
    with torch.no_grad():
      with amp.autocast():
        running_val_bf_loss = 0.0
        running_val_af_loss = 0.0

        for seq_nbr, sequence in enumerate(tqdm(val_dataloader, "Validation", leave=False)):
          conv_gru_states = [None, None, None, None, None]
          total_items = len(sequence)
          total_elems_per_item = len(sequence[0][1])

          # We only display the inputs and results every 1 out of 4 sequences, to reduce the
          # tensorboard file size, and to make the validation a bit faster.
          # If needed, full visualization can be reconstructed after the training with the "test.py"
          # script anyway
          should_display = (seq_nbr % 4 == 0)

          for i, item in enumerate(tqdm(sequence, "Val. sequence", leave=False)):
            lidar_proj = item[0].to(device)
            conv_gru_states = model(lidar_proj, conv_gru_states, "lidar")

            # We compute the coordinates to use to remove the padding
            min_x = pad_size_x
            max_x = lidar_proj.shape[3] - pad_size_x
            min_y = pad_size_y
            max_y = lidar_proj.shape[2] - pad_size_y

            # We save the image of the projected LiDAR data (if needed), with padding removed
            if should_display:
              lidax_idx = epoch*total_nb_seq_val*total_items*total_elems_per_item \
                          + seq_nbr*total_items*total_elems_per_item \
                          + i*total_elems_per_item
              lidar_proj_disp = lidar_proj_to_img_color(item[0][:, 0, min_y:max_y, min_x:max_x])
              writer.add_images("lidar_proj", lidar_proj_disp[:, :3, :, :], lidax_idx)

            for j, (event_volume, bf_depth_image, af_depth_image) in enumerate(zip(item[1], item[2], item[3])):
              event_volume = event_volume.to(device)
              bf_depth_image = bf_depth_image.to(device)
              af_depth_image = af_depth_image.to(device)
              bf_depth_image[bf_depth_image > 1.0] = 1.0
              af_depth_image[af_depth_image > 1.0] = 1.0
              conv_gru_states = model(event_volume, conv_gru_states, "events")
              pred = model(None, conv_gru_states, "predict")

              # We correct the prediction, to force it to be in the [0, 1] range
              pred[pred < 0.0] = 0.0
              pred[pred > 1.0] = 1.0

              # We remove any padding before using the data further on
              unpadded_bf_depth_image = bf_depth_image[:, :, min_y:max_y, min_x:max_x]
              unpadded_af_depth_image = af_depth_image[:, :, min_y:max_y, min_x:max_x]
              unpadded_event_volume = event_volume[:, :, min_y:max_y, min_x:max_x]
              unpadded_pred = pred[:, :, min_y:max_y, min_x:max_x]

              # We save images of the input and output data (if needed)
              if should_display:
                # Computation of the current index
                data_idx = lidax_idx + j

                # Display of the D_bf depth image
                bf_depth_image_disp = depth_image_to_img_color(unpadded_bf_depth_image[:, 0, :, :])
                writer.add_images("bf_depth_image", bf_depth_image_disp[:, :3, :, :], data_idx)

                # Display of the D_af depth image
                af_depth_image_disp = depth_image_to_img_color(unpadded_af_depth_image[:, 0, :, :])
                writer.add_images("af_depth_image", af_depth_image_disp[:, :3, :, :], data_idx)

                # Display of the event volume
                event_volume_disp = event_volume_to_img(unpadded_event_volume)
                writer.add_images("event_volume", event_volume_disp, data_idx)

                # Display of the estimated D_bf depths (dense)
                bf_pred_disp = predicted_depths_to_img_color(unpadded_pred[:, 0, :, :])
                writer.add_images("predicted_bf_depths", bf_pred_disp[:, :3, :, :], data_idx)

                # Display of the estimated D_af depths (dense)
                af_pred_disp = predicted_depths_to_img_color(unpadded_pred[:, 1, :, :])
                writer.add_images("predicted_af_depths", af_pred_disp[:, :3, :, :], data_idx)

                # Display error of the estimated D_bf depths
                error_bf = torch.abs(unpadded_bf_depth_image[:, 0, :, :] - unpadded_pred[:, 0, :, :])
                error_bf_disp = prediction_error_to_img_color(error_bf)
                writer.add_images("error_bf_depths", error_bf_disp[:, :3, :, :], data_idx)

                # Display error of the estimated D_af depths
                error_af = torch.abs(unpadded_af_depth_image[:, 0, :, :] - unpadded_pred[:, 1, :, :])
                error_af_disp = prediction_error_to_img_color(error_af)
                writer.add_images("error_af_depths", error_af_disp[:, :3, :, :], data_idx)

              # We compute the numerical error (without forgetting to ignore NaN values)
              val_criterion = nn.L1Loss()

              not_nan_mask_bf = (~torch.isnan(unpadded_bf_depth_image))
              masked_unpadded_pred_bf = unpadded_pred[:, [0], :, :][not_nan_mask_bf]
              masked_unpadded_depth_img_bf = unpadded_bf_depth_image[not_nan_mask_bf]

              not_nan_mask_af = (~torch.isnan(unpadded_af_depth_image))
              masked_unpadded_pred_af = unpadded_pred[:, [1], :, :][not_nan_mask_af]
              masked_unpadded_depth_img_af = unpadded_af_depth_image[not_nan_mask_af]

              error_bf_num = val_criterion(masked_unpadded_pred_bf, masked_unpadded_depth_img_bf).item()
              error_af_num = val_criterion(masked_unpadded_pred_af, masked_unpadded_depth_img_af).item()
              running_val_bf_loss += error_bf_num
              running_val_af_loss += error_af_num

        # At the end, we save the error on the validation set for analysis
        writer.add_scalar("Val. error on D_bf depths", running_val_bf_loss, epoch)
        writer.add_scalar("Val. error on D_af depths", running_val_af_loss, epoch)
        writer.add_scalar("Total val. error", running_val_bf_loss+running_val_af_loss, epoch)

    # We don't forget to save the model at the end of each epoch
    torch.save(model.state_dict(), f"saves/{time_prefix}_{epoch:03d}.pth")


if __name__ == "__main__":
  main()
