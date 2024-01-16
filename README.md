# Learning to Estimate Two Dense Depths from LiDAR and Event Data

![Example results](https://www.hds.utc.fr/~vbrebion/dokuwiki/_media/fr/teaser_learning_to_estimate.png)

This repository holds the code associated with the "Learning to Estimate Two Dense Depths from LiDAR and Event Data" article. If you use this code as part of your work, please cite:

```BibTeX
@inproceedings{Brebion2023LearningTE,
  title={Learning to Estimate Two Dense Depths from LiDAR and Event Data},
  author={Vincent Brebion and Julien Moreau and Franck Davoine},
  booktitle={Image Analysis},
  publisher={Springer Nature Switzerland},
  pages={517-533},
  year={2023}
}
```

## The SLED dataset

Details and download links for the SLED dataset can be found in its dedicated GitHub repository: <https://github.com/heudiasyc/SLED>.

## Overview

In this work, we propose a novel network, ALED (Asynchronous LiDAR and Events Depths densification network), able to fuse information from a LiDAR sensor and an event-based camera to estimate dense depths. At each step, to solve the "potential change of depth problem", we propose to estimate two depth maps: one "before" the events happen, and one "after" the events happen.

## Installation

**Note:** we recommend using `micromamba` (<https://github.com/mamba-org/mamba>) as a lighter and much faster alternative to `conda`/`miniconda`.\
However, you can safely replace `micromamba` by `conda` in the following commands if you prefer!

To install the dependencies, create a micromamba environment as follows:

```txt
micromamba create --name aled
micromamba activate aled
micromamba install pytorch torchvision pytorch-cuda=12.1 h5py matplotlib opencv pandas pyyaml tensorboard tqdm -c pytorch -c nvidia -c conda-forge
```

**Note:** PyTorch versions between 1.13 and 2.1 have been tested and are compatible with this code. Slightly older/newer versions should also be compatible, but try to stick to these versions if possible!

Once the environment is created, you can then clone this repository:

```txt
git clone https://github.com/heudiasyc/ALED.git
```

## Preprocessing the datasets

If you wish to use either our SLED dataset or the MVSEC dataset (either for training or testing), we first advise you to preprocess them (this is mandatory for the moment for the MVSEC dataset!). By doing so, the data formatting, normalization, LiDAR projection, ... steps are applied on the whole dataset once, rather than each time it is loaded, greatly accelerating the training and testing.

However, be aware that preprocessing the datasets greatly increases disk space usage, as each recording is converted into multiple small sequences.

To preprocess the SLED dataset, use the following commands:

```txt
micromamba activate aled
python3 preprocess_sled_dataset.py <path_raw> <path_processed> 5 <lidar_clouds_per_seq> 200.0 90.0 -j 8
```

where:

- `<path_raw>` is the path to the folder containing the raw SLED dataset;
- `<path_processed>` is the path to the output folder which will contain the processed SLED dataset;
- `5` is the number of bins B used to create the event volumes;
- `<lidar_clouds_per_seq>` is the number of LiDAR clouds contained per sequence (set `3` for the training set, `20` for the validation set, and `0` for the testing set);
- `200.0` is the maximum LiDAR range (in meters);
- `90.0` is the FOV of the event camera (in degrees);
- `-j 8` is the number of processes spawned in parallel to preprocess the dataset.

To preprocess the MVSEC dataset, use the following command:

```txt
micromamba activate aled
python3 preprocess_mvsec_dataset.py <path_raw> <path_processed> 5 <lidar_clouds_per_seq> 100.0 -j 8
```

where:

- `<path_raw>` is the path to the folder containing the raw MVSEC dataset;
- `<path_processed>` is the path to the output folder which will contain the processed MVSEC dataset;
- `5` is the number of bins B used to create the event volumes;
- `<lidar_clouds_per_seq>` is the number of LiDAR clouds contained per sequence (set `3` for the training set, `20` for the validation set, and `0` for the testing set);
- `100.0` is the maximum LiDAR range (in meters);
- `-j 8` is the number of processes spawned in parallel to preprocess the dataset.

## Testing

If you only want to test the ALED network on the SLED or the MVSEC datasets, pretrained sets of weights (the ones used in the article) are given in the `saves/` folder. Given their size, they are stored with Git LFS, so you will need to retrieve them with the following command before being able to use them:

```bash
git lfs pull
```

If you rather wish to test the network after training it by yourself, see the [Training](#training) section first.

In both cases, use the following commands to test the network:

```bash
micromamba activate aled
python3 test.py configs/test_sled.json path/to/checkpoint.pth # To test on the SLED dataset
python3 test.py configs/test_mvsec.json path/to/checkpoint.pth # To test on the MVSEC dataset
```

By default, the testing code will try to use the first GPU available. If required, you can run the testing on the GPU of your choice (for instance here, GPU 5):

```txt
CUDA_VISIBLE_DEVICES=5 python3 test.py configs/test_sled.json path/to/checkpoint.pth
```

Results are saved in the `results/` folder, as two .txt files:

- one ending with `_per_seq.txt`, giving the detailed results for each sequence of the dataset;
- and one ending with `_global.txt`, giving a summary of the results.

If needed, the code to test the more naive Nearest Neighbor approach is also given, and can be used as follows:

```txt
micromamba activate aled
python3 test_nearest_neighbor.py configs/test_sled.json
```

## Training

If you wish to train the ALED network on the SLED or the MVSEC datasets, use the following commands:

```bash
micromamba activate aled
python3 train.py configs/train_sled.json # For the SLED dataset
python3 train.py configs/train_mvsec.json # For the MVSEC dataset
```

You will need to customize the configuration files to fit your environment (the configuration here is intended for a server with 4 GPUs). In particular, you will need to fill in the paths of the training and validation datasets (either the raw or preprocessed version).

By default, the training code will try to use all the GPUs available. If required, you can run the training on a subset of GPUs (for instance here, GPUs 0 to 3):

```txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py configs/train_sled.json
```

You can also specify a checkpoint if you wish to restart the training from a given epoch or to fine-tune the network:

```txt
micromamba activate aled
python3 train.py configs/finetune_mvsec.json --cp path/to/checkpoint.pth
```

If you wish to follow the progress of the training, all data (losses, validation error, images, ...) is saved in Tensorboard. To display it, open a second terminal and use the following command:

```txt
micromamba activate aled
tensorboard --logdir runs/ --samples_per_plugin "images=4000"
```

You can then open the web browser of your choice, and use the `http://localhost:6006/` address to access the results. If needed, you can also adapt the number of images if you want to adjust the memory usage of Tensorboard.

## Code details

The code was developed so as to be as simple and as clean as possible. For a better comprehension especially, the ALED network was split into submodules, each submodule corresponding to a block in Figure 2 and 3 of the reference article.

Each file and each function was properly documented, so do not hesitate to take a look at them!
