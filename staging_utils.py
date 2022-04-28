################################################################################
################################ Python Imports ################################
################################################################################

import os
import sys
import yaml
import argparse
import logging

from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from monai.networks.nets import (
    UNet
)

from monai.networks.layers import Norm
from monai.losses import (
    DiceFocalLoss,
    DiceCELoss
)

import wandb

################################################################################
################################# Local Imports ################################
################################################################################

from dataset import (
    OctDataset
)

from evaluation import (
    Evaluator
)


################################################################################
################################ Main Functions ################################
################################################################################

def parse_user_inputs():

    parser = argparse.ArgumentParser(description='OCT Segmentation')
    parser.add_argument('--config', default='configs/config.yaml')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.full_load(f)

    ### Populate Args with key-value pails specified in YAML File
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    return args

def build_wandb_monitor(args):

    project_name = args.wandb_project_name
    if not project_name:
        raise ValueError("No W&B Project Name Specified")

    wandb.init(

        project=project_name,

        entity="oct_segmentation_uncertainty", ### Group Name

        config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "loss_function": args.loss_function,
            "optimizer": "adam",
            "channels": args.channels,
            "stides": args.strides,
            "include_background": args.include_background
        }
    )

    return wandb, wandb.config

def build_data_loaders(args):

    ### Local Helper For Configuring Augmentations Via Config
    def build_augumentation_transforms(args):

        #### Needed To Get Labels in [1, H, W] For Training Loop
        def AddLabelChannel(labels, **params):
            return labels.unsqueeze(dim=0)

        transforms = A.Compose([

            ###############################################
            #### ADD Albumentation Augmentations Here #####
            ###############################################

            ToTensorV2(),
            A.Lambda(name="Add Channel Dimension",
                     mask=AddLabelChannel,
                     always_apply=True)
        ])

        return (transforms, transforms, transforms)

    ### Local Helper For Making Sure Specifed Directory Structure
    ### Is Correct
    def test_for_dir(path):
        if not os.path.exists(path):
            raise ValueError(f"{path} Not Found!!!")

    base_data_dir = args.data_path_dir
    test_for_dir(base_data_dir)

    train_data_image_dir = os.path.join(base_data_dir, "train_images")
    train_data_label_dir = os.path.join(base_data_dir, "train_labels")
    val_data_image_dir   = os.path.join(base_data_dir, "val_images")
    val_data_label_dir   = os.path.join(base_data_dir, "val_labels")
    test_data_image_dir  = os.path.join(base_data_dir, "test_images")
    test_data_label_dir  = os.path.join(base_data_dir, "test_labels")

    test_for_dir(train_data_image_dir)
    test_for_dir(train_data_label_dir)
    test_for_dir(val_data_image_dir)
    test_for_dir(val_data_label_dir)
    test_for_dir(test_data_image_dir)
    test_for_dir(test_data_label_dir)

    train_transforms, val_transforms, test_transforms = \
        build_augumentation_transforms(args)

    train_ds = OctDataset(train_data_image_dir,
                          train_data_label_dir,
                          sort=True,
                          transforms=train_transforms,
                          num_samples=args.num_train_samples)

    val_ds   = OctDataset(val_data_image_dir,
                          val_data_label_dir,
                          sort=True,
                          transforms=val_transforms,
                          num_samples=args.num_val_samples)

    test_ds  = OctDataset(test_data_image_dir,
                          test_data_label_dir,
                          sort=True,
                          transforms=test_transforms,
                          num_samples=args.num_test_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_train_data,
        shuffle=args.shuffle_train_data)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_val_data,
        shuffle=args.shuffle_val_data)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_test_data,
        shuffle=args.shuffle_test_data)

    return \
        (len(train_ds), train_loader), \
        (len(val_ds),   val_loader), \
        (len(test_ds),  test_loader)

def create_path_to_new_weight_file(args):

    timestamp = datetime.now()

    model_dir = os.path.join(args.model_dir, f"Model_{timestamp}")

    print(f"Creating {model_dir} For Storing Model Weights")
    os.mkdir(model_dir)

    weight_file = os.path.join(model_dir, args.new_weights_file)
    print(f"Best Weights Will Be Stored @ {weight_file}")

    return weight_file

def find_path_to_weight_file(args):

    weight_file = args.path_to_weights
    if os.path.exists(weight_file):
        return weight_file

    raise ValueError(f"Failed To Find Weights @ {weight_file}")


def build_model(args, config=None, device="cpu"):

    inputs = (args, config)[config is not None]

    if args.model == "unet":

        model= UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=inputs.channels,
            strides=inputs.strides,
            norm=Norm.BATCH,
            dropout=inputs.strides
        ).to(device)

    else:
        raise ValueError(f"Unsupported Model: {args.model}")

    return model


def build_loss_function(args, config=None):

    inputs = (args, config)[config is not None]

    loss = inputs.loss_function
    include_background = inputs.loss_include_background

    to_onehot = args.loss_to_onehot
    softmax = args.loss_softmax

    if loss == 'DiceFocalLoss':
        return DiceFocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot,
            softmax=softmax
        )
    elif loss =='DiceCELoss':
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot,
            softmax=softmax
        )

    raise ValueError(f"Unsupported Loss Function: {loss}")


def build_optimizer(args, model, config=None):
    inputs = (args, config)[config is not None]
    if (inputs.learning_rate == 0):
        raise ValueError("Invalid Input! Learning Rate is Zero!")
    return Adam(model.parameters(), inputs.learning_rate)


def build_scheduler(args, optimizer):
    return ReduceLROnPlateau(
        optimizer,
        'min',
        patience=args.patience,
        min_lr=args.min_learning_rate)


def build_train_evaluator(args, device):

    base_dir, report_file = args.report_base_dir, args.train_report_file
    if base_dir and report_file:
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        report_file = os.path.join(base_dir, report_file)
    else:
        report_file = None

    return Evaluator(
        interval=args.train_val_interval,
        k_stochastic_passes=args.train_k_stochastic_passes,
        num_to_visualize=args.train_preds_to_visualize,
        uncertainty=args.train_uncertainty,
        results_file=report_file,
        num_classes=args.num_classes,
        device=device
    )


def build_test_evaluators(args, device):

    base_dir = args.report_base_dir
    val_report, test_report = args.val_report_file, args.test_report_file

    if not base_dir:
        raise ValueError(f"No Directory Specifed For Evaluation Results.")

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    if (not val_report):
        raise ValueError(f"No Validation Report File Specified.")

    if (not test_report):
        raise ValueError(f"No Validation Report File Specified.")

    val_report = os.path.join(base_dir, val_report)
    test_report = os.path.join(base_dir, test_report)

    return \
        Evaluator(
            interval=0,
            k_stochastic_passes=args.val_k_stochastic_passes,
            num_to_visualize=args.val_preds_to_visualize,
            uncertainty=args.val_uncertainty,
            results_file=val_report,
            num_classes=args.num_classes,
            device=device
        ), \
        Evaluator(
            interval=0,
            k_stochastic_passes=args.test_k_stochastic_passes,
            num_to_visualize=args.test_preds_to_visualize,
            uncertainty=args.test_uncertainty,
            results_file=test_report,
            num_classes=args.num_classes,
            device=device
        )
