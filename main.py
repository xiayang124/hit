from __future__ import print_function
from __future__ import division
# Torch
import torch.utils.data as data
# Numpy, scipy, scikit-image, spectral
import numpy as np
from utils import (
    get_device,
)
from datasets import get_dataset, HyperX, DATASETS_CONFIG
import torch
import argparse
from utils import HSIRecoder, AvgrageMeter
from evaluation import HSIEvaluation
import utils
import data_gen

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default="IndianPines", choices=dataset_names, help="Dataset to use."
)
parser.add_argument(
    "--model",
    type=str,
    default="hit",
)
# TODO: Learning Rate
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001
)

parser.add_argument(
    "--folder",
    type=str,
    default="./Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=0,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=10,
    help="Percentage of samples to use for training (default: 10%%)",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="fixed",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    default=100,
    help="Training epochs (optional, if" " absent will be set by the model)",
)
group_train.add_argument(
    "--patch_size",
    type=int,
    default=15,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Model name
MODEL = args.model
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

hyperparams = vars(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch
import torch.optim as optim

from vit_pytorch.hit import HiT, ConvPermuteMLP


def get_model(name, band, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
        band
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    kwargs["weights"] = weights

    kwargs.setdefault("patch_size", 15)
    center_pixel = True
    layers = [4, 3, 14, 3]
    transitions = [False, True, False, False]
    segment_dim = [8, 8, 4, 4]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [band, band, 512, 512]  ## for IN 368, for GRSS 256, for PU 168, for KSC 320 for XA 480
    model = HiT(layers, img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=embed_dims, patch_size=3,
                transitions=transitions,
                segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermuteMLP, )
    lr = kwargs.setdefault("learning_rate", 1e-4)  ## for KSC 0.000003
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

    model = model.to(device)
    gamma = 0.7
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


def get_logits(output):
    if type(output) == tuple:
        return output[0]
    return output


def train_epoch(model, train_loader, criterion, optimizer, dataset, epochs):
    evalation = HSIEvaluation(dataset)
    recorder = HSIRecoder()
    epoch_avg_loss = AvgrageMeter()
    total_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        model.train()
        epoch_avg_loss.reset()
        for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)

            optimizer.zero_grad()
            loss = criterion(batch_pred, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_avg_loss.update(loss.item(), batch_data.shape[0])
        recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
        print(
            '[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (
                epoch + 1,
                epoch_avg_loss.get_avg(),
                total_loss / (epoch + 1),
                loss.item(), epoch_avg_loss.get_num()))
        if (epoch + 1) % 10 == 0:
            y_pred_test, y_test = test(test_loader)
            temp_res = evalation.eval(y_pred_test, y_test)
            recorder.append_index_value("train_oa", epoch + 1, temp_res['oa'])
            recorder.append_index_value("train_aa", epoch + 1, temp_res['aa'])
            recorder.append_index_value("train_kappa", epoch + 1, temp_res['kappa'])
            print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (
                epoch + 1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
    final_pred_test, final_test = test(test_loader)
    temp_res = evalation.eval(final_test, final_pred_test)
    recorder.record_eval(temp_res)
    return recorder


def test(test_loader):
    count = 0
    model.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = get_logits(model.forward(inputs))
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
    return y_pred_test, y_test


for dataset_name in ['Honghu', 'Indian', 'Pavia']:
    for train_num in [0.1, 5, 10, 15, 20, 25, 30]:
        mat_file = "{}_{}_split.mat".format(dataset_name, train_num)
        if utils.result_file_exists('./dataset/{}'.format(dataset_name), mat_file):
            print("{} had been generated...skip".format(mat_file))
            continue
        data_gen.generate_data(dataset_name, train_num)
print("All data had been generated!")

for dataset_name in ['Indian', 'Pavia', 'Honghu']:
    if dataset_name == "Indian":
        norm_band = 400
    elif dataset_name == "Pavia":
        norm_band = 208
    elif dataset_name == "Honghu":
        norm_band = 544
    for train_num in [5, 10, 15, 20, 25, 30, 0.1]:
        for train_time in range(1):
            uniq_name = "{}_{}_{}_hit.json".format(dataset_name, train_num, train_time)
            if utils.result_file_exists('./save_path', uniq_name):
                print('%s has been run. skip...' % uniq_name)
                continue
            print("begin training {}".format(uniq_name))
            # Load the dataset
            img, TR, TE, LABEL_VALUES, IGNORED_LABELS, _, _ = get_dataset(dataset_name, train_num)
            gt = TR + TE
            # Number of classes
            N_CLASSES = len(LABEL_VALUES)
            # Number of bands (last dimension of the image tensor)
            N_BANDS = img.shape[-1]

            input_band = N_BANDS
            if dataset_name == "Indian":
                norm_band = 400
            elif dataset_name == "Pavia":
                norm_band = 208
            elif dataset_name == "Honghu":
                norm_band = 544

            hyperparams.update(
                {
                    "n_classes": N_CLASSES,
                    "n_bands": N_BANDS,
                    "ignored_labels": IGNORED_LABELS,
                    "device": CUDA_DEVICE,
                }
            )

            hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
            print(
                "{} samples selected (over {})".format(
                    np.count_nonzero(TR), np.count_nonzero(gt)
                )
            )
            print(
                "Running an experiment with the {} model".format(MODEL),
                "run {}/{}".format(train_time + 1, 3),
            )

            model, optimizer, loss, hyperparams = get_model(MODEL, norm_band, **hyperparams)

            train_dataset = HyperX(img, TR, input_band, **hyperparams)
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=hyperparams["batch_size"],
                # pin_memory=hyperparams['device'],
                shuffle=True,
            )
            test_dataset = HyperX(img, TE, input_band, **hyperparams)
            test_loader = data.DataLoader(
                test_dataset,
                # pin_memory=hyperparams['device'],
                batch_size=hyperparams["batch_size"],
            )
            print(hyperparams)

            recorder = train_epoch(model, train_loader, loss, optimizer, hyperparams["dataset"], hyperparams["epoch"])
            path = "./save_path/{}_{}_{}_hit".format(dataset_name, train_num, train_time)
            recorder.to_file(path)
        