"""Python program to evaluate ML model results."""

from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset

import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from sklearn.manifold import TSNE


plt.rcParams.update({
    # Figure size
    "figure.figsize": (8, 6),

    # Ticks on all sides
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,

    # Major and minor ticks visibility
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # Tick direction and size
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,

    # Tick width
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,

    # Grid for major ticks
    "axes.grid": False,
    #"grid.which": "major",
    #"grid.linestyle": "--",
    #"grid.color": "gray",
    #"grid.alpha": 0.5,

    # Font sizes for labels and ticks
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})


def read_unequal_table(txt_file, num_columns=383, filler=''):
    """Read table from file."""
    table = []
    with open(txt_file) as file:
        for row in file:
            values = row.split()
            if len(values) > num_columns:
                raise ValueError('Number of values more than num_columns')
            elif len(values) < num_columns:
                diff = num_columns - len(values)
                padding = [filler]*diff
                values = values + padding
            else:
                pass
            table.append(values)
    return np.array(table)


def read_norm_spectra(txt_file):
    """Read the saved set of normalized spectra"""
    alldata = read_unequal_table(txt_file)
    srcids = alldata[:, 0].astype(str)
    normed_spectra = alldata[:, 1:-2].astype(float)
    netcounts = alldata[:, -2].astype(float)
    src_class = alldata[:, -1].astype(str)
    return srcids, normed_spectra, netcounts, src_class


def refine_labels(label_array):
    """Refine labels to 4 classes."""
    refined_labels = np.full(len(label_array), '', dtype='<U16')
    for i, label in enumerate(label_array):
        if label == '':
            continue
        if label == 'AGN' or label == 'CV':
            refined_labels[i] = label
        elif label == 'LM-STAR' or label == 'HM-STAR' or label == 'YSO':
            refined_labels[i] = 'STAR'
        else:
            refined_labels[i] = 'NS/BH'
    return refined_labels


def check_spectra(normspec, refinedclass):
    """Check the normalized spectra of different classes."""
    en_bins = np.linspace(0.5, 10, len(normspec[0]))
    classes_unique = np.unique(refinedclass)
    #plt.figure(figsize=(20, 12))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Normalized counts [/bin]')
    plt.xscale('log')
    plt.xlim(0.5, 10.0)
    for classes in classes_unique:
        if classes != '':
            plt.plot(en_bins,
                     np.mean(normspec[refinedclass == classes], axis=0),
                     label=classes)
    plt.legend()
    plt.show


def plothist(value_arrs, xscale='linear', yscale='linear', xlabel=None,
             ylabel=None, labels=None):
    """Plot histogram."""
    plt.hist(value_arrs)
    plt.yscale(yscale)
    plt.xscale(xscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels)
    plt.show()


def pn_analysis_pseudolabel(pn_labelled_specs, pn_bgcountsfile):
    """Analyze the PN results."""
    if not Path(pn_labelled_specs).exists():
        raise ValueError("Spectra File does not exist.")
    if not Path(pn_bgcountsfile).exists():
        raise ValueError('Bg count file does not exist')

    pn_srcids, pn_normspec, pn_counts, pn_class = read_norm_spectra(
        pn_labelled_specs)
    pn_class_refined = refine_labels(pn_class)
    for src_type in np.unique(pn_class_refined):
        num_classes = len(np.where(pn_class_refined == src_type)[0])
        print(f"Number of {src_type}s: {num_classes}")

    check_spectra(pn_normspec, pn_class_refined)
