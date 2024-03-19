from __future__ import print_function

import argparse
import pdb
import os
import math
import sys

# internal imports
# from utils.file_utils import save_pkl, load_pkl
# from utils.utils import *
# from utils.core_utils import train, eval_model
# from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
# from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from timeit import default_timer as timer


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    val_cindex = []
    folds = np.arange(start, end)

    for i in folds:
        start = timer()
        seed_torch(args.seed)

        train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)
        
        if 'omic' in args.mode:
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            print("Genomic Dimension", args.omic_input_dim)

        val_latest, cindex_latest = eval_model(datasets, i, args)
        val_cindex.append(cindex_latest)

        #write results to pkl
        save_pkl(os.path.join(args.results_dir, 'split_val_{}_results.pkl'.format(i)), val_latest)
        end = timer()
        print('Fold %d Time: %f seconds' % (i, end - start))

    if len(folds) != args.k: save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else: save_name = 'summary.csv'
    results_df = pd.DataFrame({'folds': folds, 'val_cindex': val_cindex})
    results_df.to_csv(os.path.join(args.results_dir, 'summary.csv'))
