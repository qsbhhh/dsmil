
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
# print(np.zeros(2))


bags_csv = 'datasets/TCGA-lung/TCGA-lung.csv'
bags_path = pd.read_csv(bags_csv)
train_path = bags_path.iloc[0:int(len(bags_path)*(0.8)), :]
test_path = bags_path.iloc[int(len(bags_path)*(0.8)):, :]
print('train_path:')
print(train_path)
print('train_path.iloc[0]:')
print(train_path.iloc[0])
print('train_path.iloc[0].iloc[0]:')
print(train_path.iloc[0].iloc[0])
print('train_path.iloc[0].iloc[1]:')
print(train_path.iloc[0].iloc[1])