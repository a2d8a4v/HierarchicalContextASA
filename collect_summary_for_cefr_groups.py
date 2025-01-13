import argparse
import os
import json
import numpy as np

import torch

import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix

from tools.utils import (
    cal_pccs,
    _accuracy_within_margin,
    _compute_within_mcacc,
    _compute_mcrmse,
    _compute_over_estimate_rate,
    _compute_under_estimate_rate,
    _compute_overestimate_mcrate,
    _compute_underestimate_mcrate,
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_dirs", type=str, nargs='+', default="./test", help="directory to dump experiments")
parser.add_argument("--output_dir", type=str, default="./test", help="directory to dump experiments")

args = parser.parse_args()

# functions
def round_num(num):
    return round(num, 3)

# debug
args.which_set = 'eval'
input_dirs = args.input_dirs
output_dir = args.output_dir
which_set = args.which_set

alsy_dir = os.path.join(output_dir, "analysis")
if not os.path.exists(alsy_dir) : os.makedirs(alsy_dir)

# json collect files
json_collect = []
for input_dir in input_dirs:
    
    json_path = os.path.join(input_dir, "analysis", "metrics.json")

    f = open(json_path, 'r')
    json_collect.append(json.load(f))

saved_labels = []
saved_predictions = []
for j in json_collect:
    saved_labels.append(j.get('saved_labels'))
    saved_predictions.append(j.get('saved_predictions'))

# json collect files
score2modelcount_result = {}
for modelcount, j in enumerate(json_collect):
    kk = j.get('saved_labels')
    vv = j.get('saved_predictions')
    labels2predictions = {}
    for (k, v) in zip(kk, vv):
        k = int(k)
        labels2predictions.setdefault(
            k,
            []
        ).append(
            v
        )

    for k, v in labels2predictions.items():
        labl = np.array([float(k)]*len(v))
        pred = np.array(v)
        labl_t = torch.tensor(labl)
        pred_t = torch.tensor(pred)
        d = {
            'rmse': float(np.sqrt(((pred - labl) ** 2).mean())),
            'within_0.5': float(_accuracy_within_margin(pred_t, labl_t, 0.5)),
            'within_1': float(_accuracy_within_margin(pred_t, labl_t, 1)),
            'oe_rate': float(_compute_over_estimate_rate(pred_t, labl_t)),
            'ue_rate': float(_compute_under_estimate_rate(pred_t, labl_t)),
        }
        score2modelcount_result.setdefault(
            k,
            []
        ).append(d)

keys = {
    'rmse': [],
    'within_0.5': [],
    'within_1': [],
    'oe_rate': [],
    'ue_rate': [],
}

# save summarized information
save_file_csv_path = os.path.join(alsy_dir, 'groups_summary.csv')
f = open(save_file_csv_path, 'w')
f.write('{}\n'.format(','.join(list(keys.keys()))))
for score, vv in score2modelcount_result.items():
    means = {i: np.mean(np.array([v[i] for v in vv])) for i in keys.keys()}
    stds = {i: np.std(np.array([v[i] for v in vv])) for i in keys.keys()}
    a = []
    b = []
    for k in list(keys.keys()):
        a.append(f'{round_num(means[k]):.3f}')
        b.append(f'{round_num(stds[k]):.3f}')
    f.write('{}\n'.format(','.join(a)))
    f.write('{}\n'.format(','.join(b)))
f.close()

# save summarized information
save_file_ltx_path = os.path.join(alsy_dir, 'groups_summary.latex')
f = open(save_file_ltx_path, 'w')
f.write('{}\n'.format(','.join(['score']+list(keys.keys()))))
for score, vv in score2modelcount_result.items():
    means = {i: np.mean(np.array([v[i] for v in vv])) for i in keys.keys()}
    stds = {i: np.std(np.array([v[i] for v in vv])) for i in keys.keys()}
    a = []
    b = []
    for k in list(keys.keys()):
        a.append(f'{round_num(means[k]):.3f}')
        b.append(f'{round_num(stds[k]):.3f}')
    c = [str(score)]
    for ai, bi in zip(a, b):
        co = '\makecell{'+ai+' \\\\ $\pm$ '+bi+'}'
        c.append(co)
    f.write('{}\n'.format(' & '.join(c)))
f.close()

