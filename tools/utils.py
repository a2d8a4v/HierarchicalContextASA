#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import csv
import shutil
import copy
import pickle
import datetime
import torch
import numpy as np
import scipy.stats as stats

from .logger import *

import sys
sys.setrecursionlimit(10000)


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}", 
        "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'} 

CEFR2INT = {
    0: 0,
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C': 5,
}
INT2CEFR = {v: k for k, v in CEFR2INT.items()}

BERT2ABB = {
    'bert-base-uncased': 'bert',
    'roberta-base': 'roberta',
    'xlm-roberta-base': 'xlm',
    'distilroberta-base': 'distil',
    'allenai/longformer-base-4096': 'longformer',
    'sentence-transformers/all-mpnet-base-v2': 'mpnet',
    'databricks/dolly-v2-12b': 'dolly'
}
pos_tag_labels = ['X', '.', '\'\'', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                  'MD', 'NFP', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                  'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
deprel_labels = ['X', 'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'vocative', 'expl', 'dislocated',
                 'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod', 'acl', 'amod', 'det',
                 'clf', 'case', 'conj', 'cc', 'fixed', 'flat', 'compound', 'list', 'parataxis', 'orphan', 'goeswith',
                 'reparandum', 'punct', 'root', 'dep']
ged_difficult_labels = ['X', 'n_inf', 'n_num', 'n_cs', 'n_cnt', 'n_cmp', 'n_lxc', 'v_inf', 'v_agr', 'v_fml', 'v_tns', 'v_asp',
                        'v_vo', 'v_fin', 'v_ng', 'v_qst', 'v_cmp', 'v_lxc', 'mo_lxc', 'aj_inf', 'aj_us', 'aj_num', 'aj_agr',
                        'aj_qnt', 'aj_cmp', 'aj_lxc', 'av_inf', 'av_us', 'av_pst', 'av_lxc', 'prp_cmp', 'prp_lxc1', 'prp_lxc2',
                        'at', 'pn_inf', 'pn_agr', 'pn_cs', 'pn_lxc', 'con_lxc', 'rel_cs', 'rel_lxc', 'itr_lxc', 'o_je', 'o_lxc',
                        'o_odr', 'o_uk', 'o_uit',]
POS2INT = {v:i for i, v in enumerate(pos_tag_labels)}
DEP2INT = {v:i for i, v in enumerate(deprel_labels)}
GED2INT = {v:i for i, v in enumerate(ged_difficult_labels)}

def clean(x): 
    x = x.lower()
    return re.sub( 
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", 
            lambda m: REMAP.get(m.group()), x)


def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def accuracy(preds: torch.Tensor,
             labels: torch.Tensor) -> float:
    correct = (preds == labels).double()
    correct = correct.sum().item()
    return correct / len(labels)


def correlation(output: torch.Tensor,
                labels: torch.Tensor,
                mask: torch.Tensor,
                mode: str,
                conversion: str = "max") -> float:
    if mode == "regression":
        preds = output.squeeze()
    elif mode == "classification":
        # Convert classification to real values
        betas = output.new_tensor([cefr_to_beta(cefr) for cefr in CEFR_LEVELS])
        if conversion == "max":
            pred_idx = torch.argmax(output, dim=1)
            preds = torch.gather(betas, 0, pred_idx)
        else:
            preds = F.softmax(output, dim=1)
            preds = torch.sum(preds * betas, dim=1)


    preds = preds[mask.nonzero()].cpu()
    labels = labels[mask.nonzero()].cpu()
    rho, _ = stats.spearmanr(preds.detach().numpy(), labels.numpy())
    return rho


def compute_micro_overestimate_rate(preds: torch.Tensor,
                                    labels: torch.Tensor) -> float:
    return torch.sum((preds > labels).double()).item() / len(preds) * 100
    
def compute_macro_overestimate_rate(preds: torch.Tensor,
                                    labels: torch.Tensor) -> float:
    unique_classes = torch.unique(labels)
    num_classes = len(unique_classes)
    score_rate = 0.
    
    for c in unique_classes:
        indices = torch.where(labels == c)
        score_predictions = preds[indices]
        score_targets = labels[indices]
        score_rate += 1 / num_classes * compute_micro_overestimate_rate(score_predictions, score_targets)

    return score_rate

def compute_micro_underestimate_rate(preds: torch.Tensor,
                              labels: torch.Tensor) -> float:
    return torch.sum((preds < labels).double()).item() / len(preds) * 100
    
def compute_macro_underestimate_rate(preds: torch.Tensor,
                              labels: torch.Tensor) -> float:
    unique_classes = torch.unique(labels)
    num_classes = len(unique_classes)
    score_rate = 0.
    
    for c in unique_classes:
        indices = torch.where(labels == c)
        score_predictions = preds[indices]
        score_targets = labels[indices]
        score_rate += 1 / num_classes * compute_micro_underestimate_rate(score_predictions, score_targets)

    return score_rate


def _compute_over_estimate_rate(score_predictions, score_target):
    margin = 0.5
    return torch.sum(
        torch.where(
            (score_predictions - score_target) > margin,
            torch.ones(len(score_predictions)),
            torch.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

def _compute_under_estimate_rate(score_predictions, score_target):
    margin = -0.5
    return torch.sum(
        torch.where(
            (score_predictions - score_target) < margin,
            torch.ones(len(score_predictions)),
            torch.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

def _accuracy_within_margin(score_predictions, score_target, margin):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return torch.sum(
        torch.where(
            torch.abs(score_predictions - score_target) <= margin,
            torch.ones(len(score_predictions)),
            torch.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

# See: https://github.com/teinhonglo/automated-english-transcription-grader/blob/master/helpers/metrics.py
# See: https://github.com/teinhonglo/automated-english-transcription-grader/blob/master/local/metrics_cpu.py
def _compute_rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

def _compute_mcrmse(all_score_predictions, all_score_targets):
    # print(all_score_predictions.shape, all_score_targets.shape)
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = (all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _compute_rmse(score_predictions, score_targets)
    
    return score_rmse

def _compute_within_mcacc(all_score_predictions, all_score_targets, margin):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = torch.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _accuracy_within_margin(score_predictions, score_targets, margin)
    
    return score_rmse

def _compute_underestimate_mcrate(all_score_predictions, all_score_targets):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = torch.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _compute_under_estimate_rate(score_predictions, score_targets)
    
    return score_rmse

def _compute_overestimate_mcrate(all_score_predictions, all_score_targets):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = torch.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _compute_over_estimate_rate(score_predictions, score_targets)
    
    return score_rmse

def cal_pccs(x, y):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    n = len(x)
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def weight_groups(numbers, hps):
    numbers = numbers.tolist()
    groups = []
    current_group = []
    for number in numbers:
        if not current_group or number == current_group[-1] + 1:
            current_group.append(number)
        else:
            groups.append(current_group)
            current_group = [number]
    groups.append(current_group)
    # assert len(groups) == hps.batch_size
    w = []
    for g in groups:
        w += [float(1/len(g))]*len(g)
    w = torch.FloatTensor(w).to(hps.device)
    return w

def make_up_weights(weight_dict, labels, hps):
    if isinstance(labels, torch.Tensor):
        labels = labels.to(torch.int64).cpu().detach().tolist()
    r = list()
    for i in labels:
        r.append(1-weight_dict[i])
    r = torch.FloatTensor(r).to(hps.device)
    return r

def make_up_wcefr_weights(weight_dict, labels, hps):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().tolist()
    labels = list(map(int, list(map(round, labels))))
    r = list()
    for i in labels:
        r.append(1-weight_dict[i])
    r = torch.FloatTensor(r).to(hps.device)
    return r

def pikleOpen(filename):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p

def pickleStore(savethings , filename):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    columns = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for i, line in enumerate(reader):
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if i == 0:
                columns = line
                continue
            lines.append(line)
        return columns, lines


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count