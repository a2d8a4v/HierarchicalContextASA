#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import datetime
import os
import time
import json
import glob

import numpy as np
import pandas as pd
from pytablewriter import ExcelXlsxTableWriter

import torch
import torch.nn as nn

from HiGraph import HSumGraph, HSentPromptGraph, HSumPromptGraph, HSumDocGraph
from PromptAwareLSTM import PromptAwareLSTM
from HierConGraph import HierConGraph
from module.dataloader import ExampleSet, ExamplePromptSet, MultiExampleSet, ExampleHierSet, graph_collate_fn
from module.INFOembedding import CEFREmbed, FILLEDEmbed
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.args import get_eval_args
from tools.recipe_name import get_recipe_name
from tools.logger import *
import seaborn as sns
import matplotlib.pyplot as plt
from tools.utils import (
    CEFR2INT,
    BERT2ABB,
    compute_micro_underestimate_rate, 
    compute_macro_underestimate_rate,
    compute_micro_overestimate_rate,
    compute_macro_overestimate_rate,
    cal_pccs,
    _accuracy_within_margin,
    _compute_within_mcacc,
    _compute_mcrmse,
    _compute_over_estimate_rate,
    _compute_under_estimate_rate,
    _compute_overestimate_mcrate,
    _compute_underestimate_mcrate,
    pickleStore
)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report, f1_score

from transformers import AutoTokenizer, AutoConfig

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

aux_objs = ['pos', 'deprel',
            'gra_sim', 'gra_med', 'gra_dif',
            'dif_sim', 'dif_med',
            'laugh', 'sil', 'w_cefr']

def auxiliary_inputs(args, tokenizer, aux_objs):
    auxiliary_inputs_dict = dict()
    for obj in aux_objs:
        if getattr(args, 'use_{}'.format(obj)):
            num_predictions = tokenizer.vocab_size if obj in ['mlm', 'lm'] else len(getattr(bert_data, 'get_{}_labels'.format(obj))())
            auxiliary_inputs_dict[obj] = num_predictions
    return auxiliary_inputs_dict

def load_test_model(model, model_name, eval_dir, save_root, save_dir_name, device):
    """ choose which model will be loaded for evaluation """
    if model_name.startswith('eval'):
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])
    elif model_name.startswith('train'):
        train_dir = os.path.join(save_root, save_dir_name, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])
    elif model_name == "earlystop":
        train_dir = os.path.join(save_root, save_dir_name, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not os.path.exists(bestmodel_load_path):
        logger.error("[ERROR] Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path, map_location=device))

    return model

def round_num(num):
    return round(num, 3)

def run_test(model, dataset, loader, model_name, hps, vocab):
    test_dir = os.path.join(hps.save_root, hps.save_dir_name, "test") # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, hps.save_dir_name, "eval")
    alsy_dir = os.path.join(hps.save_root, hps.save_dir_name, "analysis")
    tsne_dir = os.path.join(hps.save_root, hps.save_dir_name, "tsne")
    if not os.path.exists(test_dir) : os.makedirs(test_dir)
    if not os.path.exists(alsy_dir) : os.makedirs(alsy_dir)
    if not os.path.exists(tsne_dir) : os.makedirs(tsne_dir)
    if not os.path.exists(eval_dir) :
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    resfile = None
    if hps.save_label:
        log_dir = os.path.join(test_dir, hps.cache_dir.split("/")[-1])
        resfile = open(log_dir, "w")
        logger.info("[INFO] Write the Evaluation into %s", log_dir)

    model = load_test_model(model, model_name, eval_dir, hps.save_root, hps.save_dir_name, hps.device)
    model.eval()

    predictions, labels, wlabels, wclabels, sentcefrlabels, speaker_id_labels = [], [], [], [], [], []
    if hps.tsne:
        sembeds, wembeds, pembeds, sentcefrrtn = [], [], [], []
    iter_start_time=time.time()
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")

        for i, data in enumerate(loader):

            # data to gpu
            if hps.cuda:
                # data = {k: v if k in ['bert_sent_boundary', 'bert_feature', 'ie_count', 'ir_count', 'label', 'speaker_id'] else v.to(hps.device) for k, v in data.items()}
                label = torch.cat(data.get('label'), dim=0).reshape(-1, 1).to(hps.device)

                # except:
                #     G = G.to('cuda')
                #     if G_c:
                #         G_c.to('cuda')
                #     if G_wp:
                #         G_wp.to('cuda')
                #     if G_sp:
                #         G_sp.to('cuda')
                #     if G_itvr:
                #         G_itvr.to('cuda')
                #     if G_P_itvr:
                #         G_P_itvr.to('cuda')

            if hps.tsne:
                p2w_embed = []

            if hps.problem_type == 'classification':

                output = model.forward(data)  # [n_snodes, 6]
                w_outputs = output['results']['w']
                outputs = output['results']['s']
                p_outputs = output['results']['p']
                word_feature = output['embed']['after_gat']['w']
                sent_states = output['embed']['after_gat']['s']
                paragraph_feature = output['embed']['after_gat']['p']
                outputs = p_outputs if hps.sentaspara == 'sent' else s_outputs if hps.sentaspara == 'para' else s_outputs
                
                pnode_id = G_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                wnode_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
                wid = G.nodes[wnode_id].data["id"]
                label = torch.argmax(G.ndata["label"][snode_id] if hps.sentaspara == 'para' else G_sp.ndata["label"][pnode_id], dim=1)

                if hps.eval_speaker_wise:
                    predictions.append(np.mean(torch.argmax(outputs, dim=1).cpu().numpy()).item())
                    labels.append(label.tolist()[0])
                    wlabels.extend(wid.cpu().tolist())
                    speaker_id_labels.append(speaker_id)
                else:
                    predictions.extend(torch.argmax(outputs, dim=1).tolist())
                    labels.extend(label.tolist())
                    wlabels.extend(wid.cpu().tolist())

            if hps.problem_type == 'regression':
                
                output = model.forward(data)  # [n_snodes, 6]
                w_outputs = output['results']['w']
                outputs = output['results']['s']
                p_outputs = output['results']['p']
                word_feature = output['embed']['after_gat']['w']
                sent_states = output['embed']['after_gat']['s']
                paragraph_feature = output['embed']['after_gat']['p']
                outputs = p_outputs if hps.sentaspara == 'sent' else s_outputs if hps.sentaspara == 'para' else s_outputs

                G = data.get('G')
                # pnode_id = G_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                wnode_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
                wid = G.nodes[wnode_id].data["id"]
                words = [vocab.id2word(wd.item()) for wd in wid]
                # w_cefr_labels = torch.FloatTensor([np.mean(np.array([hps.cefr_loader.get_CEFR2INT_w_zero()[c] for c in hps.cefr_loader.get_cefr_tags(w)])).item() for w in words]).to(hps.device)
                # label = torch.argmax(G.ndata["label"][snode_id] if hps.sentaspara == 'para' else G_sp.ndata["label"][pnode_id], dim=1)

                speaker_id = data.get('speaker_id')

                if hps.wcefr:
                    scnode_id = G_c.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
                    label_cefr_index = G_c.ndata["label"][scnode_id]
                    dec_lens   = output['embed']['dec_outputs']['lens']
                    dec_output = output['embed']['dec_outputs']['wcefr']
                    dec_output = torch.cat([t.unsqueeze(0) for t in dec_output], dim=0).transpose(0, 1)
                    for i, l in enumerate(dec_lens):
                        get_pred_output = torch.argmax(dec_output[i][:l], dim=1)
                        get_pred_label  = label_cefr_index[i][:l][:max(dec_lens)]
                        sentcefrrtn.append(dec_output[i][:l])
                        sentcefrlabels.append(get_pred_label.cpu())

                if hps.tsne:
                    # wembeds.append(word_feature.cpu() if word_feature is not None else None)
                    sembeds.append(sent_states.cpu() if sent_states is not None else None)
                    pembeds.append(paragraph_feature.cpu() if paragraph_feature is not None else None)

                a = float(outputs.reshape(-1).cpu().numpy().item())
                b = float(label.reshape(-1).tolist()[0])
                # debug
                # if abs(a-b) < 0.1:
                #     print(data.get('utt_id'))
                #     input()
                predictions.append(outputs.reshape(-1).cpu().numpy().item())
                labels.append(label.reshape(-1).tolist()[0])
                wlabels.append(wid.cpu().tolist())
                # wclabels.append(w_cefr_labels.cpu().tolist())
                speaker_id_labels.append(speaker_id[0])

    if hps.tsne:
        speaker_id_list_file_path = os.path.join(alsy_dir, '{}.speaker_id.labels'.format(hps.which_set))
        if hps.sentaspara == 'sent':
            pembeds_save_file_path = os.path.join(alsy_dir, '{}.plabels.pembeds'.format(hps.which_set))
        # sembeds_save_file_path = os.path.join(alsy_dir, '{}.slabels.sembeds'.format(hps.which_set))
        # wembeds_save_file_path = os.path.join(alsy_dir, '{}.wlabels.wembeds'.format(hps.which_set))
        # wembeds_cefr_save_file_path = os.path.join(alsy_dir, '{}.wclabels.wembeds'.format(hps.which_set))
        if hps.wcefr:
            sent_cefr_save_file_path = os.path.join(alsy_dir, '{}.sclabels.ghinit'.format(hps.which_set))
        pickleStore(speaker_id_labels, speaker_id_list_file_path)
        if hps.sentaspara == 'sent':
            pickleStore((labels, pembeds), pembeds_save_file_path)
        # pickleStore((labels, sembeds), sembeds_save_file_path)
        # pickleStore((wlabels, wembeds), wembeds_save_file_path)
        # pickleStore((wclabels, wembeds), wembeds_cefr_save_file_path)
        if hps.wcefr:
            pickleStore((sentcefrlabels, sentcefrrtn), sent_cefr_save_file_path)
        print('Paragraph Nodes Speaker ids list for t-SNE were saved at {}'.format(speaker_id_list_file_path))
        if hps.sentaspara == 'sent':
            print('Paragraph Nodes Embeddings for t-SNE were saved at {}'.format(pembeds_save_file_path))
        # print('{} Nodes Embeddings for t-SNE were saved at {}'.format('Paragraph' if hps.sentaspara == 'para' else 'Sentence', sembeds_save_file_path))
        # print('Word Nodes Embeddings for t-SNE were saved at {}'.format(wembeds_save_file_path))
        # print('Word Nodes Embeddings of CEFR for t-SNE were saved at {}'.format(wembeds_cefr_save_file_path))
        if hps.wcefr:
            print('Sentence of CEFR token predicted results and labels were saved at {}'.format(sent_cefr_save_file_path))

    predictions = torch.FloatTensor(predictions)
    labels = torch.FloatTensor(labels)

    INT2CEFR = { idx:scale for scale, idx in CEFR2INT.items() }

    if hps.problem_type == 'regression':
        labels = labels + torch.ones_like(labels)

    if hps.problem_type == 'classification':
        logger.info('[INFO] End of test | time: {:5.2f}s | acc | test micro accuracy {:5.3f} | '.format((time.time() - iter_start_time), round_num(precision_score(predictions, labels, average='micro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_acc |test macro accuracy {:5.3f} | '.format((time.time() - iter_start_time), round_num(precision_score(predictions, labels, average='macro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | rc | test micro recall {:5.3f} | '.format((time.time() - iter_start_time), round_num(recall_score(predictions, labels, average='micro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_rc | test macro recall {:5.3f} | '.format((time.time() - iter_start_time), round_num(recall_score(predictions, labels, average='macro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | f1| test micro f1 {:5.3f} | '.format((time.time() - iter_start_time), round_num(f1_score(predictions, labels, average='micro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_f1 | test macro f1 {:5.3f} | '.format((time.time() - iter_start_time), round_num(f1_score(predictions, labels, average='macro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | ur | test micro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_micro_underestimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_ur| test macro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_macro_underestimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | or | test micro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_micro_overestimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_or | test macro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_macro_overestimate_rate(predictions, labels)) ))
    elif hps.problem_type == 'regression':
        logger.info('[INFO] End of test | time: {:5.2f}s | rmse | test micro rmse {:5.3f} | '.format((time.time() - iter_start_time), round_num(np.sqrt(((predictions.cpu().numpy() - labels.cpu().numpy()) ** 2).mean())) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_rmse | test macro rmse {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_mcrmse(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | pearson | test pcc {:5.3f} | '.format((time.time() - iter_start_time), round_num(cal_pccs(predictions.cpu().numpy(), labels.cpu().numpy())) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | within_0.5 | test micro marginal accuracy 0.5 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_accuracy_within_margin(predictions, labels, 0.5)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | within_1 | test micro marginal accuracy 1.0 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_accuracy_within_margin(predictions, labels, 1)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_within_0.5 | test macro marginal accuracy 0.5 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_within_mcacc(predictions, labels, 0.5)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_within_1 | test macro marginal accuracy 1.0 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_within_mcacc(predictions, labels, 1)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | oe_rate | test micro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_over_estimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_oe_rate | test macro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_overestimate_mcrate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | ue_rate | test micro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_under_estimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_ue_rate | test macro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_underestimate_mcrate(predictions, labels)) ))

    if hps.problem_type == 'regression':
        # save metrics to json format
        json_data = {
            'rmse': float(np.sqrt(((predictions.cpu().numpy() - labels.cpu().numpy()) ** 2).mean())),
            'mc_rmse': float(_compute_mcrmse(predictions, labels)),
            'pearson': float(cal_pccs(predictions.cpu().numpy(), labels.cpu().numpy())),
            'within_0.5': float(_accuracy_within_margin(predictions, labels, 0.5)),
            'within_1': float(_accuracy_within_margin(predictions, labels, 1)),
            'mc_within_0.5': float(_compute_within_mcacc(predictions, labels, 0.5)),
            'mc_within_1': float(_compute_within_mcacc(predictions, labels, 1)),
            'oe_rate': float(_compute_over_estimate_rate(predictions, labels)),
            'mc_oe_rate': float(_compute_overestimate_mcrate(predictions, labels)),
            'ue_rate': float(_compute_under_estimate_rate(predictions, labels)),
            'mc_ue_rate': float(_compute_underestimate_mcrate(predictions, labels)),
        }
        saved_predictions = predictions.to(float).tolist()
        saved_labels = labels.to(float).tolist()

    if hps.problem_type == 'regression':
        y_pred = []
        for score_num in predictions.tolist():
            ori_list = [ num - score_num for num in list(INT2CEFR.keys()) ]
            abs_list = [ abs(num) for num in ori_list ]
            min_num = min(abs_list) # get index is zero
            if min_num >=0.0 and min_num <= 1.0:
                get_idx_num = abs_list.index(min_num)
                scale_label = list(INT2CEFR.keys())[get_idx_num]
            else:
                get_idx_num = abs_list.index(min_num)
                get_ori_num = ori_list[get_idx_num]
                if get_ori_num > 0:
                    get_idx_num += 1
                    if get_idx_num > 6:
                        get_idx_num = 6
                else:
                    get_idx_num -= 1
                    if get_idx_num < 0:
                        get_idx_num = 0
                scale_label = list(INT2CEFR.keys())[get_idx_num]
            y_pred.append(scale_label)
        predictions = torch.FloatTensor(y_pred)

    labels_keys = list(CEFR2INT.keys())
    labels_keys.remove(0)
    if hps.problem_type == 'regression':
        labels = [INT2CEFR[i] for i in labels.to(int).tolist()] # 1 - 5
        predictions = [INT2CEFR[i] for i in predictions.to(int).tolist()]
    elif hps.problem_type == 'classification':
        labels = [INT2CEFR[i] for i in (labels + torch.ones_like(labels)).to(int).tolist()] # (0 - 4) ++ 1 = (1 - 5)
        predictions = [INT2CEFR[i] for i in (predictions + torch.ones_like(predictions)).tolist()]
    cm = confusion_matrix(labels, predictions, labels=labels_keys)

    if hps.problem_type == 'regression':
        json_data['labels'] = labels
        json_data['predictions'] = predictions
        json_data['saved_predictions'] = saved_predictions
        json_data['saved_labels'] = saved_labels
        json_data['labels_keys'] = labels_keys
        json_path = os.path.join(alsy_dir, "metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    # confusion matrix - by count show count
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cbar_kws={'label': 'counts'}) #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(labels_keys)
    ax.yaxis.set_ticklabels(labels_keys)
    save_file_path = os.path.join(alsy_dir, '{}.confusion_matrix.count.sns.{}.png'.format(hps.which_set, model_name))
    fig.savefig(save_file_path, bbox_inches='tight')
    print('confusion matrix (by count show count) is saved at {}'.format(save_file_path))

    # confusion matrix - by correctness show count
    cf_matrix_dict = dict()
    for scale in labels_keys:
        cf_matrix_dict.setdefault(
            scale,
            [0] * len(labels_keys)
        )
    for (true, pred) in zip(labels, predictions):
        # debug: if pred is zero
        if pred == 0:
            continue
        scale_row = cf_matrix_dict[true]
        get_row_idx = labels_keys.index(pred)
        scale_row[get_row_idx] += 1
        cf_matrix_dict[true] = scale_row
    cf_matrix = np.array(list(cf_matrix_dict.values()))
    df_confusion = pd.DataFrame(cf_matrix, index=labels_keys, columns=labels_keys)
    df_confusion['TOTAL'] = df_confusion.sum(axis=1)
    df_confusion.loc['TOTAL']= df_confusion.sum()
    df_percentages = df_confusion.div(df_confusion.TOTAL, axis=0) # get percentages
    df_percentages.TOTAL = 0
    df_percentages.drop('TOTAL', inplace=True, axis=1) # drop col TOTAL
    df_percentages.drop('TOTAL', inplace=True, axis=0) # drop row TOTAL
    df_confusion.drop('TOTAL', inplace=True, axis=1) # drop col TOTAL
    df_confusion.drop('TOTAL', inplace=True, axis=0) # drop row TOTAL

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(data=df_percentages, annot=df_percentages, cmap='Blues', fmt=".3f", ax=ax, cbar_kws={'label': 'percentages'})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    save_file_path = os.path.join(alsy_dir, '{}.confusion_matrix.correctness.percentage.sns.{}.png'.format(hps.which_set, model_name))
    fig.savefig(save_file_path, bbox_inches='tight')
    print('confusion matrix (by correctness show correctness) is saved at {}'.format(save_file_path))
    
    # confusion matrix - by correctness show count
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(data=df_percentages, annot=df_confusion, cmap='Blues', fmt="d", ax=ax, cbar_kws={'label': 'percentages'})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    save_file_path = os.path.join(alsy_dir, '{}.confusion_matrix.correctness.count.sns.{}.png'.format(hps.which_set, model_name))
    fig.savefig(save_file_path, bbox_inches='tight')
    print('confusion matrix (by correctness show count) is saved at {}'.format(save_file_path))
    
    return alsy_dir

def write_xlsx_table(hps, log_file, alsy_dir):
    # write table
    log_content = open(log_file, 'r').readlines()
    log_content = {line.split('|')[2].strip(): line.split('|')[3].strip().split()[-1].strip() for line in log_content if '[INFO] End of test |' in line}
    writer = ExcelXlsxTableWriter()
    writer.table_name = "output"
    if hps.problem_type == 'regression':
        writer.headers = ["rmse", "mc_rmse", "pearson", "within_0.5", "within_1", "mc_within_0.5", "mc_within_1", "oe_rate", "mc_oe_rate", "ue_rate", "mc_ue_rate"]
        writer.value_matrix = [
            [str(float(log_content[k])) for k in writer.headers],
        ]
    elif hps.problem_type == 'classification':
        writer.headers = ["acc", "mc_acc", "rc", "mc_rc", "f1", "mc_f1", "ur", "mc_ur", "or", "mc_or"]
        writer.value_matrix = [
            [str(float(log_content[k])) for k in writer.headers],
        ]
    logger.info('[INFO] Save report.xlsx file at {}'.format(os.path.join(alsy_dir, 'report.{}.xlsx'.format(hps.which_set))))
    writer.dump(os.path.join(alsy_dir, 'report.{}.xlsx'.format(hps.which_set)))

def main():

    args = get_eval_args(aux_objs)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "{}_combo.{}.label.jsonl".format(args.which_set, args.sentaspara))
    ABREAST_FILE = os.path.join(args.data_dir, "{}_combo.abreast.txt".format(args.which_set))
    RELATION_FILE = os.path.join(args.data_dir, "{}_combo.relation".format(args.which_set))
    RELATION_DATABASE_PATH = os.path.join(args.data_dir, "relation_database.json")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab.combine" if args.interviewer else 'vocab')
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.{}.txt".format(args.sentaspara))
    LOG_PATH = args.log_root
    INTERVIEWER_DATA_FILE, BERT_DATA_PATH = None, None
    if args.interviewer:
        INTERVIEWER_DATA_FILE = os.path.join(args.data_dir, "{}_combo.{}.interviewer.label.jsonl".format(args.which_set, args.sentaspara))
        logger.info("[INFO] Use interviewer's information")
    if args.bert:
        BERT_DATA_PATH = os.path.join(args.data_dir, "{}_combo.{}.combine.label.jsonl".format(args.which_set, args.sentaspara))
    OIE_DATA_PATH = os.path.join(args.data_dir, "{}_combo.{}.oie.label.jsonl".format(args.which_set, args.sentaspara))

    # CEFR node and Filled Pauses node
    cefr_loader, filled_pauses_loader = None, None
    if args.cefr_word:
        VOCABPROFILE_FILE = os.path.join(args.data_dir, 'cefrj1.6_c1c2.final.txt')
        cefr_loader = CEFREmbed(args.word_emb_dim, VOCABPROFILE_FILE)
    if args.filled_pauses_word and (args.cefr_info == 'embed_init'):
        FLUENCYPAUSE_FILE = os.path.join(args.data_dir, 'all.filled_pauses.txt')
        filled_pauses_loader = FILLEDEmbed(args.word_emb_dim, FLUENCYPAUSE_FILE)

    # train_log setting
    if not os.path.exists(LOG_PATH):
        logger.exception("[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
        raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    log_path = os.path.join(LOG_PATH, "do_test_log")
    file_handler = logging.FileHandler(log_path, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = None
        if (cefr_loader or filled_pauses_loader):
            if args.cefr_info == 'embed_init':
                vectors = embed_loader.load_my_vecs_with_infos(cefr_loader, filled_pauses_loader, k=args.word_emb_dim)
            else:
                vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        else:
            vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train
        logger.info("[INFO] Use GLOVE to get the embeddings of both words and paragraph")
    else:
        embed.weight.requires_grad = args.embed_train
        logger.info("[INFO] Use random initial embeddings of both words and paragraph")

    hps = args
    logger.info(hps)
    
    # something do after hps stdout
    hps.cefr_loader = cefr_loader

    tokenizer = None
    args.bert_config = None
    if args.bert:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path,
                                                  is_split_into_words=True)
        bert_config = AutoConfig.from_pretrained(args.bert_model_path, output_hidden_states=True)
        auxiliary_inputs_dict = auxiliary_inputs(args, tokenizer, aux_objs)
        bert_config.auxiliary_inputs_dict = auxiliary_inputs_dict
        args.bert_config = bert_config

        if args.bert_config.model_type in ['roberta', 'longformer'] and args.bert_roberta_to_long:
            args.tokenizer = tokenizer

    test_w2s_path = os.path.join(args.cache_dir, "{}_combo.{}.w2s.tfidf.jsonl".format(args.which_set, args.sentaspara))

    if hps.model == "HSG":
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
        dataset = ExampleSet(DATA_FILE, INTERVIEWER_DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, hps.pmi_window_width, tokenizer, hps)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    elif hps.model == "HSPG":
        model = HSumPromptGraph(hps, embed)
        logger.info("[MODEL] HSumPromptGraph ")
        dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, test_w2s_path, hps.pmi_window_width, tokenizer, hps)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    elif hps.model == "HSSG":
        model = HSentPromptGraph(hps, embed)
        logger.info("[MODEL] HSentPromptGraph ")
        dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, test_w2s_path, hps.pmi_window_width, tokenizer, hps)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    elif hps.model == "HSDG":
        model = HierConGraph(hps, embed)
        logger.info("[MODEL] HierConGraph ")
        dataset = ExampleHierSet(DATA_FILE, INTERVIEWER_DATA_FILE, RELATION_DATABASE_PATH, RELATION_FILE, ABREAST_FILE, BERT_DATA_PATH, OIE_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, hps.pmi_window_width, tokenizer, hps)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    elif hps.model == "HDSG":
        model = HSumDocGraph(hps, embed)
        logger.info("[MODEL] HeterDocSumGraph ")
        test_w2d_path = os.path.join(args.cache_dir, "eval_combo.{}.w2s.tfidf.jsonl".format(args.sentaspara))
        dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, test_w2d_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    elif hps.model == "PAL":
        model = PromptAwareLSTM(hps, embed)
        logger.info("[MODEL] PromptAwareLSTM ")
        dataset = ExampleHierSet(DATA_FILE, INTERVIEWER_DATA_FILE, RELATION_DATABASE_PATH, RELATION_FILE, ABREAST_FILE, BERT_DATA_PATH, OIE_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, hps.pmi_window_width, tokenizer, hps)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    hps.device = torch.device("cpu")
    if args.cuda:
        hps.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        logger.info("[INFO] Use cuda")

    # try:
    model = model.to(hps.device)
    # except:
    #     model = model.cuda()

    # get recipe name
    hps = get_recipe_name(hps)

    # debug
    if hps.debug:
        print('hps.save_dir_name: ', hps.save_dir_name)
        return None
    # debug: only print name


    logger.info("[INFO] Decoding...")
    if hps.test_model == "multi":
        spath = os.path.join(hps.save_root, hps.save_dir_name, "eval", "bestmodel_4")
        for model_name in sorted(glob.glob(spath)):
            model_name = "eval" + os.path.basename(model_name)
            run_test(model, dataset, loader, model_name, hps, vocab)
    else:
        alsy_dir = run_test(model, dataset, loader, hps.test_model, hps, vocab)
        # model_name = os.path.join(hps.save_root, hps.save_dir_name, "eval", "bestmodel_4")
        # alsy_dir = run_test(model, dataset, loader, model_name, hps, vocab)
        write_xlsx_table(hps, log_path, alsy_dir)

if __name__ == '__main__':
    main()
