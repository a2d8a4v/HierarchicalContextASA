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
import sys
import shutil
import time
import math
import random

import wandb

import dgl
import numpy as np
import torch

from HiGraph import HSumGraph, HSentPromptGraph, HSumPromptGraph, HSumDocGraph
from HierConGraph import HierConGraph
from PromptAwareLSTM import PromptAwareLSTM

from module.dataloader import ExampleSet, ExampleHierSet, ExamplePromptSet, MultiExampleSet, graph_collate_fn
import module.bert_data as bert_data 
from module.embedding import Word_Embedding
from module.INFOembedding import CEFREmbed, FILLEDEmbed
from module.vocabulary import Vocab
from tools.args import get_train_args
from tools.recipe_name import get_recipe_name
from tools.logger import *
from tools.utils import CEFR2INT, INT2CEFR, make_up_weights, make_up_wcefr_weights, weight_groups, POS2INT, DEP2INT, GED2INT
# from helpers.loss import Perplexity
from helpers.BalancedMSE import BMCLoss, GAILoss
from torch.nn import NLLLoss
from helpers.OrdinalEntropy import ordinal_entropy, ordinal_entropy_tightness

import deepspeed
from tools.distri import init_dataloader, torch_distributed_master_process_first
from transformers import AutoTokenizer, AutoConfig

from sklearn.metrics import precision_score, recall_score, f1_score
from tools.utils import (
    AverageMeter,
    read_tsv,
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
    _compute_underestimate_mcrate
)

_DEBUG_FLAG_ = False

aux_objs = ['pos', 'deprel',
            'gra_sim', 'gra_med', 'gra_dif',
            'dif_sim', 'dif_med',
            'laugh', 'sil', 'w_cefr']

def precompute_loss_weights(train_levels, max_score, epsilon=1e-5, alpha=2e-1):
    train_sentlv_ratio = np.array([np.sum(train_levels == float(lv)) for lv in range(0, int(max_score))])
    train_sentlv_ratio = train_sentlv_ratio / np.sum(train_sentlv_ratio)
    train_sentlv_weights = np.power(train_sentlv_ratio, alpha) / np.sum(
        np.power(train_sentlv_ratio, alpha)) / (train_sentlv_ratio + epsilon)
    return torch.Tensor(train_sentlv_weights)


def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file)


def auxiliary_inputs(args, tokenizer, aux_objs):
    auxiliary_inputs_dict = dict()
    for obj in aux_objs:
        if getattr(args, 'use_{}'.format(obj)):
            num_predictions = tokenizer.vocab_size if obj in ['mlm', 'lm'] else len(getattr(bert_data, 'get_{}_labels'.format(obj))())
            auxiliary_inputs_dict[obj] = num_predictions
    return auxiliary_inputs_dict


def setup_training(model, train_loader, valid_loader, valset, hps, vocab):
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param vocab: vocab for model
        :return: 
    """

    train_dir = os.path.join(hps.save_root, hps.save_dir_name, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir): shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, train_loader, valid_loader, valset, hps, train_dir, vocab)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


def calculate_local_metrics(predictions, labels, hps, mode):
    assert mode in ['trn', 'dev', 'eval']
    metrics = {}
    if hps.problem_type == 'classification':
        metrics['acc'] = precision_score(predictions, labels, average='micro')
        metrics['mc_acc'] = precision_score(predictions, labels, average='macro')
        metrics['rc'] = recall_score(predictions, labels, average='micro')
        metrics['mc_rc'] = recall_score(predictions, labels, average='macro')
        metrics['f1'] = f1_score(predictions, labels, average='micro')
        metrics['mc_f1'] = f1_score(predictions, labels, average='macro')
        metrics['ur'] = compute_micro_underestimate_rate(predictions, labels)
        metrics['mc_ur'] = compute_macro_underestimate_rate(predictions, labels)
        metrics['or'] = compute_micro_overestimate_rate(predictions, labels)
        metrics['mc_or'] = compute_macro_overestimate_rate(predictions, labels)
    elif hps.problem_type == 'regression':
        metrics['rmse'] = np.sqrt(((predictions.cpu().numpy() - labels.cpu().numpy()) ** 2).mean())
        metrics['mc_rmse'] = _compute_mcrmse(predictions, labels)
        metrics['pearson'] = cal_pccs(predictions.cpu().numpy(), labels.cpu().numpy())
        metrics['within_0.5'] = _accuracy_within_margin(predictions, labels, 0.5)
        metrics['within_1'] = _accuracy_within_margin(predictions, labels, 1)
        metrics['mc_within_0.5'] = _compute_within_mcacc(predictions, labels, 0.5)
        metrics['mc_within_1'] = _compute_within_mcacc(predictions, labels, 1)
        metrics['oe_rate'] = _compute_over_estimate_rate(predictions, labels)
        metrics['mc_oe_rate'] = _compute_overestimate_mcrate(predictions, labels)
        metrics['ue_rate'] = _compute_under_estimate_rate(predictions, labels)
        metrics['mc_ue_rate'] = _compute_underestimate_mcrate(predictions, labels)
    metrics = {'{}_{}'.format(mode, k): v for k, v in metrics.items()}
    return metrics

def setup_device(args, logger):
    if torch.cuda.is_available():
        # if torch.distributed.is_initialized() and args.multiple_device:
        #     device = torch.device("cuda", args.local_rank)
        # else:
        device = torch.device("cuda", int(args.gpu))
        # device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logger.info("[INFO] Use cuda")
    return device

def run_training(model, train_loader, valid_loader, valset, hps, train_dir, vocab):
    '''  Repeatedly runs training iterations, logging loss to screen and log files
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param train_dir: where to save checkpoints
        :return: 
    '''
    logger.info("[INFO] Starting run_training")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

    # loss reweight
    reweight = None
    if hps.reweight:
        labels = []
        get_labels = train_loader.dataset.get_labels_only()
        for label in get_labels:
            label = label[0].tolist().index(1)
            labels.append(label)
        labels = np.array(labels)
        if hps.problem_type == 'classfication':
            reweight = precompute_loss_weights(labels, 6, alpha=hps.rw_alpha) # it will give the weight of each class
        elif hps.problem_type == 'regression':
            labels = (labels + np.ones_like(labels.shape)).tolist()
            reweight = {i: (labels.count(i)/len(labels)) ** hps.rw_alpha for i in range(0, 7)}
    if hps.imp:
        noise_var = AverageMeter()

    criterions_dict = {
        'main': torch.nn.CrossEntropyLoss(weight=reweight, reduction='none', ignore_index=-1) if hps.problem_type == 'classfication' else torch.nn.MSELoss(reduction='none')
    }
    if hps.problem_type == 'regression':
        if hps.imp == 'bmc':
            if not reweight:
                criterions_dict['main'] = BMCLoss(hps.init_noise_sigma, hps.device)
            else:
                criterions_dict['bmc'] = BMCLoss(hps.init_noise_sigma, hps.device)
        elif hps.imp == 'gai':
            if not reweight:
                criterions_dict['main'] = GAILoss(hps.init_noise_sigma, hps.gmm, hps.device)
            else:
                criterions_dict['gai'] = GAILoss(hps.init_noise_sigma, hps.gmm, hps.device)
    if hps.wcefr:
        # criterions_dict['dec_wcefr'] = Perplexity(torch.ones(8), vocab.get_pad_id())
        criterions_dict['dec_wcefr'] = NLLLoss(ignore_index=0)

    if hps.cuda:
        criterions_dict = {k: v.to(hps.device) if k in ['main', 'dec_wcefr'] else v for k, v in criterions_dict.items()}

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0
    Lambda_d = hps.oe_weight

    # torch.cuda.empty_cache() # out of memory
    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        trn_pred, trn_lab = [], []
        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):

            # start
            iter_start_time = time.time()
            model.train()

            # data to gpu
            if hps.cuda:
                label = torch.cat(data.get('label'), dim=0).reshape(-1, 1).to(hps.device)

            if hps.problem_type == 'regression':
                output = model.forward(data)  # [n_snodes, 6]
                # ie_count = output['record']['ie']
                w_outputs = output['results']['w']
                s_outputs = output['results']['s']
                p_outputs = output['results']['p']
                word_feature = output['embed']['after_gat']['w']
                sent_states = output['embed']['after_gat']['s']
                paragraph_feature = output['embed']['after_gat']['p']
                outputs = p_outputs if hps.sentaspara == 'sent' else s_outputs if hps.sentaspara == 'para' else s_outputs
                
                G = data.get('G')
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1).to(hps.device)
                # pnode_id = G_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
                
                # if hps.wcefr:
                #     wnode_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
                #     wid = G.nodes[wnode_id].data["id"].detach().cpu().tolist()
                #     words = [vocab.id2word(wd) for wd in wid]
                #     w_cefr_labels = torch.FloatTensor([np.mean(np.array([hps.cefr_loader.get_CEFR2INT_w_zero()[c] for c in hps.cefr_loader.get_cefr_tags(w)])).item() for w in words]).to(hps.device)
                #     get_zero_row_idxs = (w_cefr_labels == 0.).nonzero(as_tuple=False)
                #     adjust_loss = torch.ones_like(w_cefr_labels).to(hps.device)
                #     adjust_loss[get_zero_row_idxs] = 0.

                # label = torch.argmax(G.ndata["label"][snode_id] if hps.sentaspara == 'para' else G_sp.ndata["label"][pnode_id], dim=1).unsqueeze(-1).to(torch.float32)
                # try:
                label = label + torch.ones_like(label).to(hps.device)
                # except:
                #     label = label + torch.ones_like(label).cuda()
                outputs = outputs.squeeze(-1)
                label = label.squeeze(-1)
                loss_value = criterions_dict['main'](outputs, label)  ## [batch_size]

                if hps.oe:
                    loss_oe = ordinal_entropy_tightness(sent_states if hps.sentaspara == 'para' else paragraph_feature, label.unsqueeze(-1)) * Lambda_d
                    loss_value = loss_value + loss_oe

                if hps.reweight:
                    loss_value = loss_value * make_up_weights(reweight, label.to(int), hps)

                if hps.imp == 'bmc':
                    if reweight:
                        bmc_loss_value = criterions_dict['bmc'](outputs, label)  ## [batch_size]
                        te = 1 / (1+math.exp(1e-6*(hps.n_epochs/2-epoch)))
                        loss_value = te*loss_value + (1-te)*bmc_loss_value
                        criterions_dict['epoch'] = epoch

                if hps.wcefr:
                    # w_cefr_labels = w_cefr_labels.unsqueeze(-1)
                    # w_loss_value = w_criterion(w_outputs, w_cefr_labels)
                    # if hps.wcefr_reweight:
                    #     w_loss_value = w_loss_value * make_up_wcefr_weights(hps.wc_r, w_cefr_labels.squeeze(-1), hps)
                    # if hps.oe:
                    #     w_loss_oe = ordinal_entropy(word_feature, w_cefr_labels) * Lambda_d
                    #     w_loss_value = w_loss_value + w_loss_oe
                    # w_loss_value = w_loss_value * adjust_loss
                    scnode_id = G_c.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
                    label_cefr_index = G_c.ndata["label"][scnode_id]
                    dec_lens   = output['embed']['dec_outputs']['lens']
                    dec_output = output['embed']['dec_outputs']['wcefr']
                    dec_output = torch.cat([t.unsqueeze(0) for t in dec_output], dim=0).transpose(0, 1)
                    # print('dec_lens: ', len(dec_lens))
                    # print('dec_output: ', dec_output[0].shape)
                    # dec_output = dec_output.reshape(-1, dec_output.shape[-1])
                    # label_cefr_index = label_cefr_index.view(-1)
                    w_loss_value = []
                    for i, l in enumerate(dec_lens):
                        sent_w_loss_value = dec_criterion(dec_output[i], label_cefr_index[i][:max(dec_lens)])
                        w_loss_value.append(sent_w_loss_value)
                    try:
                        w_loss_value = torch.tensor(w_loss_value).to(hps.device)
                    except:
                        w_loss_value = torch.tensor(w_loss_value).cuda()
                    # w_loss_value = dec_criterion(dec_output, label_cefr_index).unsqueeze(0)

                if hps.train_speaker_wise:
                    loss_value = loss_value * weight_groups(snode_id, hps)

                if hps.gradient_accumulation_steps:
                    loss_value = loss_value / hps.gradient_accumulation_steps

                # bug: fix loss value size
                if len(loss_value.shape) == 2:
                    loss_value = loss_value.squeeze(0)

                if hps.sentaspara == 'para':
                    G.nodes[snode_id].data["loss"] = loss_value  # [n_nodes, 1]
                    loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                else:
                    # G_sp.nodes[pnode_id].data["loss"] = loss_value
                    # loss = dgl.sum_nodes(G_sp, "loss")
                    loss = loss_value

                # start_id_count = 0
                # for k, l in enumerate(ie_count):
                #     chunk_snode_start_id = start_id_count
                #     chunk_snode_end_id = chunk_snode_start_id + l -1
                #     start_id_count = chunk_snode_end_id + 1
                #     chunk_snode_id = torch.arange(chunk_snode_start_id, chunk_snode_end_id + 1)
                #     G.nodes[chunk_snode_id].data["loss"] = loss_value[k].repeat(l, 1)  # [n_nodes, 1]
                # loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]

                if hps.wcefr:
                    G_c.nodes[scnode_id].data["scloss"] = w_loss_value

                loss = loss.mean()

                # add w loss
                if hps.wcefr:
                    wloss = dgl.sum_nodes(G_c, "scloss")  # [batch_size, 1]
                    wloss = wloss.mean()
                    loss = loss + wloss

                trn_pred.extend(outputs.tolist())
                trn_lab.extend(label.tolist())
            elif hps.problem_type == 'classification':
                if hps.oe:
                    raise NotImplementedError('Not yet!')
                output = model.forward(data)  # [n_snodes, 6]
                w_outputs = output['results']['w']
                outputs = output['results']['s']
                p_outputs = output['results']['p']
                word_feature = output['embed']['after_gat']['w']
                sent_states = output['embed']['after_gat']['s']
                paragraph_feature = output['embed']['after_gat']['p']
                outputs = p_outputs if hps.sentaspara == 'sent' else s_outputs if hps.sentaspara == 'para' else s_outputs
                
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                pnode_id = G_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
                label = G.ndata["label"][snode_id].float() if hps.sentaspara == 'para' else G_sp.ndata["label"][pnode_id].float()
                loss_value = criterions_dict['main'](outputs, label).unsqueeze(-1)
                if hps.gradient_accumulation_steps:
                    loss_value = loss_value / hps.gradient_accumulation_steps
                # if hps.sentaspara == 'para':
                #     G.nodes[snode_id].data["loss"] = loss_value   # [n_nodes, 1]
                #     loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                # else:
                #     G_sp.nodes[pnode_id].data["loss"] = loss_value
                #     loss = dgl.sum_nodes(G_sp, "loss")
                loss = loss_value.mean()
                # loss = loss.mean()
                trn_pred.extend(torch.argmax(outputs, dim=1).tolist())
                trn_lab.extend(torch.argmax(label, dim=1).tolist())

            if not (np.isfinite(loss.data.cpu())).numpy():
                logger.error("train Loss is not finite. Stopping.")
                logger.info(loss)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                        # logger.info(param.grad.data.sum())
                raise Exception("train Loss is not finite. Stopping.")

            if hps.imp == 'bmc':
                if not reweight:
                    noise_var.update(criterions_dict['main'].noise_sigma.item() ** 2)
                else:
                    noise_var.update(criterions_dict['bmc'].noise_sigma.item() ** 2)
            elif hps.imp == 'gai':
                if not reweight:
                    noise_var.update(criterions_dict['main'].noise_sigma.item() ** 2)
                else:
                    noise_var.update(criterions_dict['gai'].noise_sigma.item() ** 2)
                
            loss.backward()
            # torch.cuda.empty_cache() # out of memory
            if hps.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)

            if (i + 1) % hps.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i % 1 == 0:
                if _DEBUG_FLAG_:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                                .format(i, (time.time() - iter_start_time),float(train_loss / 100)))
                train_loss = 0.0

        if hps.lr_descent:
            new_lr = max(5e-6, hps.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        if torch.distributed.is_initialized():
            with torch_distributed_master_process_first(torch.distributed.get_rank()):
                trn_pred, trn_lab = torch.FloatTensor(trn_pred), torch.FloatTensor(trn_lab)
                std_metric = calculate_local_metrics(trn_pred, trn_lab, hps, 'trn')
                if (torch.distributed.get_rank() == 0) and hps.stdout_metric:
                    print(std_metric)
                if hps.wandb:
                    wandb_info = {
                        'train_loss': train_loss,
                        'epoch_loss': epoch_loss,
                        'lr': list(optimizer.param_groups)[0]['lr']
                    }
                    wandb_info.update(std_metric)
                    wandb.log(wandb_info)
        else:
            trn_pred, trn_lab = torch.FloatTensor(trn_pred), torch.FloatTensor(trn_lab)
            std_metric = calculate_local_metrics(trn_pred, trn_lab, hps, 'trn')
            if hps.stdout_metric:
                print(std_metric)
            if hps.wandb:
                wandb_info = {
                    'train_loss': train_loss,
                    'epoch_loss': epoch_loss,
                    'lr': list(optimizer.param_groups)[0]['lr']
                }
                wandb_info.update(std_metric)
                wandb.log(wandb_info)            

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss
        
        # MAYBE try several times, not stop training while encounter temporary bad result
        # elif epoch_avg_loss >= best_train_loss:
        #     logger.error("[Error] training loss does not descent. Stopping supervisor...")
        #     save_model(model, os.path.join(train_dir, "earlystop"))
        #     sys.exit(1)

        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo, criterions_dict, reweight, vocab)

        if non_descent_cnt >= hps.non_descent_count:
            
            if hps.wandb:
                wandb.finish()
            
            logger.error("[Error] val loss does not descent for {} times. Stopping supervisor...".format(hps.non_descent_count))
            save_model(model, os.path.join(train_dir, "earlystop"))
            return


def run_eval(model, loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo, criterions_dict, reweight, vocab):
    ''' 
        Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.
        :param model: the model
        :param loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param best_loss: best valid loss so far
        :param best_F: best valid F so far
        :param non_descent_cnt: the number of non descent epoch (for early stop)
        :param saveNo: the number of saved models (always keep best saveNo checkpoints)
        :return: 
    '''
    logger.info("[INFO] Starting eval for this model ...")
    eval_dir = os.path.join(hps.save_root, hps.save_dir_name, "eval")  # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)
    Lambda_d = hps.oe_weight

    model.eval()

    with torch.no_grad():

        dev_pred, dev_lab = [], []
        running_loss, batch_number = 0., 0

        for i, data in enumerate(loader):

            # data to gpu
            if hps.cuda:
                label = torch.cat(data.get('label'), dim=0).reshape(-1, 1).to(hps.device)

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
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1).to(hps.device)
                # pnode_id = G_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
                # label = torch.argmax(G.ndata["label"][snode_id] if hps.sentaspara == 'para' else G_sp.ndata["label"][pnode_id], dim=1).unsqueeze(-1).to(torch.float32)
                label = label + torch.ones_like(label).to(hps.device)
                
                # if hps.wcefr:
                #     wnode_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
                #     wid = G.nodes[wnode_id].data["id"].detach().cpu().tolist()
                #     words = [vocab.id2word(wd) for wd in wid]
                #     w_cefr_labels = torch.FloatTensor([np.mean(np.array([hps.cefr_loader.get_CEFR2INT_w_zero()[c] for c in hps.cefr_loader.get_cefr_tags(w)])).item() for w in words]).to(hps.device)
                #     get_zero_row_idxs = (w_cefr_labels == 0.).nonzero(as_tuple=False)
                #     adjust_loss = torch.ones_like(w_cefr_labels).to(hps.device)
                #     adjust_loss[get_zero_row_idxs] = 0.
                
                outputs = outputs.squeeze(-1)
                label = label.squeeze(-1)
                loss = criterions_dict['main'](outputs, label)
                loss = loss / torch.distributed.get_world_size() if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 else loss

                if hps.oe:
                    loss_oe = ordinal_entropy_tightness(sent_states if hps.sentaspara == 'para' else paragraph_feature, label.unsqueeze(1), debug=True) * Lambda_d
                    loss = loss + loss_oe

                if hps.reweight:
                    loss = loss * make_up_weights(reweight, label.to(int), hps)
                    
                if hps.imp == 'bmc':
                    if reweight:
                        bmc_loss = criterions_dict['bmc'](outputs, label)  ## [batch_size]
                        te = 1 / (1+math.exp(1e-6*(hps.n_epochs/2-criterions_dict['epoch'])))
                        loss = te*loss + (1-te)*bmc_loss

                if hps.wcefr:
                    # w_cefr_labels = w_cefr_labels.unsqueeze(-1)
                    # w_loss_value = w_criterion(w_outputs, w_cefr_labels)
                    # if hps.wcefr_reweight:
                    #     w_loss_value = w_loss_value * make_up_wcefr_weights(hps.wc_r, w_cefr_labels.squeeze(-1), hps)
                    # if hps.oe:
                    #     w_loss_oe = ordinal_entropy(word_feature, w_cefr_labels) * Lambda_d
                    #     w_loss_value = w_loss_value + w_loss_oe
                    # w_loss_value = w_loss_value * adjust_loss
                    scnode_id = G_c.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
                    label_cefr_index = G_c.ndata["label"][scnode_id]
                    dec_lens   = output['embed']['dec_outputs']['lens']
                    dec_output = output['embed']['dec_outputs']['wcefr']
                    dec_output = torch.cat([t.unsqueeze(0) for t in dec_output], dim=0).transpose(0, 1)
                    # dec_output = dec_output.reshape(-1, dec_output.shape[-1])
                    # label_cefr_index = label_cefr_index.view(-1)
                    w_loss_value = []
                    for i, l in enumerate(dec_lens):
                        # dec_criterion(dec_output[i][:l], label_cefr_index[i][:l])
                        sent_w_loss_value = dec_criterion(dec_output[i], label_cefr_index[i][:max(dec_lens)])
                        w_loss_value.append(sent_w_loss_value)
                    w_loss_value = torch.tensor(w_loss_value).to(hps.device)
                    w_loss = w_loss_value.mean()

                if hps.train_speaker_wise:
                    loss = loss * weight_groups(snode_id, hps)

                loss = loss.mean()
                # combine s_loss and w_loss
                if hps.wcefr:
                    loss = loss + w_loss
                running_loss += float(loss.data)
                batch_number += 1
                dev_pred.extend(outputs.tolist())
                dev_lab.extend(label.tolist())
            elif hps.problem_type == 'classification':
                output = model.forward(data)  # [n_snodes, 6]
                w_outputs = output['results']['w']
                outputs = output['results']['s']
                p_outputs = output['results']['p']
                word_feature = output['embed']['after_gat']['w']
                sent_states = output['embed']['after_gat']['s']
                paragraph_feature = output['embed']['after_gat']['p']
                outputs = p_outputs if hps.sentaspara == 'sent' else s_outputs if hps.sentaspara == 'para' else s_outputs
                
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                pnode_id = G_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
                label = G.ndata["label"][snode_id].float() if hps.sentaspara == 'para' else G_sp.ndata["label"][pnode_id].float()
                loss = criterions_dict['main'](outputs, label).unsqueeze(-1)  # [n_nodes, 1]
                loss = loss / torch.distributed.get_world_size() if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 else loss
                if hps.sentaspara == 'para':
                    G.nodes[snode_id].data["loss"] = loss
                    loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                else:
                    G_sp.nodes[pnode_id].data["loss"] = loss
                    loss = dgl.sum_nodes(G_sp, "loss")
                loss = loss.mean()
                running_loss += float(loss.data)
                batch_number += 1
                dev_pred.extend(torch.argmax(outputs, dim=1).tolist())
                dev_lab.extend(torch.argmax(label, dim=1).tolist())

    running_avg_loss = running_loss / batch_number
    
    if hps.wandb:
        dev_pred, dev_lab = torch.FloatTensor(dev_pred), torch.FloatTensor(dev_lab)
        wandb_info = {
            "valid_loss": running_avg_loss
        }
        wandb_info.update(calculate_local_metrics(dev_pred, dev_lab, hps, 'dev'))
        wandb.log(wandb_info)

    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % saveNo)  # this is where checkpoints of best models are saved
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    return best_loss, None, non_descent_cnt, saveNo


def fix_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if not args.use_amp:
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(cmd_args):

    args = get_train_args(aux_objs)

    # set the seed
    fix_seed(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "trn_combo.{}.label.jsonl".format(args.sentaspara))
    VALID_FILE = os.path.join(args.data_dir, "dev_combo.{}.label.jsonl".format(args.sentaspara))
    ABREAST_FILE = os.path.join(args.data_dir, "trn_combo.abreast.txt")
    ABREAST_VALID_FILE = os.path.join(args.data_dir, "dev_combo.abreast.txt")
    RELATION_FILE = os.path.join(args.data_dir, "trn_combo.relation")
    VALID_RELATION_FILE = os.path.join(args.data_dir, "dev_combo.relation")
    RELATION_DATABASE_PATH = os.path.join(args.data_dir, "relation_database.json")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab.combine" if args.interviewer else 'vocab')
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.{}.txt".format(args.sentaspara))
    LOG_PATH = args.log_root
    INTERVIEWER_DATA_FILE, INTERVIEWER_VALID_FILE, BERT_DATA_PATH, BERT_VALID_FILE = None, None, None, None
    if args.interviewer:
        INTERVIEWER_DATA_FILE  = os.path.join(args.data_dir, "trn_combo.{}.interviewer.label.jsonl".format(args.sentaspara))
        INTERVIEWER_VALID_FILE = os.path.join(args.data_dir, "dev_combo.{}.interviewer.label.jsonl".format(args.sentaspara))
        logger.info("[INFO] Use interviewer's information")
    if args.bert:
        BERT_DATA_PATH = os.path.join(args.data_dir, "trn_combo.{}.combine.label.jsonl".format(args.sentaspara))
        BERT_VALID_FILE = os.path.join(args.data_dir, "dev_combo.{}.combine.label.jsonl".format(args.sentaspara))
    OIE_DATA_PATH = os.path.join(args.data_dir, "trn_combo.{}.oie.label.jsonl".format(args.sentaspara))
    OIE_VALID_FILE = os.path.join(args.data_dir, "dev_combo.{}.oie.label.jsonl".format(args.sentaspara))

    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # CEFR node and Filled Pauses node
    cefr_loader, filled_pauses_loader = None, None
    if args.cefr_word:
        VOCABPROFILE_FILE = os.path.join(args.data_dir, 'cefrj1.6_c1c2.final.txt')
        cefr_loader = CEFREmbed(args.word_emb_dim, VOCABPROFILE_FILE)
    if args.filled_pauses_word and (args.filled_pauses_info == 'embed_init'):
        FLUENCYPAUSE_FILE = os.path.join(args.data_dir, 'all.filled_pauses.txt')
        filled_pauses_loader = FILLEDEmbed(args.word_emb_dim, FLUENCYPAUSE_FILE)

    # Word Embedding with or without Glove
    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    logger.info("[INFO] Vocab size is %s", vocab.size())
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
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

    # BERT encoder
    tokenizer = None
    args.bert_config = None
    if args.bert:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path,
                                                  is_split_into_words=True)
        bert_config = AutoConfig.from_pretrained(args.bert_model_path, output_hidden_states=True)
        auxiliary_inputs_dict = auxiliary_inputs(args, tokenizer, aux_objs)
        bert_config.auxiliary_inputs_dict = auxiliary_inputs_dict
        args.bert_config = bert_config
        logger.info("[INFO] Use BERT Encoder for paragraph")

        if args.bert_config.model_type in ['roberta', 'longformer'] and args.bert_roberta_to_long:
            args.tokenizer = tokenizer

    # Multiple GPUs
    args.multiple_device = True if len(args.gpu.split(',')) > 1 else False
    if args.multiple_device:
        deepspeed.init_distributed()
        args.local_rank = int(args.gpu.split(',')[0])
        logger.info("[INFO] Distributed Training")

    hps = args
    logger.info(hps)
    
    # something do after hps stdout
    hps.cefr_loader = cefr_loader
    if hps.cefr_word and hps.wcefr_reweight:
        VOCABPROFILE_FILE = os.path.join(args.data_dir, 'cefrj1.6_c1c2.final.txt')
        wc_cols, wc_ls = read_tsv(VOCABPROFILE_FILE)
        wc_s = wc_cols.index('score')
        wc_r = {}
        for l in wc_ls:
            wc_r.setdefault(
                int(l[wc_s]),
                []
            ).append(1)
        wc_r = {k: 1-(sum(v)/len(wc_ls)) ** hps.rw_alpha for k, v in wc_r.items()}
        wc_r[0] = 0
        hps.wc_r = wc_r

    # append pos, dep, ged
    hps.pos_vocab_size = len(POS2INT)
    hps.dep_vocab_size = len(DEP2INT)
    hps.ged_vocab_size = len(GED2INT)

    train_w2s_path = os.path.join(args.cache_dir, "trn_combo.{}.w2s.tfidf.jsonl".format(args.sentaspara))
    val_w2s_path = os.path.join(args.cache_dir, "dev_combo.{}.w2s.tfidf.jsonl".format(args.sentaspara))

    hps.device = setup_device(args, logger)
    hps.vocab_size = vocab.size()

    if args.multiple_device:
        if hps.model == "HSG":
            model = HSumGraph(hps, embed)
            logger.info("[MODEL] HeterSumGraph ")
            dataset = ExampleSet(DATA_FILE, INTERVIEWER_DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = init_dataloader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleSet(VALID_FILE, INTERVIEWER_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = init_dataloader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        elif hps.model == "HSPG":
            model = HSumPromptGraph(hps, embed)
            logger.info("[MODEL] HeterSumPromptGraph ")
            dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = init_dataloader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExamplePromptSet(VALID_FILE, INTERVIEWER_VALID_FILE, BERT_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = init_dataloader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        elif hps.model == "HSSG":
            model = HSentPromptGraph(hps, embed)
            logger.info("[MODEL] HSentPromptGraph ")
            dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = init_dataloader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExamplePromptSet(VALID_FILE, INTERVIEWER_VALID_FILE, BERT_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = init_dataloader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        elif hps.model == "HSDG":
            model = HierConGraph(hps, embed)
            logger.info("[MODEL] HierConGraph ")
            dataset = ExampleHierSet(DATA_FILE, INTERVIEWER_DATA_FILE, RELATION_DATABASE_PATH, RELATION_FILE, ABREAST_FILE, BERT_DATA_PATH, OIE_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = init_dataloader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleHierSet(VALID_FILE, INTERVIEWER_VALID_FILE, RELATION_DATABASE_PATH, VALID_RELATION_FILE, ABREAST_VALID_FILE, BERT_VALID_FILE, OIE_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = init_dataloader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        elif hps.model == "HDSG":
            model = HSumDocGraph(hps, embed)
            logger.info("[MODEL] HeterDocSumGraph ")
            train_w2d_path = os.path.join(args.cache_dir, "trn_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, train_w2d_path)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            val_w2d_path = os.path.join(args.cache_dir, "dev_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, val_w2d_path)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)  # Shuffle Must be False for ROUGE evaluation
        elif hps.model == "PAL":
            model = PromptAwareLSTM(hps, embed)
            logger.info("[MODEL] PromptAwareLSTM ")
            dataset = ExampleHierSet(DATA_FILE, INTERVIEWER_DATA_FILE, RELATION_DATABASE_PATH, RELATION_FILE, ABREAST_FILE, BERT_DATA_PATH, OIE_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = init_dataloader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleHierSet(VALID_FILE, INTERVIEWER_VALID_FILE, RELATION_DATABASE_PATH, VALID_RELATION_FILE, ABREAST_VALID_FILE, BERT_VALID_FILE, OIE_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = init_dataloader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        else:
            logger.error("[ERROR] Invalid Model Type!")
            raise NotImplementedError("Model Type has not been implemented")
    else:
        if hps.model == "HSG":
            model = HSumGraph(hps, embed)
            logger.info("[MODEL] HeterSumGraph ")
            dataset = ExampleSet(DATA_FILE, INTERVIEWER_DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleSet(VALID_FILE, INTERVIEWER_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)
        elif hps.model == "HSPG":
            model = HSumPromptGraph(hps, embed)
            logger.info("[MODEL] HSumPromptGraph ")
            dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExamplePromptSet(VALID_FILE, INTERVIEWER_VALID_FILE, BERT_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)
        elif hps.model == "HSSG":
            model = HSentPromptGraph(hps, embed)
            logger.info("[MODEL] HSentPromptGraph ")
            dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExamplePromptSet(VALID_FILE, INTERVIEWER_VALID_FILE, BERT_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, hps.passage_length, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)
        elif hps.model == "HSDG":
            model = HierConGraph(hps, embed)
            logger.info("[MODEL] HierConGraph ")
            dataset = ExampleHierSet(DATA_FILE, INTERVIEWER_DATA_FILE, RELATION_DATABASE_PATH, RELATION_FILE, ABREAST_FILE, BERT_DATA_PATH, OIE_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleHierSet(VALID_FILE, INTERVIEWER_VALID_FILE, RELATION_DATABASE_PATH, VALID_RELATION_FILE, ABREAST_VALID_FILE, BERT_VALID_FILE, OIE_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        elif hps.model == "HDSG":
            model = HSumDocGraph(hps, embed)
            logger.info("[MODEL] HeterDocSumGraph ")
            train_w2d_path = os.path.join(args.cache_dir, "trn_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, train_w2d_path)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            val_w2d_path = os.path.join(args.cache_dir, "dev_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, val_w2d_path)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)  # Shuffle Must be False for ROUGE evaluation
        elif hps.model == "PAL":
            model = PromptAwareLSTM(hps, embed)
            logger.info("[MODEL] PromptAwareLSTM ")
            dataset = ExampleHierSet(DATA_FILE, INTERVIEWER_DATA_FILE, RELATION_DATABASE_PATH, RELATION_FILE, ABREAST_FILE, BERT_DATA_PATH, OIE_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleHierSet(VALID_FILE, INTERVIEWER_VALID_FILE, RELATION_DATABASE_PATH, VALID_RELATION_FILE, ABREAST_VALID_FILE, BERT_VALID_FILE, OIE_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        else:
            logger.error("[ERROR] Invalid Model Type!")
            raise NotImplementedError("Model Type has not been implemented")

    model = model.to(hps.device)

    # get recipe name
    hps = get_recipe_name(hps)
 
    # WanDB
    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=hps.save_dir_name,
            group="ddp" if args.multiple_device else "single",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": "NICTJLE",
                "epochs": args.n_epochs,
            }
        )
        logger.info("[INFO] Use WANDB")

    setup_training(model, train_loader, valid_loader, valid_dataset, hps, vocab)


if __name__ == '__main__':
    main(sys.argv[1:])
