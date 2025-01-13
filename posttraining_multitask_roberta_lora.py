# https://github.com/theartificialguy/NLP-with-Deep-Learning/blob/master/BERT/Fine%20Tune%20BERT/fine_tuning_bert_with_MLM.ipynb

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from module.roberta.modeling_roberta import RobertaForMaskedLM
from module.longformer.modeling_longformer import LongformerModel, LongformerForMaskedLM
from transformers import get_linear_schedule_with_warmup

import os
import sys
import csv
import pickle
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from tools.utils import POS2INT, DEP2INT, GED2INT

from tqdm import tqdm
from collections import OrderedDict

from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, Union
# from transformers.models.roberta.modeling_roberta import (
#     _CONFIG_FOR_DOC,
#     _CHECKPOINT_FOR_DOC,
#     ROBERTA_INPUTS_DOCSTRING,
#     add_start_docstrings_to_model_forward,
#     add_code_sample_docstrings,
#     RobertaPreTrainedModel,
#     RobertaLMHead,
#     MaskedLMOutput,
# )
# from transformers.models.roberta.modeling_roberta import (
#     _CONFIG_FOR_DOC,
#     _CHECKPOINT_FOR_DOC,
#     ROBERTA_INPUTS_DOCSTRING,
#     add_start_docstrings_to_model_forward,
#     add_code_sample_docstrings,
#     RobertaPreTrainedModel,
#     RobertaLMHead,
#     MaskedLMOutput,
# )
from transformers.models.longformer.modeling_longformer import (
    LongformerLMHead,
    LongformerMaskedLMOutput,
)

# argument
model_path = 'roberta-base'

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# labels and call function
ged_difficult_labels = ['X', 'n_inf', 'n_num', 'n_cs', 'n_cnt', 'n_cmp', 'n_lxc', 'v_inf', 'v_agr', 'v_fml', 'v_tns', 'v_asp',
                        'v_vo', 'v_fin', 'v_ng', 'v_qst', 'v_cmp', 'v_lxc', 'mo_lxc', 'aj_inf', 'aj_us', 'aj_num', 'aj_agr',
                        'aj_qnt', 'aj_cmp', 'aj_lxc', 'av_inf', 'av_us', 'av_pst', 'av_lxc', 'prp_cmp', 'prp_lxc1', 'prp_lxc2',
                        'at', 'pn_inf', 'pn_agr', 'pn_cs', 'pn_lxc', 'con_lxc', 'rel_cs', 'rel_lxc', 'itr_lxc', 'o_je', 'o_lxc',
                        'o_odr', 'o_uk', 'o_uit',]

# arguments
def argparse_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir",
                    default='CEFR_LABELS_PATH/trn_cefr_scores.txt',
                    type=str)

    parser.add_argument('--rw_alpha', type=float, default=2.0, help='iteration hop [default: 2.0]')
    parser.add_argument('--batch_size', type=int, default=8, help='iteration hop [default: 8]')
    parser.add_argument('--lr', type=float, default=1e-5, help='iteration hop [default: 1e-5]')
    parser.add_argument('--num_epoch', type=int, default=20, help='iteration hop [default: 20]')
    parser.add_argument('--patience', type=int, default=4, help='iteration hop [default: 4]')

    args = parser.parse_args()

    return args

# Define your custom output directory
args = argparse_function()
output_dir = args.output_dir

# efcamdat CEFR score
CEFR2INT = {
    0: 0,
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6,
}

def get_gra_dif_labels():
    return ged_difficult_labels

# auxiliary tasks
auxiliary_inputs_dict = {
    'deprel': {
        'vs': len(DEP2INT),
        'loss_fct': CrossEntropyLoss(ignore_index=DEP2INT.get('X')),
        'bal': False,
        'padding_token_id': DEP2INT.get('X'),
        'embed': 'last'
    },
    'pos': {
        'vs': len(POS2INT),
        'loss_fct': CrossEntropyLoss(ignore_index=POS2INT.get('X')),
        'bal': False,
        'padding_token_id': POS2INT.get('X'),
        'embed': 'last'
    },
    'score': {
        'vs': 1,
        'loss_fct': torch.nn.MSELoss(reduce=True, reduction='mean'),
        'bal': True,
        'padding_token_id': 0,
        'embed': 'pooled'
    },
}
aux_tsk_pad_id_dict = {
    'pos': POS2INT.get('X'),
    'dep': DEP2INT.get('X'),
}

# tags
def get_full_tag(tag_text):
    return "<{}>".format(tag_text.split()[0].replace('</','').replace('<','').replace('>',''))

def check_same_contents(nums1, nums2):
    for x in set(nums1 + nums2):
        if nums1.count(x) != nums2.count(x):
            return False
    return True

def combine_tags(tokens, id, tag_array, other_labels, get_name=''):
    assert get_name != '', 'get_name should have value!'
    _median = ['X'] * len(tokens.split(' '))
    tokens_list = np.array(tokens.split(' '))
    sep_qa_indexes_list = np.where(tokens_list=='[SEP_QA]')[0].tolist()
    pair_indexes_list   = np.where(tokens_list=='[PAIR]')[0].tolist()
    for sep_qa_idx in sep_qa_indexes_list:
        _median[sep_qa_idx] = '[SEP_QA]'
    for pair_idx in pair_indexes_list:
        _median[pair_idx] = '[PAIR]'
    if 'X' in tag_array:
        tag_array.remove('X')
    for tag in tag_array:
        get_tag_labels = other_labels.get(id).get(get_name).get(get_full_tag(tag)).split()

        tag_sep_qa_indexes_list = np.where(np.array(get_tag_labels)=='[SEP_QA]')[0].tolist()
        tag_pair_indexes_list   = np.where(np.array(get_tag_labels)=='[PAIR]')[0].tolist()
        assert check_same_contents(tag_sep_qa_indexes_list, sep_qa_indexes_list), '{}: {} \n {}: {}'.format(len(tokens.split()), tokens, len(get_tag_labels), get_tag_labels)
        assert check_same_contents(tag_pair_indexes_list, pair_indexes_list), '{}: {} \n {}: {}'.format(len(tokens.split()), tokens, len(get_tag_labels), get_tag_labels)

        get_tag_labels = [int(l) if l.isnumeric() else 0 for l in get_tag_labels]
        get_tag_labels_np = np.array(get_tag_labels)
        get_labels_update_indexes = np.where(get_tag_labels_np==1)[0].tolist()
        for idx in get_labels_update_indexes:
            _median[idx] = tag
    return _median


# model initialization
def initial_bert_param(bin_file_path, model_type):
    state_dict = torch.load(bin_file_path)
    for name, param in model.named_parameters():
        fixed_name = model_type+'.'+name
        if fixed_name in state_dict:
            param = state_dict.get(fixed_name)

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def change_roberta_to_long_input_model(
    roberta_mlm_model,
    roberta_tokenizer,
    bert_config,
    attention_window: int = 512,
    longformer_max_length: int = 4096,
):

    ##################################
    # Create new longformer instance #
    ##################################
    bert_config.max_position_embeddings = longformer_max_length + 2
    bert_config.attention_window = attention_window
    # longformer_config = LongformerConfig(
    #     max_position_embeddings=longformer_max_length + 2,
    #     attention_window=attention_window,
    # )
    longformer_config = bert_config
    longformer_mlm_model = LongformerForMaskedLM(longformer_config)

    ###############################
    # Create longformer tokenizer #
    ###############################

    # Longformer tokenizers are Roberta tokenizers.
    # But to follow the conventions
    # and to avoid confusion we create a
    # longformer tokenizer class with the state of
    # the original tokenizer.
    # with TemporaryDirectory() as temp_dir:
    #     roberta_tokenizer.save_pretrained(temp_dir)
    #     longformer_tokenizer = LongformerTokenizerFast.from_pretrained(temp_dir)
    longformer_tokenizer = roberta_tokenizer
    longformer_tokenizer.model_max_length = longformer_max_length
    longformer_tokenizer.init_kwargs["model_max_length"] = longformer_max_length

    ######################
    # Copy model weights #
    ######################

    # We only copy the encoder weights and resize the embeddings.
    # Pooler weights are kept untouched.

    # ---------#
    # Encoder  #
    # ---------#
    roberta_parameters = roberta_mlm_model.roberta.encoder.state_dict()
    longformer_parameters = longformer_mlm_model.longformer.encoder.state_dict()

    # Load all compatible keys directly and obtain missing keys to handle later
    errors = longformer_mlm_model.longformer.encoder.load_state_dict(roberta_parameters, strict=False)
    assert not errors.unexpected_keys, "Found unexpected keys"
    missing_keys = errors.missing_keys

    # We expect, the keys to be the weights of the global attention modules and
    # reuse roberta's normal attention weights for those modules.
    for longformer_key in missing_keys:
        # Resolve layer properties
        (
            prefix,
            layer_idx,
            layer_class,
            layer_type,
            target,
            params,
        ) = longformer_key.split(".")
        assert layer_class == "attention" or target.endswith(
            "global"
        ), f"Unexcpected parameters {longformer_key}."
        # Copy the normal weights attention weights to the global attention layers too
        roberta_target_key = ".".join(
            [
                prefix,
                layer_idx,
                layer_class,
                layer_type,
                remove_suffix(target, "_global"),
                params,
            ]
        )
        # target.removesuffix("_global"),
        roberta_weights = roberta_parameters[roberta_target_key]
        longformer_parameters[longformer_key] = roberta_weights

    # Update the state of the longformer model
    longformer_mlm_model.longformer.encoder.load_state_dict(longformer_parameters, strict=True)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    longformer_mlm_model.longformer.resize_token_embeddings(len(roberta_tokenizer))

    roberta_embeddings_parameters = roberta_mlm_model.roberta.embeddings.state_dict()
    embedding_parameters2copy = []

    for key, item in roberta_embeddings_parameters.items():
        if not "position" in key and not "token_type_embeddings" in key:
            embedding_parameters2copy.append((key, item))

    # 2. Positional embeddings
    # The positional embeddings are repeatedly copied over
    # to longformer to match the new max_seq_length

    roberta_pos_embs = roberta_mlm_model.roberta.embeddings.state_dict()[
        "position_embeddings.weight"
    ][:-2]
    roberta_pos_embs_extra = roberta_mlm_model.roberta.embeddings.state_dict()[
        "position_embeddings.weight"
    ][-2:]

    assert (
        roberta_pos_embs.size(0) < longformer_max_length
    ), "Longformer sequence length has to be longer than roberta original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(longformer_max_length / roberta_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    longformer_pos_embs = roberta_pos_embs.repeat((n_copies, 1))

    n_pos_embs_left = longformer_max_length - longformer_pos_embs.size(0) # 1300 - 1536 = -236

    if n_pos_embs_left < 0:
        longformer_pos_embs = longformer_pos_embs[:n_pos_embs_left] # 1536 - 236 = 1300
    else:
        longformer_pos_embs = torch.cat(
            [longformer_pos_embs, roberta_pos_embs[:n_pos_embs_left]], dim=0
        )

    # Add the last extra embeddings.
    longformer_pos_embs = torch.cat(
        [longformer_pos_embs, roberta_pos_embs_extra], dim=0
    )

    embedding_parameters2copy.append(
        ("position_embeddings.weight", longformer_pos_embs)
    )

    # Load the embedding weights into the longformer model
    embedding_parameters2copy = OrderedDict(embedding_parameters2copy)
    longformer_mlm_model.longformer.embeddings.load_state_dict(embedding_parameters2copy, strict=False)

    return longformer_mlm_model

def load_paramters_to_model(config, model, longformer_mlm_model):
    model.longformer.embeddings.token_type_embeddings = nn.Embedding(2, config.hidden_size)
    model.longformer.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    model.longformer = longformer_mlm_model.longformer
    model.lm_head = longformer_mlm_model.lm_head
    # model.longformer.load_state_dict(longformer_mlm_model.longformer.state_dict(), strict=False)
    # model.lm_head.load_state_dict(longformer_mlm_model.lm_head.state_dict(), strict=False)
    return model

# dataset
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, is_eval, aux_tsk_pad_id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.other_data = pickle.load(open(other_labels_file_path, 'rb'))
        self.data = self._prep(self._read_tsv(file_path))
        self.reweight = self._get_reweight(self.data)
        self.is_eval = is_eval
        self.aux_tsk_pad_id = aux_tsk_pad_id

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            column = []
            for i, line in enumerate(reader):
                if i == 0:
                    column = line
                    continue
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return column, lines

    def _parse_nictjle_content(self, content):

        # special tokens
        unk_token = '[UNK]'
        splitter_qa = '[SEP_QA]' # it is a pesudo label for distinct <A> from <B>
        splitter_pair = '[PAIR]' # it is a pesudo label for distinct pairs from each other
        newline_token = '[NEW]'

        token_list = content.split()
        special_tokens_list = [splitter_qa, splitter_pair, newline_token]
        
        sentence  = []
        sentences = []

        for i, t in enumerate(token_list):
            if t in special_tokens_list:
                sentences.append(' '.join(sentence))
                sentence = []
                continue
            elif t == splitter_pair:
                sentences.append(' '.join(sentence))
                sentence = []
                continue
            if t == 'grammarerrorword':
                t = unk_token
            sentence.append(t)
            if len(token_list)-1 == i:
                sentences.append(' '.join(sentence))
        return sentences

    def _prep(self, data):
        column, lines = data
        content_idx = column.index('pre_token')
        w_cefr_idx  = column.index('pre_tokens_cefr')
        pos_idx = column.index('pos')
        dep_rels_idx = column.index('dep_rel')
        # utt_idx  = column.index('id')
        score_idx = column.index('level')

        # sentences = []
        # dep_sents = []
        # pos_sents = []
        # cefr_sents = []
        # ged_sents = []
        # for line in lines:
        #     id = line[utt_idx]
        #     sents = self._parse_nictjle_content(line[content_idx])
        #     sentences.append(' '.join(sents))
        #     cefrs = self._parse_nictjle_content(line[w_cefr_idx])
        #     cefr_sents.append(' '.join(cefrs))
        #     poss = self._parse_nictjle_content(line[pos_idx])
        #     pos_sents.append(' '.join(poss))
        #     deps = self._parse_nictjle_content(line[dep_rels_idx])
        #     dep_sents.append(' '.join(deps))
        #     grammar_difficult = combine_tags(line[content_idx], id, get_gra_dif_labels(), self.other_data, get_name='grammar_difficult') if self.other_data is not None else [0] * len(line[content_idx].split(' '))
        #     print('grammar_difficult: ', grammar_difficult)
        # rtn = {'input': sentences, 'cefr': cefr_sents, 'pos': pos_sents, 'dep': dep_sents}
        # return rtn
        rtn = []
        for line in lines:
            
            # customized requirements
            if len(line[content_idx]) < 10 or len(line[content_idx]) > self.max_length - 100:
                continue
            
            # id = line[utt_idx]
            sents = self._parse_nictjle_content(line[content_idx])
            cefrs = self._parse_nictjle_content(line[w_cefr_idx])
            poss = self._parse_nictjle_content(line[pos_idx])
            deps = self._parse_nictjle_content(line[dep_rels_idx])
            score = line[score_idx] if line[score_idx] != '0' else int(line[score_idx])
            score = CEFR2INT.get(line[score_idx])
            # grammar_difficult = combine_tags(line[content_idx], id, get_gra_dif_labels(), self.other_data, get_name='grammar_difficult') if self.other_data is not None else [0] * len(line[content_idx].split(' '))
            rtn.append({'input': sents, 'cefr': cefrs, 'pos': poss, 'dep': deps, 'score': score})
            # rtn.append({'input': sents, 'cefr': cefrs, 'pos': poss, 'dep': deps, 'ged': grammar_difficult, 'score': score})
        return rtn

    def _get_reweight(self, data):
        dd = {i: 0 for i in list(CEFR2INT.values())}
        for d in data:
            dd[d.get('score')] += + 1
        return dd

    def _get_special_tokens_mask(self, tokenizer, labels):
        tokenizer = self.tokenizer
        """ Returns a mask for special tokens that should be ignored for sampling during masked language modelling. """
        return list(map(lambda x: 1 if x in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id] else 0,
                        labels))

    def _mask_tokens(self, inputs=None):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        tokenizer = self.tokenizer
        labels = inputs.clone()
        # Sample tokens at 0.15 probability each.
        probability_matrix = torch.full(labels.shape, 0.15)
        # special_tokens_mask = [self._get_special_tokens_mask(tokenizer, val) for val in labels.tolist()]
        special_tokens_mask = self._get_special_tokens_mask(tokenizer, labels.tolist())
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # Only compute loss on tokens that are masked out
        labels[~masked_indices] = -1

        # Replace 80% of sampled tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # Replace 10% of sampled tokens with a random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # Leave remaining 10% of tokens as is
        return inputs, labels

    def _pad_with_zeros(self, tensor, target_length=1600):
        """
        Pad a 1D tensor with zeros to reach the target length in PyTorch.
        If the tensor is already longer than the target, it will not be truncated.
        """
        current_length = tensor.shape[0]
        if current_length < target_length:
            # Calculate the number of zeros needed to reach the target length
            padding_length = target_length - current_length
            # Create a tensor of zeros with the required length
            zeros_tensor = torch.zeros(padding_length, dtype=tensor.dtype)
            # Concatenate the original tensor with the zeros tensor
            padded_tensor = torch.cat((tensor, zeros_tensor), dim=0)
            return padded_tensor
        else:
            # If the tensor is already long enough, return it as is
            return tensor
        
    def _pad_with_minus_ones(self, tensor, target_length=1600):
        """
        Pad a 1D tensor with zeros to reach the target length in PyTorch.
        If the tensor is already longer than the target, it will not be truncated.
        """
        current_length = tensor.shape[0]
        if current_length < target_length:
            # Calculate the number of zeros needed to reach the target length
            padding_length = target_length - current_length
            # Create a tensor of zeros with the required length
            minus_one_tensor = torch.full((padding_length, 1), -1).squeeze(-1)
            # Concatenate the original tensor with the zeros tensor
            padded_tensor = torch.cat((tensor, minus_one_tensor), dim=0)
            return padded_tensor
        else:
            # If the tensor is already long enough, return it as is
            return tensor

    def _pad_with_number(self, tensor, number, target_length=1600):
        """
        Pad a 1D tensor with zeros to reach the target length in PyTorch.
        If the tensor is already longer than the target, it will not be truncated.
        """
        current_length = tensor.shape[0]
        if current_length < target_length:
            # Calculate the number of zeros needed to reach the target length
            padding_length = target_length - current_length
            # Create a tensor of zeros with the required length
            padding_tensor = torch.full((padding_length, 1), number).squeeze(-1)
            # Concatenate the original tensor with the zeros tensor
            padded_tensor = torch.cat((tensor, padding_tensor), dim=0)
            return padded_tensor
        else:
            # If the tensor is already long enough, return it as is
            return tensor

    def _generate_segment_ids(self, inputs):
        segment_ids = [0]
        for i, line in enumerate(inputs):
            for t in line.split():
                segment_ids.append(i%2)
            segment_ids.append(i%2)
        inputs = "{} {} {}".format(self.tokenizer.cls_token, (" {} ".format(self.tokenizer.sep_token)).join(inputs), self.tokenizer.sep_token)
        segment_ids = torch.tensor(segment_ids)
        return segment_ids, inputs
    
    def _generate_input(self, inputs):
        inputs = "{} {} {}".format('X', (" {} ".format('X')).join(inputs), 'X')
        return inputs

    def __getitem__(self, idx):
        token_type_ids, inputs = self._generate_segment_ids(self.data[idx]['input'])
        encoding = self.tokenizer(inputs, padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        input_ids, labels = self._mask_tokens(input_ids)
        
        pos_pad_id = self.aux_tsk_pad_id['pos']
        deprel_pad_id = self.aux_tsk_pad_id['dep']
        
        pos_tokens_labels = self._pad_with_number(torch.tensor([POS2INT.get(w, pos_pad_id) for w in self._generate_input(self.data[idx]['pos']).split()]), pos_pad_id, 1600)
        deprel_tokens_labels = self._pad_with_number(torch.tensor([DEP2INT.get(w, deprel_pad_id) for w in self._generate_input(self.data[idx]['dep']).split()]), deprel_pad_id, 1600)
        token_type_ids = self._pad_with_zeros(token_type_ids)
        score = torch.tensor(self.data[idx]['score']).float()

        # TODO: debug
        # ged_tokens_labels = self._pad_with_minus_ones(torch.tensor([GED2INT[w] for w in self._generate_input(self.data[idx]['ged']).split()]), 1600)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'pos_tokens_labels': pos_tokens_labels,
            'deprel_tokens_labels': deprel_tokens_labels,
            # 'ged_tokens_labels': ged_tokens_labels,
            'labels': labels,
            'score': score,
            'attention_mask': attention_mask,
        }

    def __len__(self):
        return len(self.data)

def custom_collect_fn(batch):
    input_ids_batch = torch.cat([item['input_ids'].unsqueeze(0) for item in batch], dim=0)
    token_type_ids_batch = torch.cat([item['token_type_ids'].unsqueeze(0) for item in batch], dim=0)
    pos_tokens_labels_batch = torch.cat([item['pos_tokens_labels'].unsqueeze(0) for item in batch], dim=0)
    deprel_tokens_labels_batch = torch.cat([item['deprel_tokens_labels'].unsqueeze(0) for item in batch], dim=0)
    labels_batch = torch.cat([item['labels'].unsqueeze(0) for item in batch], dim=0)
    score_batch = torch.cat([item['score'].unsqueeze(0) for item in batch], dim=0)
    attention_mask_batch = torch.cat([item['attention_mask'].unsqueeze(0) for item in batch], dim=0)
    return {
        'input_ids': input_ids_batch,
        'token_type_ids': token_type_ids_batch,
        'pos_tokens_labels': pos_tokens_labels_batch,
        'deprel_tokens_labels': deprel_tokens_labels_batch,
        'labels': labels_batch,
        'score': score_batch,
        'attention_mask': attention_mask_batch,
    }

# Model

# # class RobertaForMultiTask(nn.Module):
# class RobertaForMultiTask(RobertaForMaskedLM):
# # class RobertaForMaskedLM(RobertaPreTrainedModel):
#     _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

#     def __init__(self, config, auxiliary_inputs):
#         super().__init__(config)
#     # def __init__(self, config, auxiliary_inputs):
#     #     super().__init__()

#         if config.is_decoder:
#             logger.warning(
#                 "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
#                 "bi-directional self-attention."
#             )

#         self.auxiliary_inputs = auxiliary_inputs
#         for name, vocab_size in auxiliary_inputs.items():
#             head_config = config
#             head_config.vocab_size = vocab_size
#             head = RobertaLMHead(head_config)
#             setattr(self, '{}_head'.format(name), head)

#         self.roberta = RobertaModel(config, add_pooling_layer=False)
#         self.lm_head = RobertaLMHead(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_output_embeddings(self):
#         return self.lm_head.decoder

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head.decoder = new_embeddings

#     # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     # @add_code_sample_docstrings(
#     #     checkpoint=_CHECKPOINT_FOR_DOC,
#     #     output_type=MaskedLMOutput,
#     #     config_class=_CONFIG_FOR_DOC,
#     #     mask="<mask>",
#     #     expected_output="' Paris'",
#     #     expected_loss=0.1,
#     # )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs
#     ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
#             config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
#             loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
#         kwargs (`Dict[str, any]`, optional, defaults to *{}*):
#             Used to hide legacy arguments that have been deprecated.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]
#         prediction_scores = self.lm_head(sequence_output)

#         # other objective but no mlm
#         loss_accumulate = {name: 0 for name, _ in self.auxiliary_inputs.items()}
#         for name, vocab_size in self.auxiliary_inputs.items():
#             decoded_objective = getattr(self, name + '_head')(sequence_output)
#             if name != 'score':
#                 loss_fct = CrossEntropyLoss(ignore_index=-1)
#                 loss = loss_fct(decoded_objective.view(-1, vocab_size), kwargs['{}_tokens_labels'].view(-1))
#             else:
#                 loss_fct = torch.nn.MSELoss(reduction='none')
#                 loss = loss_fct(decoded_objective.view(-1, vocab_size), kwargs['score'].view(-1)) * kwargs['score_reweight']
#             loss_accumulate[name] = loss
#         loss_accumulate['overall'] = sum(loss_accumulate.values())
#         print('other: ', loss_accumulate['overall'])
#         input()

#         masked_lm_loss = None
#         if kwargs['labels'] is not None:
#             # move labels to correct device to enable model parallelism
#             labels = kwargs['labels']
#             labels = labels.to(prediction_scores.device)
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

#         masked_lm_loss = masked_lm_loss + loss_accumulate['overall']

#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

#         return MaskedLMOutput(
#             loss=masked_lm_loss,
#             logits=prediction_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

class LongformerForMultiTask(LongformerForMaskedLM):
    _tied_weights_keys = ["lm_head.decoder"]

    def __init__(self, config, auxiliary_inputs):
        super().__init__(config)

        self.auxiliary_inputs = auxiliary_inputs
        for name, c in auxiliary_inputs.items():
            head_config = deepcopy(config)
            head_config.vocab_size = c.get('vs')
            head_config.pad_token_id = c.get('padding_token_id')
            head = LongformerLMHead(head_config)
            setattr(self, '{}_head'.format(name), head)
        
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @replace_return_docstrings(output_type=LongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, LongformerMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Mask filling example:

        ```python
        >>> from transformers import AutoTokenizer, LongformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        ```

        Let's try a very long input.

        ```python
        >>> TXT = (
        ...     "My friends are <mask> but they eat too many carbs."
        ...     + " That's why I decide not to eat with them." * 300
        ... )
        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['healthy', 'skinny', 'thin', 'good', 'vegetarian']
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = self.mean_pooling(sequence_output, attention_mask)
        prediction_scores = self.lm_head(sequence_output)

        # other objective but no mlm
        batch_size = input_ids.shape[0]
        loss_accumulate = {name: 0 for name, _ in self.auxiliary_inputs.items()}
        for name, c in self.auxiliary_inputs.items():
            embedding_input = sequence_output if c.get('embed') == 'last' else pooled_output if c.get('embed') == 'pooled' else sequence_output
            decoded_objective = getattr(self, name + '_head')(embedding_input)
            decoded_objective = decoded_objective.view(-1, c.get('vs'))
            decoded_objective = decoded_objective.view(-1) if c.get('vs') == 1 else decoded_objective
            loss_fct = c.get('loss_fct')
            loss = loss_fct(decoded_objective, kwargs.get('{}_labels'.format(name)).view(-1))
            loss = loss * kwargs['score_reweight'] if c.get('bal') else loss
            loss = loss.sum() if c.get('vs') == 1 else loss
            loss_accumulate[name] = loss
        loss_accumulate['overall'] = sum(loss_accumulate.values())

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss(ignore_index=-1) # -1 is mask
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        masked_lm_loss = masked_lm_loss + loss_accumulate['overall']

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LongformerMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

# train_data_path = '/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data/trn_combo/text.tsv'
# dev_data_path = '/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data/dev_combo/text.tsv'
# eval_data_path = '/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data/eval_combo/text.tsv'
train_data_path = '/share/nas167/a2y3a1N0n2Yann/DataSet/EFCAMDAT/local/output/ef_trn.txt'
# train_data_path = '/share/nas167/a2y3a1N0n2Yann/DataSet/EFCAMDAT/local/output/ef_dev.txt' # debug
dev_data_path = '/share/nas167/a2y3a1N0n2Yann/DataSet/EFCAMDAT/local/output/ef_dev.txt'
eval_data_path = '/share/nas167/a2y3a1N0n2Yann/DataSet/EFCAMDAT/local/output/ef_dev.txt'
train_dataset = CustomDataset(train_data_path, tokenizer, max_length=1600, is_eval=False, aux_tsk_pad_id=aux_tsk_pad_id_dict)
dev_dataset = CustomDataset(dev_data_path, tokenizer, max_length=1600, is_eval=True, aux_tsk_pad_id=aux_tsk_pad_id_dict)
eval_dataset = CustomDataset(eval_data_path, tokenizer, max_length=1600, is_eval=True, aux_tsk_pad_id=aux_tsk_pad_id_dict)

# Dataloader
batch_size = args.batch_size
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collect_fn,
    num_workers=1
)
valid_dataloader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collect_fn,
    num_workers=1
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collect_fn,
    num_workers=1
)

# Initialize model
config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
config.auxiliary_inputs_dict = {} # fit for modified RobertaForMaskedLM
roberta_mlm_model = RobertaForMaskedLM.from_pretrained(model_path, config=config)
# it new a longformer and replace your modified forward function in your own model, do it first!
longformer_mlm_model = change_roberta_to_long_input_model(roberta_mlm_model,
                                            tokenizer, 
                                            config, 
                                            attention_window=[512]*12, 
                                            longformer_max_length=1600
                                           )
del roberta_mlm_model

model = LongformerForMultiTask(config=config, auxiliary_inputs=auxiliary_inputs_dict)
model = load_paramters_to_model(config, model, longformer_mlm_model)
del longformer_mlm_model

# Lora process
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["linear"],
)

# freeze model parameters
for name, param in model.named_parameters():
    if name in [
        'longformer.pooler.dense.weight', 'longformer.pooler.dense.bias',
        'longformer.embeddings.LayerNorm.weight', 'longformer.embeddings.LayerNorm.bias',
        'lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias',
    ]:
        continue
    if '_head' in name:
        continue

    param.requires_grad = False

# model to GPU
model = model.to(device)

# Initialize Trainer with your model, data, and training arguments
learning_rate = args.lr
EPOCHS = args.num_epoch
PATIENCE = args.patience
best_val_loss = float('inf')
early_stopping_counter = 0
adam_epsilon = 1e-3
weight_decay = 2e+1
warmup_steps = 0
gradient_accumulation_steps = 1
t_total = ((len(train_dataloader) + batch_size - 1) // batch_size) * EPOCHS // gradient_accumulation_steps

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

# reweight
reweight = {i:train_dataset.reweight[i] + dev_dataset.reweight[i] for i in CEFR2INT.values()}
reweight = {i: 1 - ((reweight[i]/len(train_dataset+dev_dataset)) ** args.rw_alpha) for i in CEFR2INT.values()}

# Fine-tune your model
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pos_tokens_labels = batch["pos_tokens_labels"].to(device)
        deprel_tokens_labels = batch["deprel_tokens_labels"].to(device)
        score_labels = batch["score"].to(device)
        score_reweight = torch.tensor([reweight[i] for i in batch["score"].tolist()]).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        pos_labels=pos_tokens_labels,
                        deprel_labels=deprel_tokens_labels,
                        score_labels=score_labels,
                        score_reweight=score_reweight)
        loss = outputs.loss
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        if (step + 1) % gradient_accumulation_steps == 0:
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

    average_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_train_loss}")

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        # Validation
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch_val in valid_dataloader:
                input_ids = batch_val["input_ids"].to(device)
                attention_mask = batch_val["attention_mask"].to(device)
                labels = batch_val["labels"].to(device)
                pos_tokens_labels = batch_val["pos_tokens_labels"].to(device)
                deprel_tokens_labels = batch_val["deprel_tokens_labels"].to(device)
                score_labels = batch["score"].to(device)
                val_outputs = model(input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    pos_labels=pos_tokens_labels,
                                    deprel_labels=deprel_tokens_labels,
                                    score_labels=score_labels,
                                    score_reweight=score_reweight)
                val_loss = val_outputs.loss
                val_total_loss += val_loss.item()

    average_val_loss = val_total_loss / len(valid_dataloader)

    print(f"Epoch {epoch + 1}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}")

    # Save the model if the validation loss has decreased
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        # torch.save(model.state_dict(), 'roberta_model.pth')
        save_output_dir = os.path.join(output_dir, 'epoch{}'.format(epoch))
        if not os.path.exists(save_output_dir):
            os.makedirs(save_output_dir)
        model.save_pretrained(save_output_dir)
        tokenizer.save_pretrained(save_output_dir)
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= PATIENCE:
            print(f"Early stopping after {epoch + 1} EPOCHS.")
            break