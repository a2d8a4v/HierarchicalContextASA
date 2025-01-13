# https://github.com/theartificialguy/NLP-with-Deep-Learning/blob/master/BERT/Fine%20Tune%20BERT/fine_tuning_bert_with_MLM.ipynb

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from module.roberta.modeling_roberta import RobertaForMaskedLM
from module.longformer.modeling_longformer import LongformerForMaskedLM
# from transformers import RobertaModel, LongformerModel, RobertaForMaskedLM, LongformerForMaskedLM
from transformers import get_linear_schedule_with_warmup

import os
import sys
import csv

import torch
import torch.nn as nn

from tools.utils import POS2INT, DEP2INT, GED2INT

from tqdm import tqdm
from collections import OrderedDict

# argument
model_path = 'roberta-base'

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Define your custom output directory
output_dir = '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/nictjle-roberta-base_multiple_inputs'  # Replace with your desired path

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def automodel(config):
    # return RobertaModel(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/roberta-base/pytorch_model.bin'
    return RobertaForMaskedLM(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/roberta-base/pytorch_model.bin'

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

from IPython.frontend.terminal.embed import InteractiveShellEmbed
# dataset
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, is_eval):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._prep(self._read_tsv(file_path))
        self.is_eval = is_eval

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
        w_cefr_idx  = column.index('pre_token_cefr')
        pos_idx = column.index('pos')
        dep_rels_idx = column.index('dep_rel')

        rtn = []
        for line in lines:
            sents = self._parse_nictjle_content(line[content_idx])
            cefrs = self._parse_nictjle_content(line[w_cefr_idx])
            poss = self._parse_nictjle_content(line[pos_idx])
            deps = self._parse_nictjle_content(line[dep_rels_idx])
            rtn.append({'input': sents, 'cefr': cefrs, 'pos': poss, 'dep': deps})
        return rtn

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

    def _generate_segment_ids(self, inputs):
        segment_ids = [0]
        for i, line in enumerate(inputs):
            for t in line.split():
                segment_ids.append(i%2)
            segment_ids.append(i%2)
        inputs = "{} {} {}".format(self.tokenizer.cls_token, (" {} ".format(self.tokenizer.sep_token)).join(inputs), self.tokenizer.sep_token)
        return segment_ids, inputs

    def __getitem__(self, idx):
        token_type_ids, inputs = self._generate_segment_ids(self.data[idx]['input'])
        encoding = self.tokenizer(inputs, padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = torch.tensor(encoding['input_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])
        input_ids, labels = self._mask_tokens(input_ids)
        pos_tokens_ids = self._pad_with_zeros(torch.tensor([POS2INT[w] for w in self.data[idx]['pos'].split()]), 1600)
        deprel_tokens_ids = self._pad_with_zeros(torch.tensor([DEP2INT[w] for w in self.data[idx]['dep'].split()]), 1600)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            # 'pos_tokens_ids': pos_tokens_ids,
            # 'deprel_tokens_ids': deprel_tokens_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

    # def __len__(self):
    #     return len(self.data['input'])

    def __len__(self):
        return len(self.data[0]['input'])

train_data_path = '/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data/trn_combo/text.tsv'
dev_data_path = '/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data/dev_combo/text.tsv'
eval_data_path = '/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data/eval_combo/text.tsv'
train_dataset = CustomDataset(train_data_path, tokenizer, max_length=1600, is_eval=False)
dev_dataset = CustomDataset(dev_data_path, tokenizer, max_length=1600, is_eval=True)
eval_dataset = CustomDataset(eval_data_path, tokenizer, max_length=1600, is_eval=True)

banner = '*** Nested interpreter ***'
exit_msg = '*** Back in main IPython ***'
ipshell = InteractiveShellEmbed(banner1=banner, exit_msg=exit_msg)

# Dataloader
batch_size = 8
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
valid_dataloader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=True
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=True
)

# Initialize model
config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
config.auxiliary_inputs_dict = dict() # debug
# model, bin_file_path = automodel(config)
model = RobertaForMaskedLM.from_pretrained(model_path, return_dict=True)
model.config.auxiliary_inputs_dict = {} # debug
model.config.type_vocab_size = 2
model.roberta.embeddings.token_type_embeddings = nn.Embedding(2, config.hidden_size)
model.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
model = change_roberta_to_long_input_model(model,
                                            tokenizer, 
                                            config, 
                                            attention_window=[512]*12, 
                                            longformer_max_length=1600
                                           )

# initial_bert_param(bin_file_path, config.model_type)

# for name, param in model.named_parameters():
#     print(name)
# input()

# freeze model parameters
for name, param in model.named_parameters():
    if name in [
        'longformer.pooler.dense.weight', 'longformer.pooler.dense.bias',
        'longformer.embeddings.LayerNorm.weight', 'longformer.embeddings.LayerNorm.bias',
        'lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias',
    ]:
        continue
    # if 'longformer.encoder.layer.0.intermediate' in name:
    #     continue
    # if 'longformer.encoder.layer.0.dense' in name:
    #     continue
    # if 'longformer.encoder.layer.0.LayerNorm' in name:
    #     continue
    # if 'longformer.encoder.layer.6.' in name:
    #     continue
    # if 'longformer.encoder.layer.11.' in name:
    #     continue
    param.requires_grad = False

# model to GPU
model = model.to(device)

# Initialize Trainer with your model, data, and training arguments
learning_rate = 1e-5
EPOCHS = 3
PATIENCE = 3
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

# Fine-tune your model
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} (Train)")):
        # print('epoch, step: ', epoch, step)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pos_tokens_ids = batch["pos_tokens_ids"].to(device)
        deprel_tokens_ids = batch["deprel_tokens_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, pos_tokens_ids=pos_tokens_ids, deprel_tokens_ids=deprel_tokens_ids)
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
                pos_tokens_ids = batch_val["pos_tokens_ids"].to(device)
                deprel_tokens_ids = batch_val["deprel_tokens_ids"].to(device)
                val_outputs = model(input_ids, attention_mask=attention_mask, labels=labels, pos_tokens_ids=pos_tokens_ids, deprel_tokens_ids=deprel_tokens_ids)
                val_loss = val_outputs.loss
                val_total_loss += val_loss.item()

    average_val_loss = val_total_loss / len(valid_dataloader)

    print(f"Epoch {epoch + 1}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}")

    # Save the model if the validation loss has decreased
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        # torch.save(model.state_dict(), 'roberta_mlm_model.pth')
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