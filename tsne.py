import os
import torch
import random
import matplotlib

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

import torch.nn.functional as F
from module.embedding import Word_Embedding
from module.INFOembedding import CEFREmbed, FILLEDEmbed
from module.vocabulary import Vocab
from tools.utils import (
    BERT2ABB,
    pikleOpen,
)
from tools.args import get_eval_args
from tools.recipe_name import get_recipe_name

aux_objs = ['pos', 'deprel',
            'gra_sim', 'gra_med', 'gra_dif',
            'dif_sim', 'dif_med',
            'laugh', 'sil', 'w_cefr']

def fix_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if not args.use_amp:
    #     torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def visualize_layerwise_embeddings(dim_reducer, labels, embeds, title, ignore_zero=False):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    labels = np.array(labels)
    embeds = torch.cat(embeds, dim=0).detach().numpy()
    if ignore_zero:
        zs = np.where(labels == '0')[0]
        tfzs = np.ones_like(labels, dtype=bool)
        tfzs[zs] = False
        labels = labels[tfzs]
        embeds = embeds[tfzs]
    layer_dim_reduced_embeds = dim_reducer.fit_transform(embeds)
    df = pd.DataFrame.from_dict({'x':layer_dim_reduced_embeds[:,0],'y':layer_dim_reduced_embeds[:,1],'label':labels})
    sns.scatterplot(data=df,x='x',y='y',hue='label', ax=ax, 
                    palette=['green','orange','brown','dodgerblue','red', 'yellow'])
    # Turn off tick labels
    # ax.set_yticklabels([]) # it can only make the labels invisible but still remain the axis blocks
    # ax.set_xticklabels([])
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    save_path = os.path.join(alsy_dir, '{}.tsne.png'.format(title))
    plt.savefig(save_path, format='png', pad_inches=0, transparent=True)
    print('Image is saved at {}'.format(save_path))

def deal_mul_cefrs(taglist, cefr_loader, mode='minmean'):
    if taglist == [0]:
        return '0'
    assert mode in ['minmean', 'maxmean']
    CEFR2INT = cefr_loader.get_CEFR2INT()
    t_taglist = np.array([CEFR2INT[t] for t in taglist])
    if mode == 'minmean':
        m = np.array([np.mean(t_taglist).item()]*len(CEFR2INT))
        n = list(m - np.array(list(CEFR2INT.values())))
        min_v = min(n)
        mm_v = max(filter(lambda x: x == min_v, n))
        g_idx = n.index(mm_v)
        if 0. in n:
            g_idx = n.index(0.)
        return list(CEFR2INT.keys())[g_idx]
    elif mode == 'maxmean':
        m = np.array([np.mean(t_taglist).item()]*len(taglist))
        n = list(m - np.array(list(CEFR2INT.values())))
        mn_v = min(filter(lambda x: x >= 0, n))
        g_idx = n.index(mn_v)
        if 0. in n:
            g_idx = n.index(0.)
        return list(CEFR2INT.keys())[g_idx]

def draw_sphere(embeds, preds, reducer, title='sphere'):
    
    embeds = torch.cat(embeds, dim=0).numpy()
    preds = np.array(preds)
    
    reducer.fit_transform(embeds)
    rd_embeds = reducer.embedding_
    rd_embeds = torch.tensor(rd_embeds)
    rd_embeds = F.normalize(rd_embeds)
    embeds = rd_embeds.numpy()
    
    radius = 1
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    x = radius * np.outer(np.sin(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.cos(v))
    z = radius * np.outer(np.cos(u), np.ones_like(v))

    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color='black', linewidths = 0.3)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # ax.set_aspect("equal")
    ax.set_box_aspect([1,1,1])

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    plt.axis('off')
    plt.figure(dpi=3000)

    # For better visualization, i.e. the change of color will be more clear by cutting the upper and down
    value = preds
    # upper = np.percentile(value, 90)
    # upper = np.where(value<upper)[0]
    # value = value[upper]
    # embeds = embeds[upper]
    # down = np.percentile(value, 10)
    # down = np.where(value>down)[0]
    # value = value[down]
    # embeds = embeds[down]

    xs = embeds[:,0]
    ys = embeds[:,1]
    zs = embeds[:,2]

    cmapper = matplotlib.cm.get_cmap('magma_r')
    cmap = plt.get_cmap('brg', np.max(value) - np.min(value) +1)
    # sphere = ax.scatter(xs, ys, zs, c = value, cmap= 'brg', vmin=np.min(value), vmax=np.max(value))
    sphere = ax.scatter(xs, ys, zs, c=value, cmap=cmap, vmin=np.min(value)-0.5, vmax=np.max(value)+0.5)
    for i in range(xs.size):
        ax.plot([0, xs[i]], [0, ys[i]], [0, zs[i]], color = 'lightgrey')
    # fig.colorbar(sphere, fraction=0.02)
    fig.colorbar(sphere, ticks=np.arange(np.min(value), np.max(value)+1), fraction=0.02, pad=0.04)
    save_path = os.path.join(alsy_dir, '{}.tsne.png'.format(title))
    fig.savefig(save_path, format='png', pad_inches=0)
    print('Image is saved at {}'.format(save_path))

hps = get_eval_args(aux_objs)
hps = get_recipe_name(hps)  # get recipe name

alsy_dir = os.path.join(hps.save_root, hps.save_dir_name, "analysis")
if hps.sentaspara == 'para':
    pnode_file_path = os.path.join(alsy_dir, '{}.slabels.sembeds'.format(hps.which_set))
elif hps.sentaspara == 'sent':
    pnode_file_path = os.path.join(alsy_dir, '{}.plabels.pembeds'.format(hps.which_set))
(labels, embeds) = pikleOpen(pnode_file_path)
if hps.sentaspara == 'sent':
    speaker_id_list_file_path = os.path.join(alsy_dir, '{}.speaker_id.labels'.format(hps.which_set))
    speaker_id_labels = pikleOpen(speaker_id_list_file_path)

dim_reducer = TSNE(n_components=2, init='pca', learning_rate='auto', verbose=1, perplexity=20, n_iter=5000)

if hps.eval_speaker_wise:
    embeds = [torch.mean(e, dim=0).unsqueeze(0) for e in embeds]
if hps.sentaspara == 'sent':
    assert len(speaker_id_labels) == len(embeds), "{} {}".format(len(speaker_id_labels), len(embeds))
    tmp_spk_embeds = dict()
    tmp_spk_labels = dict()
    for (spk_id, embed, label) in zip(speaker_id_labels, embeds, labels):
        tmp_spk_embeds.setdefault(
            spk_id,
            []
        ).append(embed)
        tmp_spk_labels[spk_id] = label
    embeds = list({k: torch.mean(torch.cat(v, dim=0), dim=0).unsqueeze(0) for k, v in tmp_spk_embeds.items()}.values())
    labels = list(tmp_spk_labels.values())
    assert len(embeds) == len(labels)
    
## Paragraph Nodes t-SNE
visualize_layerwise_embeddings(dim_reducer=dim_reducer,
                               labels=labels,
                               embeds=embeds,
                               title='{}_data.2d'.format(hps.which_set),
                              )

## paragraph Nodes t-SNE on a sphere
dim_reducer = TSNE(n_components=3, init='pca', learning_rate='auto', verbose=1, perplexity=20, n_iter=5000)
draw_sphere(embeds, labels, dim_reducer, title='{}_data.sphere'.format(hps.which_set))

# # CEFR node and Filled Pauses node
# cefr_loader, filled_pauses_loader = None, None
# if hps.cefr_word:
#     VOCABPROFILE_FILE = os.path.join(hps.data_dir, 'cefrj1.6_c1c2.final.txt')
#     cefr_loader = CEFREmbed(hps.word_emb_dim, VOCABPROFILE_FILE)
# if hps.filled_pauses_word and (hps.filled_pauses_info == 'embed_init'):
#     FLUENCYPAUSE_FILE = os.path.join(hps.data_dir, 'all.filled_pauses.txt')
#     filled_pauses_loader = FILLEDEmbed(hps.word_emb_dim, FLUENCYPAUSE_FILE)
# VOCAL_FILE = os.path.join(hps.cache_dir, "vocab.combine" if hps.interviewer else 'vocab')
# vocab = Vocab(VOCAL_FILE, hps.vocab_size)
# assert cefr_loader is not None

# embedd = torch.nn.Embedding(vocab.size(), hps.word_emb_dim, padding_idx=0)
# embed_loader = Word_Embedding(hps.embedding_path, vocab)
# word2cefr, vectors = embed_loader.get_word_cefr_list_and_vectors(cefr_loader, filled_pauses_loader, k=hps.word_emb_dim)

# vectors = None
# if (cefr_loader or filled_pauses_loader):
#     if hps.cefr_info == 'embed_init':
#         vectors = embed_loader.load_my_vecs_with_infos(cefr_loader, filled_pauses_loader, k=hps.word_emb_dim)
#     else:
#         vectors = embed_loader.load_my_vecs(hps.word_emb_dim)
# else:
#     vectors = embed_loader.load_my_vecs(hps.word_emb_dim)

# pretrained_weight, word2cefr = embed_loader.add_unknown_words_by_avg(vectors, hps.word_emb_dim, dic=word2cefr)
# embedd.weight.data.copy_(torch.Tensor(pretrained_weight))

# # get learned cefr2embeds (cefr from referenced words. if a word mapped to multiple cefrs, get the minimum mean in the list)
# pnode_file_path = os.path.join(alsy_dir, '{}.wlabels.wembeds'.format(hps.which_set))
# (labels, embeds) = pikleOpen(pnode_file_path)
# collect_w = {}
# for data, ebd in zip(labels, embeds):
#     for i, wid in enumerate(data):
#         bd = ebd[i]
#         collect_w.setdefault(
#             wid,
#             []
#         ).append(bd)

# wid2lembed = {wid: torch.mean(torch.cat([e.unsqueeze(0) for e in c_bds], dim=0), dim=0).unsqueeze(0) for wid, c_bds in collect_w.items()}
# lembed_cefrs = [deal_mul_cefrs(word2cefr[vocab.id2word(wid)], cefr_loader, mode='minmean') for wid, _ in wid2lembed.items()]
# wid2embed = {wid: embedd(torch.tensor([wid])) for wid, _ in collect_w.items()}
# embed_cefrs = [deal_mul_cefrs(word2cefr[vocab.id2word(wid)], cefr_loader, mode='minmean') for wid, _ in wid2embed.items()]

# ## Word Nodes t-SNE
# dim_wnode_init = TSNE(n_components=2, init='pca', learning_rate='auto', verbose=1, perplexity=20, n_iter=5000)
# visualize_layerwise_embeddings(dim_reducer=dim_wnode_init,
#                                labels=embed_cefrs,
#                                embeds=list(wid2embed.values()),
#                                title='{}.wnode.init'.format(hps.which_set),
#                                ignore_zero=True,
#                               )

# dim_wnode_wenc = TSNE(n_components=2, init='pca', learning_rate='auto', verbose=1, perplexity=20, n_iter=5000)
# visualize_layerwise_embeddings(dim_reducer=dim_wnode_wenc,
#                                labels=lembed_cefrs,
#                                embeds=list(wid2lembed.values()),
#                                title='{}.wnode.wenc'.format(hps.which_set),
#                                ignore_zero=True,
#                               )
