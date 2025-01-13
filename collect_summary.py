import argparse
import os
import json
import numpy as np

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

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

# compute the mean and std. for each metric
keys = {
    'rmse': [],
    'mc_rmse': [],
    'pearson': [],
    'within_0.5': [],
    'within_1': [],
    'mc_within_0.5': [],
    'mc_within_1': [],
    'oe_rate': [],
    'mc_oe_rate': [],
    'ue_rate': [],
    'mc_ue_rate': [],
}

for j in json_collect:
    for k in list(keys.keys()):
        keys[k].append(j[k])

means = {i: np.mean(np.array(j)) for i, j in keys.items()}
stds = {i: np.std(np.array(j)) for i, j in keys.items()}

# save summarized information
save_file_path = os.path.join(alsy_dir, 'summary.csv')
f = open(save_file_path, 'w')
f.write('{}\n'.format(','.join([' ']+list(keys.keys()))))
a = ['mean']
b = ['std']
for k in list(keys.keys()):
    a.append(f'{round_num(means[k]):.3f}')
    b.append(f'{round_num(stds[k]):.3f}')
f.write('{}\n'.format(','.join(a)))
f.write('{}\n'.format(','.join(b)))
f.close()

# save summarized information
save_file_path = os.path.join(alsy_dir, 'summary.latex')
f = open(save_file_path, 'w')
f.write('{}\n'.format(','.join([' ']+list(keys.keys()))))
a = ['mean']
b = ['std']
for k in list(keys.keys()):
    a.append(f'{round_num(means[k]):.3f}')
    b.append(f'{round_num(stds[k]):.3f}')
c = ['']
for ai, bi in zip(a, b):
    co = '\makecell{'+ai+' \\\\ $\pm$ '+bi+'}'
    c.append(co)
f.write('{}\n'.format(' & '.join(c)))
f.close()

labels = []
predictions = []
labels_keys = []
for i, j in enumerate(json_collect):

    labels.extend(j.get('labels'))
    predictions.extend(j.get('predictions'))
    if i == 0:
        labels_keys.extend(j.get('labels_keys'))

cm = confusion_matrix(labels, predictions, labels=labels_keys)

# confusion matrix - by count show count
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cbar_kws={'label': 'counts'}) #annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.xaxis.set_ticklabels(labels_keys)
ax.yaxis.set_ticklabels(labels_keys)
save_file_path = os.path.join(alsy_dir, '{}.confusion_matrix.count.sns.png'.format(which_set))
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
save_file_path = os.path.join(alsy_dir, '{}.confusion_matrix.correctness.percentage.sns.png'.format(which_set))
fig.savefig(save_file_path, bbox_inches='tight')
print('confusion matrix (by correctness show correctness) is saved at {}'.format(save_file_path))

# confusion matrix - by correctness show count
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sns.heatmap(data=df_percentages, annot=df_confusion, cmap='Blues', fmt="d", ax=ax, cbar_kws={'label': 'percentages'})
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
save_file_path = os.path.join(alsy_dir, '{}.confusion_matrix.correctness.count.sns.png'.format(which_set))
fig.savefig(save_file_path, bbox_inches='tight')
print('confusion matrix (by correctness show count) is saved at {}'.format(save_file_path))
