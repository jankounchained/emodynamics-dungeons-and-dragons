''' 
'''
# %%
import os
import re
import ndjson

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from util import load_transform_sentiment, calc_ratio_of_emotions

# %%
# plt settings
# matplotlib settings
scale = 1.8

rc = {"text.usetex": False,
    "font.family": "Times New Roman",
    "font.serif": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.labelsize": 9*scale,
    "xtick.labelsize": 9*scale,
    "ytick.labelsize": 9*scale,
    "legend.fontsize": 9*scale,
    'axes.titlesize': 14,
    "axes.linewidth": 1
    }

plt.rcParams.update(rc)

sns.set_theme(
    style="ticks",
    # font='Times New Roman'
    rc=rc
    )


# %%
'''
Slope in different windows 
'''

# %%
data_dir = '../data/'
experiment_folders = os.listdir(data_dir)

pattern_bs20 = re.compile("b20.*")
b20_folders = list(filter(pattern_bs20.match, experiment_folders))
windows_result_paths = [os.path.join(data_dir, dir, 'results.ndjson') for dir in b20_folders]

# %%
results_w = []
for path, experiment_name in zip(windows_result_paths, b20_folders):
    with open(path) as fin:
        res_one = ndjson.load(fin)
        for episode in res_one:
            episode['experiment'] = experiment_name
    results_w.extend(res_one)

df_w = pd.DataFrame(results_w)
df_w['window'] = df_w['experiment'].str.extract(r'((?<=w)\d+)')
df_w['bin_size'] = df_w['experiment'].str.extract(r'((?<=b)\d+(?=_))')
    
# %%
fig = plt.figure(figsize=(8, 6))
g = sns.violinplot(
    x=df_w['window'],
    y=df_w['rn_slope'],
    order=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "30", "40", "50"],
    color="lightblue"
)

g.set_xlabel("Window size (in documents)")
g.set_ylabel("Episode-wise  " + r"$\mathcal{R} \times \mathcal{N}$" + "  slope")
g.grid(axis='y')

# %%
fig = g.get_figure()
fig.savefig('../window_comparison_b20.png', dpi=300)

# %%
'''
Slope in different bin sizes 
'''
# %%
data_dir = '../data/'
experiment_folders = os.listdir(data_dir)

pattern_w6 = re.compile(".*_w6")
w6_folders = list(filter(pattern_w6.match, experiment_folders))
bins_result_paths = [os.path.join(data_dir, dir, 'results.ndjson') for dir in w6_folders]

# %%
results_b = []
for path, experiment_name in zip(bins_result_paths, w6_folders):
    with open(path) as fin:
        res_one = ndjson.load(fin)
        for episode in res_one:
            episode['experiment'] = experiment_name
    results_b.extend(res_one)

df_b = pd.DataFrame(results_b)
df_b['window'] = df_b['experiment'].str.extract(r'((?<=w)\d+)')
df_b['bin_size'] = df_b['experiment'].str.extract(r'((?<=b)\d+(?=_))')

# %%
fig = plt.figure(figsize=(8, 6))
g = sns.violinplot(
    x=df_b['bin_size'],
    y=df_b['rn_slope'],
    order=["5", "10", "15", "20", "25", "30", "40"],
    color="lightblue"
)

g.set_xlabel("Timebin size (in seconds)")
g.set_ylabel("Episode-wise  " + r"$\mathcal{R} \times \mathcal{N}$" + "  slope")
g.grid(axis='y')

# %%
fig = g.get_figure()
fig.savefig('../bin_comparison_w6.png', dpi=300)


# %%
###
### average document length
###

bins_5s = '/home/jan/emodynamics-dungeons-and-dragons/data/b5_w6/bins'
paths_5s = [os.path.join(bins_5s, path) for path in os.listdir(bins_5s)]

# %%
n_chars = []
n_words = []
len_episodes = []
for file in paths_5s:
    # text
    with open(file, 'r') as f:
        text = ndjson.load(f)
    n_datapoints = len(text)
    len_episodes.append(n_datapoints)
    for idx in range(n_datapoints):
        n_chars.append(len(text[idx]['text']))
        n_words.append(len(text[idx]['text'].split(' ')))
# %%
