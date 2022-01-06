'''
To be turned into a jupyter notebook
'''

# %%
import os
import ndjson

import numpy as np
import pandas as pd
from tqdm import tqdm

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
# get emotion ratios per episode
signal_dir = '/home/jan/D&D/data/211001_20s_sentiment/signal_sentiment'
signal_paths = sorted(
    [os.path.join(signal_dir, path) for path in os.listdir(signal_dir)]
    )

results = pd.DataFrame([])
for path in tqdm(signal_paths):
    matrix, tag = load_transform_sentiment(path)
    row = calc_ratio_of_emotions(matrix, tag)
    results = results.append(row)

# %%
# import stuff
meta = pd.read_csv('/home/jan/D&D/data/211013_MetaData_episode_yt.csv', sep=';').rename(columns={'INDEX-NUMBER': 'episode'}).drop(columns=['n_views', 'n_likes', 'n_comments', 'n_dislikes'])
df_raw = pd.read_csv('/home/jan/D&D/data/211001_20s_sentiment/211013_signal_metadata.csv')
df = pd.merge(df_raw, results, on='episode', how='left')

df_long = df_long.dropna(subset=['joy', 'sadness', 'fear', 'love', 'surprise'])


# %%
# correlations between emotion
df_emo = df_long[['joy', 'sadness', 'fear', 'love', 'surprise', 'anger']]
cor_emo = df_emo.corr()

mask = np.triu(np.ones_like(cor_emo, dtype=bool))
f, ax = plt.subplots(figsize=(10, 8))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
fig = sns.heatmap(cor_emo, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

fig = fig.get_figure()
fig.savefig('../correlations.png')

# %%
fig = sns.pairplot(df_emo)
# fig = fig.get_figure()
fig.savefig('../emotion_proportions.png')

# %%
# plt.plot()
# plot = sns.scatterplot(
#     x=df_long['joy'],
#     y=df_long['anger']
# )

# fig = plot.get_figure()
# fig.savefig('metadata_interactions/misc/joy_anger.png')

# %%
# RN vs. anger
from sklearn.linear_model import LinearRegression

X = df_long['anger'].tolist()
X = np.array(X).reshape(-1, 1)

y = df_long['rn_slope'].tolist()
y = np.array(y).reshape(-1, 1)

lm = LinearRegression()
lm.fit(X, y)
lm.score(X, y)

y_pred = lm.predict(X)

plt.scatter(X, y, color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.xlabel('Anger (proportion of documents)')
plt.ylabel('R~N slope')
plt.savefig('metadata_interactions/misc/anger_rn.png')
plt.close()

# %%
def plot_emotion(emotion, df_, slope=False):
    X = df_[emotion].tolist()
    X = np.array(X).reshape(-1, 1)

    y = df_['rn_slope'].tolist()
    y = np.array(y).reshape(-1, 1)

    if slope:
        lm = LinearRegression()
        lm.fit(X, y)
        lm.score(X, y)

        y_pred = lm.predict(X)

        plt.scatter(X, y, color='black')
        plt.plot(X, y_pred, color='blue', linewidth=3)
        plt.xlabel(f'{emotion} (proportion of documents)')
        plt.ylabel('R~N slope')
        plt.savefig(f'metadata_interactions/misc/{emotion}_rn_slope.png')
        plt.close()
    else:
        plt.plot()
        plot = sns.scatterplot(
            x=df_[emotion],
            y=df_['rn_slope']
        )
        fig = plot.get_figure()
        fig.savefig(f'metadata_interactions/misc/{emotion}_rn.png')

for emo in ['joy', 'sadness', 'fear', 'love', 'surprise']:
    plot_emotion(emo, df_=df_long, slope=False)
    plot_emotion(emo, df_=df_long, slope=True)


# %%
###
###
###
df_long_nonan = df_long.dropna()
df_long_nonan.to_csv('/home/jan/D&D/data/211001_20s_sentiment/211013_emotion_ratio_nonan.csv', index=False)
df_long_nonan.info()

# %%
###
### comparte to meta
###
meta = pd.read_csv('/home/jan/D&D/data/211013_MetaData_episode_yt.csv', sep=';').rename(columns={'INDEX-NUMBER': 'episode'}).drop(columns=['n_views', 'n_likes', 'n_comments', 'n_dislikes'])
df_long_meta = pd.merge(df_long_nonan, meta, how='left', on='episode')

# %%
cols = df_long_meta.columns.tolist()

for var in cols[32:92]:
    plt.figure()
    plot = sns.violinplot(
        x=df_long_meta[var],
        y=df_long_meta['anger']
    )
    fig = plot.get_figure()
    fig.savefig(f'metadata_interactions/anger/anger_{var}.png')
    plt.close()

# %%
##
## R~N slopes distribution
sns.distplot(df_long['rn_slope'])

# %%
##
## decision tree
from sklearn.tree import export_graphviz  
from sklearn.tree import DecisionTreeRegressor 

X = df_emo
y = df_long['rn_slope']

regressor = DecisionTreeRegressor(random_state=0, max_depth=6)  
regressor.fit(X, y)

export_graphviz(regressor, out_file ='../tree.dot', 
               feature_names=['joy', 'sadness', 'fear', 'love', 'surprise', 'anger'])  
# %%
##
## distribution of emotions
df_emo_melt = df_emo.melt(var_name="emotion", value_name="proportion")
sns.violinplot(
    x=df_emo_melt['emotion'],
    y=df_emo_melt['proportion']
)


# %%
### episode length to 
fig = sns.scatterplot(
    x=df['n_datapoints'],
    y=df['rn_slope']
)

fig = fig.get_figure()
fig.savefig('../episode_length.png')


# %%
