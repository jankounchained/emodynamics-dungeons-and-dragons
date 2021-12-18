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

df_long = df.query('n_datapoints > 160')
df_long = df_long.dropna(subset=['joy', 'sadness', 'fear', 'love', 'surprise'])


# %%
plt.plot()
plot = sns.scatterplot(
    x=df_long['joy'],
    y=df_long['anger']
)

fig = plot.get_figure()
fig.savefig('metadata_interactions/misc/joy_anger.png')

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
df_long_meta = pd.merge(df_long_nonan, meta, how='left', on ='episode')

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
