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
# 