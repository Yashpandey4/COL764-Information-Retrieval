import glob
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from fastai.collab import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './data'
CSV_DIR = './converted'
MODEL_DIR = './models'


def train():
    global df
    dls = CollabDataLoaders.from_df(df, bs=64)
    learn = collab_learner(dls, y_range=(-0.5, 1.5))
    learn.lr_find()
    learn.fit_one_cycle(2, 5e-3)
    learn.save(MODEL_DIR)


for path in glob.glob(os.path.join(DATA_DIR, "*.json")):
    f = open(path)
    data = json.load(f)
    f.close()
    res = []  # [userid, trackid]
    for i in tqdm(data['playlists']):
        playlist_name = i['name']
        res.append([playlist_name, playlist_name, 1])
        for j in i['tracks']:
            res.append([playlist_name, j['track_name'], 1])
    df = pd.DataFrame(res, columns=['p_name', 't_name', 'rating'])
    df['t_name'] = df['t_name'].astype('string')
    df['t_name'] = pd.Categorical(df['t_name'])
    df['p_name'] = df['p_name'].astype('string')
    df['p_name'] = pd.Categorical(df['p_name'])
    df['rating'] = df['rating'].astype('int32')

    df['p_num'] = df.p_name.cat.codes
    df['t_num'] = df.t_name.cat.codes

    del df['p_name']
    del df['t_name']
    df.reset_index()

    df.to_csv(os.path.join(CSV_DIR, "all_files_converted.csv"), mode='a', index=False)

a = df['p_num'].value_counts()
df_1 = pd.DataFrame()
df_1['p_num'] = a.index
df_1['count'] = a.values

for index, rows in tqdm(df_1.iterrows()):
    # rows['p_name'] and rows['count'] gives the playlist name and count of tracks under that.
    count = int(rows['count'])
    p_num = rows['p_num']

    to_merge = df[df['p_num'] != p_num].sample(count).copy(deep=True)
    neg_samples = (list(to_merge['t_num']))
    for i in neg_samples:
        df.loc[-1] = [0, p_num, i]
        df.index = df.index + 1

df = df[['p_num', 't_num', 'rating']]
df.columns = ['userId', 'movieId', 'rating']

train()
