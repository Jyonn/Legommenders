import os.path
import random

import pandas as pd
from UniTok import UniDep, Fut

movie_depot_dir = '../data/MovieLens-100k/movie-desc'
movie_sum_depot_dir = '../data/MovieLens-100k/movie-sum'

movie_depot = UniDep(movie_depot_dir)

movie_sum_depot = UniDep(movie_sum_depot_dir)

names = []
for index in range(len(movie_depot)):
    sample = movie_depot[index]
    target_len = len(movie_sum_depot[index]['sum'])
    name = sample['name'] + sample['desc']
    indices = list(range(len(name)))
    random.shuffle(indices)
    indices = indices[:target_len]
    indices.sort()
    name = [name[i] for i in indices]
    names.append(name)

df = pd.DataFrame(dict(
    mid=movie_depot.data['mid'],
    name=names,
))

Fut(
    df,
    movie_depot,
    id_col='mid',
    refer_cols=['mid', 'name']
).store('../data/MovieLens-100k/distillation/movie-random')
