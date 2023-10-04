import os.path
import random

import pandas as pd
from UniTok import UniDep, Fut

book_depot_dir = '../data/Goodreads/book-desc'
book_sum_depot_dir = '../data/Goodreads/book-sum'

book_depot = UniDep(book_depot_dir)
book_sum_depot = UniDep(book_sum_depot_dir)

titles = []
for index in range(len(book_depot)):
    sample = book_depot[index]
    target_len = len(book_sum_depot[index]['sum'])
    title = sample['title'] + sample['desc']
    indices = list(range(len(title)))
    random.shuffle(indices)
    indices = indices[:target_len]
    indices.sort()
    title = [title[i] for i in indices]
    titles.append(title)

df = pd.DataFrame(dict(
    bid=book_depot.data['bid'],
    title=titles,
))

Fut(
    df,
    book_depot,
    id_col='bid',
    refer_cols=['bid', 'title']
).store('../data/Goodreads/distillation/book-random')
