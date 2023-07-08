import os

import pandas as pd
from tqdm import tqdm

large_data_dir = '/data1/qijiong/Data/MIND-large/'


def read_user_data(data_dir, mode):
    return pd.read_csv(
        filepath_or_buffer=os.path.join(data_dir, mode, 'behaviors.tsv'),
        sep='\t',
        names=['imp', 'uid', 'time', 'history', 'predict'],
        usecols=['imp', 'uid', 'history', 'predict'],
    )


train_user_data = read_user_data(large_data_dir, 'train')
dev_user_data = read_user_data(large_data_dir, 'dev')
test_user_data = read_user_data(large_data_dir, 'test')

user_dict = dict()


def rename_user(user_data, mode):
    behaviors = dict(imp=[], uid=[], history=[], predict=[])
    for line in tqdm(user_data.itertuples()):
        if line.uid not in user_dict:
            user_dict[line.uid] = []
        for i in range(len(user_dict[line.uid])):
            if user_dict[line.uid][i] == line.history:
                index = i
                break
        else:
            user_dict[line.uid].append(line.history)
            index = len(user_dict[line.uid]) - 1

        behaviors['imp'].append(line.imp)
        behaviors['uid'].append(f'{line.uid}_{index}')
        behaviors['history'].append(line.history)
        behaviors['predict'].append(line.predict)

    pd.DataFrame(behaviors).to_csv(
        path_or_buf=os.path.join(large_data_dir, mode, 'behaviors_v2.tsv'),
        sep='\t',
        index=False,
        header=False,
    )


rename_user(train_user_data, 'train')
rename_user(dev_user_data, 'dev')
rename_user(test_user_data, 'test')


# analyse user dict
count = 0
for uid in user_dict:
    if len(user_dict[uid]) > 1:
        count += 1

print(count)
