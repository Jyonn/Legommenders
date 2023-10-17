import os
import subprocess

import pandas as pd
from UniTok import UniDep
from tqdm import tqdm

from loader.column_map import ColumnMap
from utils.timing import Timing


class Submission:
    def __init__(
            self,
            depot: UniDep,
            column_map: ColumnMap,
            group_worker=5,
    ):
        self.base_dir = 'submission'
        os.makedirs(self.base_dir, exist_ok=True)

        self.depot = depot
        self.column_map = column_map
        self.group_worker = group_worker

        self.group_vocab = depot.vocabs[depot.cols[column_map.group_col].voc.name]
        self.item_vocab = depot.vocabs[depot.cols[column_map.candidate_col].voc.name]

        # self.group_dict = dict()
        # index = 0
        # for sample in tqdm(depot):
        #     if sample[self.column_map.group_col] not in self.group_dict:
        #         self.group_dict[sample[self.column_map.group_col]] = dict()
        #         index = 0
        #     index += 1
        #     self.group_dict[sample[self.column_map.group_col]][sample[self.column_map.candidate_col]] = str(index)
        self.group_dict_path = os.path.join(self.base_dir, 'group_dict.pkl')
        self.group_dict = self.get_group_dict()

    def get_group_dict(self):
        try:
            return pd.read_pickle(self.group_dict_path)
        except FileNotFoundError:
            pass

        group_dict = dict()
        index = 0
        for sample in tqdm(self.depot):
            if sample[self.column_map.group_col] not in group_dict:
                group_dict[sample[self.column_map.group_col]] = dict()
                index = 0
            index += 1
            group_dict[sample[self.column_map.group_col]][sample[self.column_map.candidate_col]] = str(index)
        pd.to_pickle(group_dict, self.group_dict_path)
        return group_dict

    def group_sort(self, group_id, items, scores):
        group = list(zip([self.group_dict[group_id][i] for i in items], scores))
        group.sort(key=lambda x: x[1], reverse=True)
        rank = [x[0] for x in group]
        reverse_rank = [0] * len(rank)
        for i, item in enumerate(rank):
            reverse_rank[int(item) - 1] = i + 1
        return group_id, reverse_rank

    def run(self, groups, scores, items, model_name):
        timestamp = Timing()['str']
        export_dir = os.path.join(self.base_dir, f'{model_name}_{timestamp}')
        os.makedirs(export_dir, exist_ok=True)

        df = pd.DataFrame(dict(groups=groups, scores=scores, news=items))
        df.to_csv(os.path.join(export_dir, 'prediction.csv'), index=False, header=False)

        groups = df.groupby('groups')

        # tasks = []
        # pool = Pool(processes=self.group_worker)
        # for g in groups:
        #     group = g[1]
        #     g_news = group.news.tolist()
        #     g_scores = group.scores.tolist()
        #     tasks.append(pool.apply_async(self.group_sort, args=(g[0], g_news, g_scores)))
        # pool.close()
        # pool.join()

        export_path = os.path.join(export_dir, 'prediction.txt')

        # with open(export_path, 'w') as f:
        #     for t in tasks:
        #         group_id, items = t.get()
        #         group_str = self.group_vocab.i2o[group_id]
        #         items_str = ','.join(items)
        #         f.write(f'{group_str} [{items_str}]\n')

        with open(export_path, 'w') as f:
            for g in tqdm(groups):
                group_id = g[0]
                group = g[1]
                g_news = group.news.tolist()
                g_scores = group.scores.tolist()
                _, items = self.group_sort(group_id, g_news, g_scores)
                group_str = self.group_vocab.i2o[group_id]
                items_str = ','.join(map(str, items))
                f.write(f'{group_str} [{items_str}]\n')

        subprocess.run(['zip', '-j', f'{export_dir}.zip', '-r', export_path])
        subprocess.run(['rm', '-rf', export_dir])

        return f'{export_dir}.zip'
