import json
import os
import random

import pandas as pd
from UniTok import Vocab


class Processor:
    def __init__(self):
        self.base_path = "/data8T_1/qijiong/Data/Goodreads"
        self.inter_path = os.path.join(self.base_path, "sixth_session.csv")
        self.df = pd.read_csv(
            filepath_or_buffer=self.inter_path,
            sep='\t',
            header=None,
            names=['uid', 'history', 'neg'],
        )
        self.df['history'] = self.df['history'].apply(eval)
        self.df['neg'] = self.df['neg'].apply(eval)

        self.book_path = os.path.join(self.base_path, "goodreads_book_works.json")
        self.book_vocab = Vocab(name='bid')

    def split_inter(self):
        # use df['uid'] and df['history']
        uid_list = []
        history_list = []
        train_list = []
        dev_list = []
        test_list = []
        neg_train_list = []
        neg_dev_list = []
        neg_test_list = []
        neg_list = []
        m = {
            '0.1': 5,
            '0.2': 6,
            '0.3': 7,
            '0.4': 8,
        }

        book_set = set()

        for uid, history in zip(self.df['uid'], self.df['history']):
            book_set.update(history)

            uid_list.append(uid)
            cut_ratio = random.random()
            for v in m:
                if cut_ratio <= float(v):
                    cut = m[v]
                    break
            else:
                cut = 9
            if cut == 5:
                cut = random.choice(range(5)) + 1

            history_list.append(history[:cut])
            besides = history[cut:]
            portion = len(besides) // 10
            if portion == 0:
                if len(besides) >= 2:
                    train_list.append(besides[:-2])
                    dev_list.append([besides[-2]])
                    test_list.append([besides[-1]])
                else:
                    train_list.append([])
                    dev_list.append([])
                    test_list.append(besides)
            else:
                train_list.append(besides[:-portion*2])
                dev_list.append(besides[-portion*2:-portion])
                test_list.append(besides[-portion:])

        for uid, neg in zip(self.df['uid'], self.df['neg']):
            book_set.update(neg)
            neg_list.append(neg)

            neg.extend(random.sample(book_set, 10))
            random.shuffle(neg)
            portion = len(neg) // 10
            neg_train_list.append(neg[:-portion*2])
            neg_dev_list.append(neg[-portion*2:-portion])
            neg_test_list.append(neg[-portion:])

        return uid_list, history_list, \
            train_list, dev_list, test_list, \
            neg_train_list, neg_dev_list, neg_test_list, neg_list, book_set

    def build_inter_df(self, uid_list, inter_list, neg_inter_list):
        uid = []
        inter = []
        click = []

        for index in range(len(uid_list)):
            if not inter_list[index]:
                continue
            uid.extend([uid_list[index]] * len(inter_list[index] + neg_inter_list[index]))
            inter.extend(inter_list[index] + neg_inter_list[index])
            click.extend([1] * len(inter_list[index]) + [0] * len(neg_inter_list[index]))

        return pd.DataFrame({
            'uid': uid,
            'bid': inter,
            'click': click,
        })

    def build_data(self):
        uid_list, history_list, \
            train_list, dev_list, test_list, \
            neg_train_list, neg_dev_list, neg_test_list, neg_list, book_set = self.split_inter()

        train_inter = self.build_inter_df(uid_list, train_list, neg_train_list)
        dev_inter = self.build_inter_df(uid_list, dev_list, neg_dev_list)
        test_inter = self.build_inter_df(uid_list, test_list, neg_test_list)

        user_df = pd.DataFrame({
            'uid': uid_list,
            'history': history_list,
        })

        neg_df = pd.DataFrame({
            'uid': uid_list,
            'neg': neg_list,
        })

        print(len(book_set))
        book_df = self.read_book_data(book_set)

        train_inter.to_csv(
            path_or_buf=os.path.join(self.base_path, "train_inter.csv"),
            sep='\t',
            index=False,
        )

        dev_inter.to_csv(
            path_or_buf=os.path.join(self.base_path, "dev_inter.csv"),
            sep='\t',
            index=False,
        )

        test_inter.to_csv(
            path_or_buf=os.path.join(self.base_path, "test_inter.csv"),
            sep='\t',
            index=False,
        )

        user_df.to_csv(
            path_or_buf=os.path.join(self.base_path, "user.csv"),
            sep='\t',
            index=False,
        )

        neg_df.to_csv(
            path_or_buf=os.path.join(self.base_path, "neg.csv"),
            sep='\t',
            index=False,
        )

        book_df.to_csv(
            path_or_buf=os.path.join(self.base_path, "book.csv"),
            sep='\t',
            index=False,
        )

    def read_book_data(self, book_set):
        bid_list = []
        title_list = []
        with open(self.book_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['best_book_id'] in book_set:
                    bid_list.append(data['best_book_id'])
                    title_list.append(data['original_title'])

        return pd.DataFrame({
            'bid': bid_list,
            'title': title_list,
        })


if __name__ == '__main__':
    processor = Processor()
    processor.build_data()
