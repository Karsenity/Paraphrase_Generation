import re
from io import open

import unicodedata
from sklearn.model_selection import train_test_split
import pandas as pd

eng_prefixes = ("i am ", "i m ", "he is",
                "he s ", "she is", "she s ",
                "you are", "you re ", "we are",
                "we re ", "they are", "they re "
                )


class PrepareData:
    @classmethod
    def prepare_data(cls, filename, src_name, tar_name, as_json=False, max_length=15, filter_data=False, split=False, test_size=0.2):
        # convert data to ascii, convert it into pairs, filter pairs, and return as 2 separate sets.
        lines = open('data/%s' % filename, encoding='utf-8').read().strip().split('\n')
        pairs = [[cls.normalize_string(s) for s in l.split('\t')[:2]] for l in lines]
        if filter_data:
            pairs = cls.filter_pairs(pairs, max_length)
        src_data, target_data = map(list, zip(*pairs))
        df = pd.DataFrame({
            src_name: src_data,
            tar_name: target_data
        })
        # split into training and test data if we are
        training, test = train_test_split(df, test_size=test_size) if split else (df, None)
        # save using appropriate method
        cls.to_json(training, test) if as_json else cls.to_csv(training, test)
        return training, test

    @classmethod
    def filter_pairs(cls, pairs, max_length):
        return [
            p for p in pairs
            if min(len(p[0].split(' ')), len(p[1].split(' '))) <= max_length
            and p[0].startswith(eng_prefixes)
        ]

    @staticmethod
    def to_json(training, test=None):
        training.to_json('train.json', orient='records', lines=True)
        if test is not None:
            test.to_json('test.json', orient='records', lines=True)

    @staticmethod
    def to_csv(training, test=None):
        training.to_csv('train.csv', index=False)
        if test is not None:
            test.to_csv('test.csv', index=False)

    @classmethod
    def normalize_string(cls, s):
        s = cls.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
