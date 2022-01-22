"""
This version of the dataset will load all of the data directly to the GPU

You can iterate through src and target to train your model
"""

from collections import Counter

import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset


class GpuOnlyDataset(Dataset):
    def __init__(self, root_dir, parallel_corpus_file, src_spacy, tar_spacy, transform=None, freq_threshold=0):
        self.root_dir = root_dir
        self.df = pd.read_csv(parallel_corpus_file)
        self.markers = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }

        self.transform = transform

        columns = list(self.df.columns)
        self.src, self.tar = self.df[columns[0]], self.df[columns[1]]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.src_vocab = Vocabulary(columns[0], freq_threshold, src_spacy, self.markers)
        self.tar_vocab = Vocabulary(columns[1], freq_threshold, tar_spacy, self.markers)

        self.src = self.src_vocab.build_vocabulary(self.src)
        self.tar = self.tar_vocab.build_vocabulary(self.tar)

        self.src, self.tar = torch.tensor(self.src, device=self.device), torch.tensor(self.tar, device=self.device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.src[index], self.tar[index]


class Vocabulary:
    def __init__(self, name, freq_threshold, spacy_lang, markers):
        self.name = name
        self.s2i = markers.copy()
        self.i2s = dict((v, k) for k, v in markers.items())
        self.freq_threshold = freq_threshold
        self.spacy_lang = spacy.load(spacy_lang)

    def __len__(self):
        return len(self.i2s)

    def tokenizer_lang(self, text):
        return ["<SOS>"] + [tok.text.lower() for tok in self.spacy_lang.tokenizer(text)] + ["<EOS>"]

    def build_vocabulary(self, sentences):
        tokenized = [self.tokenizer_lang(s) for s in sentences]
        max_size = max([len(s) for s in tokenized])
        freqs = Counter([word for s in tokenized for word in s])
        [self.s2i.update({word: len(self.s2i)}) for word in freqs.keys() if freqs[word] >= self.freq_threshold and word not in self.s2i.keys()]
        self.i2s = {v: k for k, v in self.s2i.items()}
        numericalized = [self.numericalize(t_s) for t_s in tokenized]
        return [s + [0] * (max_size - len(s)) for s in numericalized]

    def numericalize(self, tokenized_text):
        return [self.s2i.get(token) if self.s2i.get(token) is not None else self.s2i.get("<UNK>") for token in tokenized_text]

