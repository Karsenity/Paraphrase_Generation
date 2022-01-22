from collections import Counter

import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    def __init__(self, root_dir, parallel_corpus_file, src_spacy, tar_spacy, transform=None, freq_threshold=5):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.src, self.tar = self.df[columns[0]], self.df[columns[1]]

        # Initialize vocabulary and build vocabulary
        self.src_vocab = Vocabulary(columns[0], freq_threshold, src_spacy, self.markers)
        self.tar_vocab = Vocabulary(columns[1], freq_threshold, tar_spacy, self.markers)
        self.src_vocab.build_vocabulary(self.src)
        self.tar_vocab.build_vocabulary(self.tar)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        src, tar = self.src[index], self.tar[index]

        numericalized_src = [self.src_vocab.s2i["<SOS>"]] \
                            + self.src_vocab.numericalize(src) \
                            + [self.src_vocab.s2i["<EOS>"]]
        numericalized_tar = [self.tar_vocab.s2i["<SOS>"]] \
                            + self.tar_vocab.numericalize(tar) \
                            + [self.tar_vocab.s2i["<EOS>"]]
        return torch.tensor(numericalized_src), torch.tensor(numericalized_tar)


class Vocabulary:
    def __init__(self, name, freq_threshold, spacy_lang, markers):
        self.name = name
        self.s2i = markers
        self.i2s = dict((v, k) for k, v in markers.items())
        self.freq_threshold = freq_threshold
        self.spacy_lang = spacy.load(spacy_lang)

    def __len__(self):
        return len(self.i2s)

    @staticmethod
    def tokenizer_lang(text, spacy_lang):
        return [tok.text.lower() for tok in spacy_lang.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        index = 4
        frequencies = Counter([word for sentence in sentence_list for word in sentence])
        for word in frequencies.keys():
            if frequencies[word] >= self.freq_threshold:
                self.s2i[word], self.i2s[index] = index, word
                index += 1

    # sentence is passed in, tokenized adn then converted to indices.
    def numericalize(self, text):
        tokenized_text = self.tokenizer_lang(text, self.spacy_lang)
        return [self.s2i.get(token) if self.s2i.get(token) is not None else self.s2i.get("<UNK>") for token in tokenized_text]


class TextCollate:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        # given two tensors: english and french
        src = [item[0] for item in batch]
        src = pad_sequence(src, batch_first=False, padding_value=self.pad_index)
        target = [item[1] for item in batch]
        target = pad_sequence(target, batch_first=False, padding_value=self.pad_index)
        return src, target
