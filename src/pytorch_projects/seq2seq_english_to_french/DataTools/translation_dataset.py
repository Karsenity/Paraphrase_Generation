import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator


class TranslationDataset(Dataset):
    def __init__(self, filepath, tokenizers, special_symbols, freq_threshold=5):
        self.df = pd.read_csv(filepath)
        src, tar = [k for k in tokenizers.keys()]
        # Need to tokenize
        for t in [src, tar]:
            self.df[t] = self.df.apply(lambda row: tokenizers[t](row[t]), axis=1)
        # create vocabulary for source and target
        src_iter, tar_iter = iter(self.df[src]), iter(self.df[tar])
        self.src_vocab = build_vocab_from_iterator(src_iter, min_freq=freq_threshold, specials=list(special_symbols.keys()))
        self.tar_vocab = build_vocab_from_iterator(tar_iter, min_freq=freq_threshold, specials=list(special_symbols.keys()))
        [vocab.set_default_index(vocab["<UNK>"]) for vocab in [self.src_vocab, self.tar_vocab]]
        # Go ahead and convert all data to indexes
        self.df[src] = self.df.apply(lambda row: torch.tensor(self.src_vocab(["<SOS>"] + row[src] + ["<EOS>"])), axis=1)
        self.df[tar] = self.df.apply(lambda row: torch.tensor(self.tar_vocab(["<SOS>"] + row[tar] + ["<EOS>"])), axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        val = self.df.iloc[index]
        return val[0], val[1]

class CollateBatch:
    def __init__(self, src_vocab, tar_vocab):
        self.src_vocab = src_vocab
        self.tar_vocab = tar_vocab

    def __call__(self, batch):
        src, tar = map(list, zip(*batch))
        return pad_sequence(src, batch_first=False, padding_value=self.src_vocab["<PAD>"]), \
               pad_sequence(tar, batch_first=False, padding_value=self.tar_vocab["<PAD>"])