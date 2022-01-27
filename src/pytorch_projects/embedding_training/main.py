import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        return embeds


