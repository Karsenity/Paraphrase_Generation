import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.tensorboard import SummaryWriter

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
freq_threshold = 1
# We will use Shakespeare Sonnet 2
src_spacy, tar_spacy = "en_core_web_sm", "fr_core_news_sm"

tokenizers = {
    src_spacy: get_tokenizer("spacy", language=src_spacy),
    tar_spacy: get_tokenizer("spacy", language=tar_spacy)
}
special_symbols = {
    "<UNK>": 0, "<PAD>": 1, "<SOS>": 2, "<EOS>": 3
}

df = pd.read_csv("training.csv")
src, tar = [k for k in tokenizers.keys()]
# Need to tokenize
for t in [src, tar]:
    df[t] = df.apply(lambda row: tokenizers[t](row[t]), axis=1)
# create vocabulary for source and target
src_iter, tar_iter = iter(df[src]), iter(df[tar])
src_vocab = build_vocab_from_iterator(src_iter, min_freq=freq_threshold, specials=list(special_symbols.keys()))
tar_vocab = build_vocab_from_iterator(tar_iter, min_freq=freq_threshold, specials=list(special_symbols.keys()))
[vocab.set_default_index(vocab["<UNK>"]) for vocab in [src_vocab, tar_vocab]]
# Go ahead and convert all data to indexes
df[src] = df.apply(lambda row: torch.tensor(src_vocab(["<SOS>"] + row[src] + ["<EOS>"])), axis=1)
df[tar] = df.apply(lambda row: torch.tensor(tar_vocab(["<SOS>"] + row[tar] + ["<EOS>"])), axis=1)

src_sents = list(df[src])

src_ngrams = [[([sent[i - j - 1] for j in range(CONTEXT_SIZE)], sent[i]) for i in range(CONTEXT_SIZE, len(sent))] for sent in src_sents]
ngrams = [item for sublist in src_ngrams for item in sublist]
ngrams = [(torch.stack(i[0], dim=0), i[1]) for i in ngrams]

vocab = src_vocab
# word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
"""
Perform training Loop """
writer = SummaryWriter(f'runs/MNIST/tensorboard')
step = 0
emb_step = 0
for epoch in range(50):
    total_loss = 0
    for context, target in ngrams:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        # context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.unsqueeze(target, dim=0))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!
writer.add_embedding(
    model.embeddings(torch.tensor([i for i in range(len(src_vocab))])),
    list(src_vocab.get_stoi().keys()),
    global_step=0)
# To get the embedding of a particular word, e.g. "beauty"
# print(model.embeddings.weight[word_to_ix["beauty"]])
