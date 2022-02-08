import torch
import torch.nn as nn

# p = dropout = # of nodes to drop
"""
The encoder will take in our input_vector, which is a matrix of sentences
This matrix then will get multiplied by our embedding matrix
our product is then passed into our RNN, along with our previous hidden state
This gives us an output and a new hidden state"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=1)
        # gru is an optimization of LSTM, ie a further optimized RNN
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p if num_layers > 1 else 0)
        """
        Choose the most relevant part from forward and backwards layers of GRU"""
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)

        self.device = device

    def forward(self, input_vector):
        """
        :param input_vector: has shape (seq_length, batch)
        ---
        * (seq_len, batch) --embed--> (seq_len, batch, embed_size)
        * (seq_len, batch, embed_size) --rnn--> hidden=(2 * num_layers, batch, hidden_size)
            - the 2 comes from the fact our RNN is bidirectional
        * (2 * num_layers, batch, hidden_size) --cat--> (1, batch, 2 * hidden_size)
        * (1, batch, 2 * hidden_size) --linear_fc--> (1, batch, hidden_size)
        """
        embedding = self.dropout(self.embedding(input_vector))
        encoder_states, hidden = self.rnn(embedding)
        # hidden = torch.stack([torch.cat((t[i:i+1], t[i+1:i+2]), dim=2) for i in range(0, layers*2, 2)], dim=0).squeeze(1)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        return encoder_states, hidden