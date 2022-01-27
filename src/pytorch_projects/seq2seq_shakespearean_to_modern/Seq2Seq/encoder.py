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
        self.embedding = nn.Embedding(input_size, embedding_size)
        # gru is an optimization of LSTM, ie a further optimized RNN
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True, dropout=p if num_layers > 1 else 0)
        """
        Choose the most relevant part from forward and backwards layers of GRU"""
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)

        self.device = device

    # input_vector is an array of indices corresponding to a tokenized sentence
        # input_vector shape: (seq_length, batch_size)
    # hidden is an array representing the hidden state of the previous input
        # hidden shape: (num_layers, batch_size, hidden_size)
        # embedding shape: (seq_length, batch_size, embedding_size)
    def forward(self, input_vector):
        embedding = self.dropout(self.embedding(input_vector))
        encoder_states, hidden = self.rnn(embedding)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        return encoder_states, hidden
