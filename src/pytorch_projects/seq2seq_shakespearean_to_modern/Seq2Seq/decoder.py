import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout=p if num_layers > 1 else 0)
        # fc = fully connected
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.energy = nn.Linear(hidden_size*3, 1)
        self.softmax_attn = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.device = device

    # input_vector is an array of indices corresponding to a tokenized sentence
        # input_vector shape: (N), squeeze reshapes it to (1, N)
    # hidden is an array representing the hidden state of the previous input
        # hidden shape: (num_layers, batch_size, hidden_size)
        # embedding shape: (1, batch_size, embedding_size)
        # outputs shape: (1, batch_size, hidden_size)
        # predictions shape: (1, batch_size, length_of_vocab)
    def forward(self, input_vector, encoder_states, hidden):
        input_vector = input_vector.unsqueeze(0)
        embedding = F.relu(self.dropout(self.embedding(input_vector)))
        # After embedding, need to compute energy
        seq_len = encoder_states.shape[0]
        h_reshaped = hidden.repeat(seq_len, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax_attn(energy)
        # (seq_len, batch_size, 1)
        # attention = attention.permute(1, 2, 0)
        # (N, 1, seq_len)
        # encoder_states = encoder_states.permute(1, 0, 2)
        # (N, seq_len, hidden_size*2)
        # context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)

        outputs, hidden = self.rnn(rnn_input, hidden)
        #outputs = self.fc(outputs).squeeze(0)
        # predictions = self.softmax(outputs)
        predictions = self.fc(outputs).squeeze(0)
        return predictions, hidden
