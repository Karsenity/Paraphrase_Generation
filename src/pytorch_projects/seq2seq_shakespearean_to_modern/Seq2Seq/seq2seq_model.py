import torch
import torch.nn as nn
from numpy import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, target, target_vocab, teacher_force_ratio=0.5):
        batch_size = src.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(target_vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size, device=self.device)
        encoder_states, hidden = self.encoder(src)

        input_vector = target[0]
        for t in range(1, target_len):
            output, hidden = self.decoder(input_vector, encoder_states, hidden)
            outputs[t] = output
            best_guess = output.argmax(1)
            input_vector = target[t] if random.random() < teacher_force_ratio else best_guess

        # take max for each word
        return outputs
