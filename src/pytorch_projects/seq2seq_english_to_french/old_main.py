import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

from pytorch_projects.seq2seq_english_to_french.DataTools.prepare_data import PrepareData
from pytorch_projects.seq2seq_english_to_french.DataTools.GpuOnlyDataset import GpuOnlyDataset
from pytorch_projects.seq2seq_english_to_french.DataTools.LanguageDataset import LanguageDataset, TextCollate
from pytorch_projects.seq2seq_english_to_french.Decoders.decoder import DecoderRNN
from pytorch_projects.seq2seq_english_to_french.Encoders.encoder import EncoderRNN
from pytorch_projects.seq2seq_english_to_french.seq2seq_model import Seq2Seq
from pytorch_projects.seq2seq_english_to_french.utils import translate_sentence


def get_loader(root, data_file, transform, src_spacy, tar_spacy, batch_size=16, num_workers=4,
               shuffle=True, pin_memory=True):
    dataset = GpuOnlyDataset(root, data_file, src_spacy, tar_spacy, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory,
                        collate_fn=TextCollate(pad_index=dataset.src_vocab.s2i["<PAD>"]))
    return loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Parameters for Data Structure
    data_filename = "eng-fra.txt"
    src_spacy = "en_core_web_sm"
    tar_spacy = "fr_core_news_sm"
    # higher batch_size is faster but uses more memory
    batch_size = 32
    # workers are only used when data is being moved from cpu to GPU
    num_workers = 0
    # pin_memory is False if data is loaded directly onto GPU
    pin_memory = False

    """
    Enable this function if the data needs to be recreated"""
    # training, test = PrepareData.prepare_data(
    #     data_filename,
    #     src_spacy,
    #     tar_spacy,
    #     as_json=False,
    #     max_length=15,
    #     filter_data=True,
    #     split=True,
    #     test_size=0.2
    # )
    train_data, test_data = PrepareData.prepare_data(
        data_filename, src_spacy, tar_spacy, max_length=10, filter_data=True
    )

    dataloader = get_loader(
        root="data/",
        data_file="train.csv",
        transform=None,
        src_spacy=src_spacy,
        tar_spacy=tar_spacy,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Training hyperparameters
    num_epochs = 100
    learning_rate = 3e-4

    # Model hyperparameters
    load_model = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size_encoder = len(dataloader.dataset.src_vocab.s2i)
    input_size_decoder = len(dataloader.dataset.tar_vocab.s2i)
    output_size = input_size_decoder
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 256
    num_layers = 1
    enc_dropout = 0.5
    dec_dropout = 0.5
    src_vocab = dataloader.dataset.src_vocab
    tar_vocab = dataloader.dataset.tar_vocab
    sentence = "she is very smart and beautiful ."
    target_vocab = dataloader.dataset.tar_vocab.s2i

    encoder_net = EncoderRNN(
        input_size_encoder,
        encoder_embedding_size,
        hidden_size,
        num_layers,
        enc_dropout,
        device
    ).to(device)

    decoder_net = DecoderRNN(
        input_size_decoder,
        decoder_embedding_size,
        hidden_size,
        output_size,
        num_layers,
        dec_dropout,
        device
    ).to(device)

    model = Seq2Seq(encoder_net, decoder_net, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def load_checkpoint(checkpoint, model, optimizer):
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    if load_model:
        model = torch.load("model.tar")

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")
        translated_sentence = translate_sentence(model, sentence, src_vocab, tar_vocab, src_spacy, device,
                                                 max_length=50)
        print(f"Translated example sentence: \n {translated_sentence}")
        model.eval()
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            # Get input and targets and get to cuda
            inp_data = batch[0].to(device)
            target = batch[1].to(device)

            # Forward prop
            optimizer.zero_grad()
            model.zero_grad()
            output = model(inp_data, target, target_vocab)
            output = output[1:].flatten(0, 1)
            target = target[1:].reshape(-1)

            # Back prop
            loss = criterion(output, target)
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

    torch.save(model, "model.tar")


    # checkpoint = {
    #     "state_dict": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    # }
    # print("=> Saving checkpoint")
    # torch.save(checkpoint, "model.tar")

if __name__ == "__main__":
    main()

