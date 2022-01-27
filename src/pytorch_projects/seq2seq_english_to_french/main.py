import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from pytorch_projects.seq2seq_english_to_french.DataTools.translation_dataset import TranslationDataset, CollateBatch
from pytorch_projects.seq2seq_english_to_french.Seq2Seq.decoder import DecoderRNN
from pytorch_projects.seq2seq_english_to_french.Seq2Seq.encoder import EncoderRNN
from pytorch_projects.seq2seq_english_to_french.Seq2Seq.seq2seq_model import Seq2Seq
from pytorch_projects.seq2seq_english_to_french.utils import load_checkpoint, translate_sentence, \
    save_checkpoint, regenerate_data

from torch.utils.tensorboard import SummaryWriter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    src_spacy, tar_spacy = "en_core_web_sm", "fr_core_news_sm"
    data_filename = "eng-fra.txt"

    tokenizers = {
        src_spacy: get_tokenizer("spacy", language=src_spacy),
        tar_spacy: get_tokenizer("spacy", language=tar_spacy)
    }

    special_symbols = {
        "<UNK>": 0, "<PAD>": 1, "<SOS>": 2, "<EOS>": 3
    }
    # Generate data if needed

    # Parameters for Dataloader
    batch_size = 512
    num_workers = 0
    pin_memory = True
    # Parameters for model training
    load_model = True
    save_model = True
    num_epochs = 100
    learning_rate = 0.0003
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 256
    num_layers = 1
    enc_dropout = 0.1
    dec_dropout = 0.1
    sentence = "he is painting a picture ."

    print("generating data...")
    # regenerate_data(data_filename, tokenizers, src_spacy, tar_spacy)
    print("creating dataset...")
    train_dataset = TranslationDataset("data/training.csv", tokenizers, special_symbols)
    print("success!")
    src_vocab, tar_vocab = train_dataset.src_vocab, train_dataset.tar_vocab
    input_size_encoder, input_size_decoder = len(src_vocab), len(tar_vocab)
    output_size = input_size_decoder

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  shuffle=True,
                                  collate_fn=CollateBatch(src_vocab, tar_vocab))

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
    criterion = nn.CrossEntropyLoss(ignore_index=special_symbols["<PAD>"])
    print("attempting to load model...")
    if load_model:
        load_checkpoint(torch.load("model_checkpoint.tar"), model, optimizer)

    """
    Perform training Loop """
    writer = SummaryWriter(f'runs/MNIST/tensorboard')
    step = 0
    emb_step = 0
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")
        if save_model:
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            })

        model.eval()

        translated_sentence = translate_sentence(model, sentence, src_vocab, tar_vocab, src_spacy, device,
        max_length=50)
        print(f"Translated example sentence: \n {translated_sentence}")

        model.train()
        print("waiting to begin batching...")
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 10 == 1:
                print("beginning batch... %s" % batch_idx)
            # Get input and targets and get to cuda
            src, tar = batch[0].to(device), batch[1].to(device)
            with torch.cuda.amp.autocast():
                output = model(src, tar, tar_vocab)
                output = output[1:].flatten(0, 1)
                tar = tar[1:].reshape(-1)
                loss = criterion(output, tar)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar('Training Loss', loss, global_step=step)
            step += 1
    # get embedding of whole source vocab
    model.eval()
    writer.add_embedding(
        encoder_net.embedding(torch.tensor([i for i in range(len(src_vocab))], device=device)),
        list(src_vocab.get_stoi().keys()),
        global_step=emb_step)
    emb_step += 1
    model.train()

if __name__ == "__main__":
    main()
