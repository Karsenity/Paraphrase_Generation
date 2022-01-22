import re

import pandas as pd
import spacy
import torch
import unicodedata
from sklearn.model_selection import train_test_split


def regenerate_data(data_filename, tokenizers, src_spacy, tar_spacy):
    lines = open('data/%s' % data_filename, encoding='utf-8').read().strip().split('\n')
    src_data, target_data = map(list, zip(*[[normalize_str(s) for s in l.split('\t')[:2]] for l in lines]))
    df = pd.DataFrame({
        src_spacy: src_data,
        tar_spacy: target_data
    })
    # df = df.iloc[:1000]
    training, test = train_test_split(df, test_size=0.1)
    [v.to_csv('data/%s.csv' % k, index=False) for (k, v) in {"training": training, "test": test}.items()]


def normalize_str(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', s.lower().strip()) if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([.!?])", r" \1", s)
    return re.sub(r"[^a-zA-Z.!?]+", r" ", s)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def save_checkpoint(state, filename="model_checkpoint.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def translate_sentence(model, sentence, src_vocab, tar_vocab, src_spacy, device, max_length=50):
    src_spacy = spacy.load(src_spacy)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in src_spacy(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, "<SOS>")
    tokens.append("<EOS>")
    text_to_indices = src_vocab.lookup_indices(tokens)
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_states, hidden = model.encoder(sentence_tensor)
    outputs = [tar_vocab["<SOS>"]]
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(previous_word, encoder_states, hidden)
            best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        # Model predicts it's the end of the sentence
        if best_guess == tar_vocab["<EOS>"]:
            break
        return tar_vocab.lookup_tokens(outputs)[1:]
