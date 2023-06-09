#!/usr/bin/env python
# coding: utf-8

# # IFT6135-A2022
# # Assignment 2: Practical
# 
# You must fill in your answers to various questions in this notebook, following which you must export this notebook to a Python file named `solution.py` and submit it on Gradescope.
# 
# Only edit the functions specified in the PDF (and wherever marked – `# WRITE CODE HERE`). Do not change definitions or edit the rest of the template, else the autograder will not work.
# 
# **Make sure you request a GPU runtime!**

# DO NOT MODIFY!
import math
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

device = "cuda" if torch.cuda.is_available() else "cpu"

def fix_experiment_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_experiment_seed()


# #### Data pre-processing
# 
# Run the cells below by clicking the Run/Play button, but do not modify the code.


from collections import Counter
import io
import os
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive


os.system("python3 -m spacy download en_core_web_sm")
os.system("python3 -m spacy download fr_core_news_sm")


en_tokenizer = torchtext.data.utils.get_tokenizer("spacy", "en_core_web_sm")
fr_tokenizer = torchtext.data.utils.get_tokenizer("spacy", "fr_core_news_sm")


base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
train_urls = ["train.en.gz", "train.fr.gz"]
val_urls = ["val.en.gz", "val.fr.gz"]
test_urls = ["test_2016_flickr.en.gz", "test_2016_flickr.fr.gz"]
train_paths = [extract_archive(download_from_url(base_url + url))[0] for url in train_urls]
val_paths = [extract_archive(download_from_url(base_url + url))[0] for url in val_urls]
test_paths = [extract_archive(download_from_url(base_url + url))[0] for url in test_urls]


def generate_vocabulary(path, tokenizer):
    counter = Counter()
    with io.open(path, encoding="utf8") as f:
        for item in f:
            counter.update(tokenizer(item))
    vocabulary = torchtext.vocab.vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocabulary.set_default_index(vocabulary["<unk>"])
    return vocabulary

en_vocabulary = generate_vocabulary(train_paths[0], en_tokenizer)
fr_vocabulary = generate_vocabulary(train_paths[1], fr_tokenizer)


def get_processed_data(paths):
    en_file_io = iter(io.open(paths[0], encoding="utf8"))
    fr_file_io = iter(io.open(paths[1], encoding="utf8"))
    data = []
    for en_item, fr_item in zip(en_file_io, fr_file_io):
        en_processed = torch.tensor([en_vocabulary[token] for token in en_tokenizer(en_item)], dtype=torch.long)
        fr_processed = torch.tensor([fr_vocabulary[token] for token in fr_tokenizer(fr_item)], dtype=torch.long)
        data.append((en_processed, fr_processed))
    return data

train_data = get_processed_data(train_paths)
val_data = get_processed_data(val_paths)
test_data = get_processed_data(test_paths)


# ### GRU Encoder-Decoder Model for Machine Translation

# #### Dataloader
# 
# Running this cell will create `train_loader`, `val_loader` and `test_loader` for training, validation and testing respectively. The batch size is 128 and all sequences are of length 60 – shorter sequences are padded to this length. For this question, the data is provided in the shape `[sequence_length, batch_size]`. You need not modify this.


special_idx = {
    "bos": en_vocabulary["<bos>"],
    "pad": en_vocabulary["<pad>"],
    "eos": en_vocabulary["<eos>"]
}

max_len = 60

def get_batch(data):
    en_batch_, fr_batch_ = [], []
    for en_item, fr_item in data:
        en_batch_.append(torch.cat([torch.tensor([special_idx["bos"]]), en_item, torch.tensor([special_idx["eos"]])], dim=0))
        fr_batch_.append(torch.cat([torch.tensor([special_idx["bos"]]), fr_item, torch.tensor([special_idx["eos"]])], dim=0))
    en_batch_ = nn.utils.rnn.pad_sequence(en_batch_, padding_value=special_idx["pad"])
    fr_batch_ = nn.utils.rnn.pad_sequence(fr_batch_, padding_value=special_idx["pad"])
    en_batch = torch.full((max_len, en_batch_.shape[1]), special_idx["pad"])
    en_batch[:en_batch_.shape[0], :] = en_batch_
    fr_batch = torch.full((max_len, fr_batch_.shape[1]), special_idx["pad"])
    fr_batch[:fr_batch_.shape[0], :] = fr_batch_
    return en_batch, fr_batch

batch_size = 128
if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=get_batch)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=get_batch)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=get_batch)


# #### Encoder-Decoder Model


class Encoder(nn.Module):
    """
    Defines an encoder model containing a GRU.
    """
    def __init__(
        self,
        vocabulary_size=10838, # Size of source vocabulary, given by len(en_vocabulary)
        embedding_size=32,     # Size of nn.Embedding
        hidden_size=64,        # Hidden size for nn.GRU
        num_layers=1,          # Number of layers in the nn.GRU
        use_dropout=False,     # Whether or not to use Dropout
        p_dropout=0.1          # Dropout probability
    ):
        super(Encoder, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        
        self.embedding   = nn.Embedding(vocabulary_size, embedding_size, padding_idx=special_idx["pad"])
        if self.use_dropout:
            self.dropout = nn.Dropout(p_dropout)
        self.gru         = nn.GRU(embedding_size, hidden_size)

    def forward(self, x):
        out = self.embedding(x)
        if self.use_dropout:
            out = self.dropout(out)
        output, h = self.gru(out)
        return output, h


class Decoder(nn.Module):
    """
    Defines a decoder model containing a GRU.
    """
    def __init__(
        self,
        vocabulary_size=11510, # Size of target vocabulary, given by len(fr_vocabulary)
        embedding_size=32,     # Size of nn.Embedding
        hidden_size=64,        # Hidden size for nn.GRU
        num_layers=1,          # Number of layers in the nn.GRU
        use_dropout=False,     # Whether or not to use Dropout
        p_dropout=0.1          # Dropout probability
    ):
        super(Decoder, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_dropout = use_dropout

        self.embedding   = nn.Embedding(vocabulary_size, embedding_size)
        if self.use_dropout:
            self.dropout = nn.Dropout(p_dropout)
        self.gru         = nn.GRU(embedding_size, hidden_size)
        self.fc          = nn.Linear(hidden_size, vocabulary_size)
    
    def forward(self, x, hidden):
        x = x.unsqueeze(0)

        out = self.embedding(x)
        if self.use_dropout:
            out = self.dropout(out)
        output, h = self.gru(out, hidden)
        output = self.fc(output.squeeze(0))
        return output, h


class Seq2Seq(nn.Module):
    """
    Defines an encoder-decoder (sequence-to-sequence) model for machine translation.
    """
    def __init__(
        self,
        encoder,
        decoder,
        strategy="greedy"
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.strategy = strategy

    def forward(self, source, target):
        batch_size = source.shape[1]
        max_length = target.shape[0]
        target_vocabulary_size = self.decoder.vocabulary_size

        outputs = torch.zeros(max_length, batch_size, target_vocabulary_size).to(device)
        sentences = torch.zeros(max_length, batch_size).to(device)

        # WRITE CODE HERE
        # Get the encoder outputs and hidden state for the source sentences
        encoder_out, hidden = self.encoder(source)

        # The first input token passed to the decoder is <bos>
        out = target[0, :]

        if self.strategy == "greedy":
            for idx in range(1, max_length):
                # WRITE CODE HERE
                # Get the decoder outputs and hidden state for the input token, encoder outputs and hidden state
                out, hidden = self.decoder(out, hidden)

                outputs[idx] = out

                # WRITE CODE HERE
                # Reassign out with the greedy choice (argmax) for each batch
                # (batch_size, vocabulary_size) -> (batch_size)
                out = out.argmax(dim=1)

                sentences[idx] = out
        elif self.strategy == "random":
            temperature = 0.5
            for idx in range(1, max_length):
                # WRITE CODE HERE
                # Get the decoder outputs and hidden state for the input token, encoder outputs and hidden state
                out, hidden = self.decoder(out, hidden)

                outputs[idx] = out

                # WRITE CODE HERE
                # Reassign out with the randomly sampled choice for each batch
                # p(x_i | x_{1:i-1}) =  exp(o_i / temperature) / sum(exp(o / temperature))
                # (batch_size, vocabulary_size) -> (batch_size)
                temperature = 0.5
                p = F.softmax(out / temperature, dim=1)
                d = torch.multinomial(p, 1)
                out = d.squeeze().to(device)

                sentences[idx] = out
        else:
            raise Exception("Invalid decoding strategy!")

        return outputs, sentences


# #### Training


def get_criterion():
    # WRITE CODE HERE
    # The criterion must compute the cross-entropy loss but ignore special_idx["pad"].
    criterion = nn.CrossEntropyLoss(ignore_index=special_idx["pad"])
    return criterion


def train(model, dataloader, optimizer, criterion):
    train_loss = 0.
    train_ppl = 0.
    model.train()
    start = time.time()

    for source, target in dataloader:
        source, target = source.to(device), target.to(device)

        optimizer.zero_grad()
        output, _ = model(source, target)
        output_reshape = output[1:].view(-1, output.shape[-1])
        target = target[1:].view(-1)
        loss = criterion(output_reshape, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    end = time.time()
    train_loss /= len(dataloader)
    train_ppl = math.exp(train_loss)
    train_time = end - start
    return train_loss, train_ppl, train_time

def validate(model, dataloader, criterion):
    val_loss = 0.
    val_ppl = 0.
    model.eval()
    start = time.time()

    for source, target in dataloader:
        source, target = source.to(device), target.to(device)

        output, _ = model(source, target)
        output_reshape = output[1:].view(-1, output.shape[-1])
        target = target[1:].view(-1)
        loss = criterion(output_reshape, target)
        val_loss += loss.item()

    end = time.time()
    val_loss /= len(dataloader)
    val_ppl = math.exp(val_loss)
    val_time = end - start
    return val_loss, val_ppl, val_time

def compute_bleu_gru(model, dataloader):
    targets = []
    translations = []

    for source, target in dataloader:
        source, target = source.to(device), target.to(device)

        _, output = model(source, target)
        for i in range(target.shape[1]):
            targets.append([[]])
            for token in target[1:, i]:
                if token in special_idx.values():
                    break
                targets[-1][0].append(fr_vocabulary.get_itos()[token])
            translations.append([])
            for token in output[1:, i]:
                if token in special_idx.values():
                    break
                translations[-1].append(fr_vocabulary.get_itos()[token.int().item()])

    return torchtext.data.metrics.bleu_score(translations, targets, max_n=1, weights=[1]) * 100,            torchtext.data.metrics.bleu_score(translations, targets, max_n=2, weights=[0.5]*2) * 100,


fix_experiment_seed()
use_dropout = False
strategy = "greedy"
encoder = Encoder(use_dropout=use_dropout).to(device)
decoder = Decoder(use_dropout=use_dropout).to(device)
model = Seq2Seq(encoder, decoder, strategy=strategy).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu_gru(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"seq2seq-adam-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)


fix_experiment_seed()
use_dropout = True
strategy = "greedy"
encoder = Encoder(use_dropout=use_dropout).to(device)
decoder = Decoder(use_dropout=use_dropout).to(device)
model = Seq2Seq(encoder, decoder, strategy=strategy).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu_gru(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"seq2seq-sgd-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)


fix_experiment_seed()
use_dropout = True
strategy = "greedy"
encoder = Encoder(use_dropout=use_dropout).to(device)
decoder = Decoder(use_dropout=use_dropout).to(device)
model = Seq2Seq(encoder, decoder, strategy=strategy).to(device)
optimizer = optim.Adam(model.parameters())
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu_gru(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"seq2seq-adam-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)


fix_experiment_seed()
use_dropout = True
strategy = "random"
encoder = Encoder(use_dropout=use_dropout).to(device)
decoder = Decoder(use_dropout=use_dropout).to(device)
model = Seq2Seq(encoder, decoder, strategy=strategy).to(device)
optimizer = optim.Adam(model.parameters())
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu_gru(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"seq2seq-adam-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)


# ### Transformer Encoder-Decoder Model for Machine Translation

# #### Dataloader
# 
# Running this cell will create `train_loader`, `val_loader` and `test_loader` for training, validation and testing respectively. The batch size is 128 and all sequences are of length 60 – shorter sequences are padded to this length. For this question, the data is provided in the shape `[batch_size, sequence_length]`. You need not modify this.


special_idx = {
    "bos": en_vocabulary["<bos>"],
    "pad": en_vocabulary["<pad>"],
    "eos": en_vocabulary["<eos>"]
}

max_len = 60

def get_batch(data):
    en_batch_, fr_batch_ = [], []
    for en_item, fr_item in data:
        en_batch_.append(torch.cat([torch.tensor([special_idx["bos"]]), en_item, torch.tensor([special_idx["eos"]])], dim=0))
        fr_batch_.append(torch.cat([torch.tensor([special_idx["bos"]]), fr_item, torch.tensor([special_idx["eos"]])], dim=0))
    en_batch_ = nn.utils.rnn.pad_sequence(en_batch_, padding_value=special_idx["pad"], batch_first=True)
    fr_batch_ = nn.utils.rnn.pad_sequence(fr_batch_, padding_value=special_idx["pad"], batch_first=True)
    en_batch = torch.full((en_batch_.shape[0], max_len), special_idx["pad"])
    en_batch[:, :en_batch_.shape[1]] = en_batch_
    fr_batch = torch.full((fr_batch_.shape[0], max_len), special_idx["pad"])
    fr_batch[:, :fr_batch_.shape[1]] = fr_batch_
    return en_batch, fr_batch

batch_size = 128
if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=get_batch)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=get_batch)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=get_batch)


# #### Embeddings
# 
# The classes below are used to get the Transformer embeddings. The embeddings are made up of positional encodings and token embeddings. You do not need to modify the code, and you may run it by just clicking the run/play button.


class PostionalEncoding(nn.Module):
    """
    Provides positional information for tokens, which is to be added with the token embeddings.
    """
    def __init__(self, d_model, max_len=max_len, device=device):
        super(PostionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    """
    Gives embeddings made up of token embeddings + positional encodings.
    """
    def __init__(self, vocabulary_size, d_model, max_len=max_len, use_dropout=False, p_dropout=0.1, device=device):
        super(TransformerEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=special_idx["pad"])
        self.positional_encoding = PostionalEncoding(d_model, max_len, device)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        positional_encodings = self.positional_encoding(x)
        embeddings = token_embeddings + positional_encodings
        if self.use_dropout:
            embeddings = self.dropout(embeddings)
        return embeddings


# #### LayerNorm


class LayerNorm(nn.Module):
    """
    Implements layer normalization.
    """
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, x):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07, slide 20.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.weight * out + self.bias
        return out

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


# #### Masked Multi-Head Attention
# 
# We use masked multi-head attention because the decoder should not attend to future tokens in the target while performing autoregressive generation (machine translation).


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.num_heads = num_heads

        # Use nn.Linear objects so as to apply both weights and biases
        self.w_q = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size)
        self.w_k = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size)
        self.w_v = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size)
        self.w_y = nn.Linear(self.num_heads * self.head_size, self.num_heads * self.head_size)

    def get_attention_weights(self, queries, keys, mask=None):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            x = Q * K^{T} / sqrt(head_size)
            weights = softmax(x, mask) = softmax(x . mask - 10^4 (1 - mask))

            If mask is None,

            weights = softmax(x)

        Here "*" is the matrix multiplication and "." is element-wise multiplication. See Lecture 06, slides 05-13.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads.

        mask (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the mask. Default: None.

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """
        if mask is not None:
            mask = mask.int()
            print(mask.shape)
        print(queries.shape, keys.shape)
        s = queries @ keys.transpose(2, 3) / np.sqrt(self.head_size)
        if mask is not None:
            s = s * mask - 10 ** 4 * (1 - mask)
        return F.softmax(s, dim=-1)

    def apply_attention(self, queries, keys, values, mask=None):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size), mask)
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 05-13.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. 

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. 

        mask (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the mask. Default: None.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. 
        """
        weights = self.get_attention_weights(queries, keys, mask=mask)
        attended_values = weights @ values
        return self.merge_heads(attended_values)

    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        return tensor.reshape(*tensor.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3)

    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        return tensor.permute(0, 2, 1, 3).reshape(tensor.shape[0], tensor.shape[2], -1)

    def forward(self, queries, keys, values, mask=None):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V, mask) # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """
        queries = self.split_heads(self.w_q(queries))
        keys = self.split_heads(self.w_k(keys))
        values = self.split_heads(self.w_v(values))
        y = self.apply_attention(queries, keys, values, mask)
        return self.w_y(y)


# #### Position-wise FFN
# Defines a feedforward network, do not modify. You may just run it by clicking the run/play button.


class PositionwiseFeedForward(nn.Module):
    """
    Implements a position-wise FFN.
    """
    def __init__(self, d_model, hidden_size, use_dropout=False, p_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.linear2(x)
        return x


# #### EncoderLayer and DecoderLayer
# 
# Defines a layer of the Encoder or Decoder respectively, do not modify. You may just run it by clicking the run/play button.
# 
# The multi-head attention in the encoder attends over all tokens in the source strings (input for translation). The decoder is comprised of 2 multi-head attention steps: one to attend over the (allowed) tokens from the target sentence, and one to attend over the encoder outputs.


class EncoderLayer(nn.Module):
    """
    Defines a layer of the encoder.
    """
    def __init__(self, head_size, hidden_size, num_heads, use_dropout=False, p_dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(head_size, num_heads)
        self.norm1 = LayerNorm(head_size * num_heads)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout1 = nn.Dropout(p=p_dropout)

        self.ffn = PositionwiseFeedForward(head_size * num_heads, hidden_size, use_dropout, p_dropout)
        self.norm2 = LayerNorm(head_size * num_heads)
        if self.use_dropout:
            self.dropout2 = nn.Dropout(p=p_dropout)

    def forward(self, x, mask):
        x_ = x
        x = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + x_)
        if self.use_dropout:
            x = self.dropout1(x)
        x_ = x
        x = self.ffn(x)
        x = self.norm2(x + x_)
        if self.use_dropout:
            x = self.dropout2(x)
        return x

class DecoderLayer(nn.Module):
    """
    Defines a layer of the decoder.
    """
    def __init__(self, head_size, hidden_size, num_heads, use_dropout=False, p_dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(head_size, num_heads)
        self.norm1 = LayerNorm(head_size * num_heads)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout1 = nn.Dropout(p=p_dropout)

        self.encoder_decoder_attention = MultiHeadAttention(head_size, num_heads)
        self.norm2 = LayerNorm(head_size * num_heads)
        if self.use_dropout:
            self.dropout2 = nn.Dropout(p=p_dropout)

        self.ffn = PositionwiseFeedForward(head_size * num_heads, hidden_size, use_dropout, p_dropout)
        self.norm3 = LayerNorm(head_size * num_heads)
        if self.use_dropout:
            self.dropout3 = nn.Dropout(p=p_dropout)

    def forward(self, x_decoder, x_encoder, mask_target, mask_source):
        x_ = x_decoder
        x = self.self_attention(x_decoder, x_decoder, x_decoder, mask=mask_target)
        x = self.norm1(x + x_)
        if self.use_dropout:
            x = self.dropout1(x)

        if x_encoder is not None:
            x_ = x
            x = self.encoder_decoder_attention(x, x_encoder, x_encoder, mask=mask_source)
            x = self.norm2(x + x_)
            if self.use_dropout:
                x = self.dropout2(x)

        x_ = x
        x = self.ffn(x)
        x = self.norm3(x + x_)
        if self.use_dropout:
            x = self.dropout3(x)
        return x


# #### Encoder and Decoder
# 
# Defines the Encoder and Decoder of the Transformer. You will have to fill in the `forward` methods of both classes.


class TEncoder(nn.Module):
    """
    Defines the encoder of the Transformer model.
    """
    def __init__(self,
                 vocabulary_size=10838,
                 max_len=max_len,
                 head_size=64,
                 hidden_size=512,
                 num_heads=8,
                 num_layers=6,
                 use_dropout=False,
                 p_dropout=0.1,
                 device=device):
        super(TEncoder, self).__init__()

        self.embedding = TransformerEmbedding(d_model=head_size * num_heads,
                                              max_len=max_len,
                                              vocabulary_size=vocabulary_size,
                                              use_dropout=use_dropout,
                                              p_dropout=p_dropout,
                                              device=device)

        self.layers = nn.ModuleList([EncoderLayer(head_size=head_size,
                                                  hidden_size=hidden_size,
                                                  num_heads=num_heads,
                                                  use_dropout=use_dropout,
                                                  p_dropout=p_dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TDecoder(nn.Module):
    """
    Defines the decoder of the Transformer model.
    """
    def __init__(self,
                 vocabulary_size=11510,
                 max_len=max_len,
                 head_size=64,
                 hidden_size=512,
                 num_heads=8,
                 num_layers=6,
                 use_dropout=False,
                 p_dropout=0.1,
                 device=device):
        super(TDecoder, self).__init__()

        self.embedding = TransformerEmbedding(d_model=head_size * num_heads,
                                              max_len=max_len,
                                              vocabulary_size=vocabulary_size,
                                              use_dropout=use_dropout,
                                              p_dropout=p_dropout,
                                              device=device)

        self.layers = nn.ModuleList([DecoderLayer(head_size=head_size,
                                                  hidden_size=hidden_size,
                                                  num_heads=num_heads,
                                                  use_dropout=use_dropout,
                                                  p_dropout=p_dropout)
                                     for _ in range(num_layers)])

        self.linear = nn.Linear(head_size * num_heads, vocabulary_size)

    def forward(self, target, encoded_source, mask_target, mask_source):
        trg = self.embedding(target)
        for layer in self.layers:
            trg = layer(trg, encoded_source, mask_target, mask_source)
        output = self.linear(trg)
        return output


# #### Transformer
# 
# This class defines the encoder-decoder Transformer model, do not modify. You may just run the cell by clicking the run/play button.


class Transformer(nn.Module):
    """
    Implements an encoder-decoder transformer.
    """
    def __init__(self,
                 enc_vocabulary_size=10838,
                 dec_vocabulary_size=11510,
                 head_size=64,
                 num_heads=8,
                 max_len=max_len,
                 hidden_size=512,
                 num_layers=6,
                 use_dropout=False,
                 p_dropout=0.1,
                 strategy="greedy",
                 device=device):
        super(Transformer, self).__init__()

        self.pad_idx = special_idx["pad"]
        self.sos_idx = special_idx["bos"]
        self.strategy = strategy
        self.device = device
        self.encoder = TEncoder(head_size=head_size,
                                num_heads=num_heads,
                                max_len=max_len,
                                hidden_size=hidden_size,
                                vocabulary_size=enc_vocabulary_size,
                                use_dropout=use_dropout,
                                p_dropout=p_dropout,
                                num_layers=num_layers,
                                device=device)

        self.decoder = TDecoder(head_size=head_size,
                                num_heads=num_heads,
                                max_len=max_len,
                                hidden_size=hidden_size,
                                vocabulary_size=dec_vocabulary_size,
                                use_dropout=use_dropout,
                                p_dropout=p_dropout,
                                num_layers=num_layers,
                                device=device)

    def forward(self, source, target):
        mask_source = self.make_pad_mask(source, source)
        mask_source_target = self.make_pad_mask(target, source)
        mask_target = self.make_pad_mask(target, target) * self.make_no_peak_mask(target, target)

        encoded_source = self.encoder(source, mask_source)
        output = self.decoder(target, encoded_source, mask_target, mask_source_target)
        return output

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        k = k.ne(self.pad_idx).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        q = q.ne(self.pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask


# #### Training


def train_transformer(model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0.
    train_ppl = 0.
    start = time.time()

    for i, (source, target) in enumerate(dataloader):
        source, target = source.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(source, target[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        target = target[:, 1:].contiguous().view(-1)
        loss = criterion(output_reshape, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    end = time.time()
    train_loss /= len(dataloader)
    train_ppl = math.exp(train_loss)
    train_time = end - start
    return train_loss, train_ppl, train_time

def validate_transformer(model, dataloader, criterion):
    model.eval()
    val_loss = 0.
    val_ppl = 0.
    start = time.time()

    with torch.no_grad():
        for i, (source, target) in enumerate(dataloader):
            source, target = source.to(device), target.to(device)
            output = model(source, target[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            target = target[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, target)
            val_loss += loss.item()

    end = time.time()
    val_loss /= len(dataloader)
    val_ppl = math.exp(val_loss)
    val_time = end - start
    return val_loss, val_ppl, val_time

def compute_bleu(model, dataloader):
    targets = []
    translations = []

    for source, target in dataloader:
        source, target = source.to(device), target.to(device)

        output = model(source, target)
        for i in range(target.shape[0]):
            targets.append([[]])
            for token in target[i, 1:]:
                if token in special_idx.values():
                    break
                targets[-1][0].append(fr_vocabulary.get_itos()[token])
            translations.append([])

            if model.strategy == "greedy":
                # WRITE CODE HERE
                # Assign the greedy choice tokens for output[i] to o
                # output[i] has shape (sequence_length, vocabulary_size)
                # o should have shape (sequence_length)
                o = output[i].argmax(dim=1)
            elif model.strategy == "random":
                # WRITE CODE HERE
                # Assign the randomly sampled tokens for output[i] to o
                # output[i] has shape (sequence_length, vocabulary_size)
                # o should have shape (sequence_length)
                temperature = 0.5
                p = F.softmax(output[i] / temperature, dim=1)
                d = torch.multinomial(p, 1)
                o = d.squeeze().to(device)

            for token in o:
                if token in special_idx.values():
                    break
                translations[-1].append(fr_vocabulary.get_itos()[token.int().item()])
                
        break

    return torchtext.data.metrics.bleu_score(translations, targets, max_n=2, weights=[0.5]*2) * 100,             torchtext.data.metrics.bleu_score(translations, targets, max_n=1, weights=[1]) * 100


fix_experiment_seed()
use_dropout = False
strategy = "greedy"
model = Transformer(head_size=64,
                    enc_vocabulary_size=10838,
                    dec_vocabulary_size=11510,
                    max_len=max_len,
                    hidden_size=512,
                    num_heads=8,
                    num_layers=6,
                    use_dropout=use_dropout,
                    p_dropout=0.1,
                    strategy=strategy,
                    device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train_transformer(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate_transformer(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate_transformer(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"transformer-adam-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)


fix_experiment_seed()
use_dropout = True
strategy = "greedy"
model = Transformer(head_size=64,
                    enc_vocabulary_size=10838,
                    dec_vocabulary_size=11510,
                    max_len=max_len,
                    hidden_size=512,
                    num_heads=8,
                    num_layers=6,
                    use_dropout=use_dropout,
                    p_dropout=0.1,
                    strategy=strategy,
                    device=device).to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train_transformer(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate_transformer(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate_transformer(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"transformer-sgd-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)


fix_experiment_seed()
use_dropout = True
strategy = "greedy"
model = Transformer(head_size=64,
                    enc_vocabulary_size=10838,
                    dec_vocabulary_size=11510,
                    max_len=max_len,
                    hidden_size=512,
                    num_heads=8,
                    num_layers=6,
                    use_dropout=use_dropout,
                    p_dropout=0.1,
                    strategy=strategy,
                    device=device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train_transformer(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate_transformer(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate_transformer(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"transformer-adam-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)


fix_experiment_seed()
use_dropout = True
strategy = "random"
model = Transformer(head_size=64,
                    enc_vocabulary_size=10838,
                    dec_vocabulary_size=11510,
                    max_len=max_len,
                    hidden_size=512,
                    num_heads=8,
                    num_layers=6,
                    use_dropout=use_dropout,
                    p_dropout=0.1,
                    strategy=strategy,
                    device=device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = get_criterion()

results = {
    "train_losses": [],
    "train_ppls":   [],
    "train_times":  [],
    "valid_losses": [],
    "valid_ppls":   [],
    "valid_times":  [],
    "test_loss":    0.,
    "test_ppl":     0.,
    "test_bleu1":   0.,
    "test_bleu2":   0.,
    "test_time":    0.
}

if __name__ == "__main__":
    n_epochs = 10

    for epoch in range(n_epochs):
        train_loss, train_ppl, train_time = train_transformer(model, train_loader, optimizer, criterion)
        valid_loss, valid_ppl, valid_time = validate_transformer(model, val_loader, criterion)

        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Train Time: {train_time:.3f}")
        print(f"Valid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f} | Valid Time: {valid_time:.3f}")

        results["train_losses"].append(train_loss)
        results["train_ppls"].append(train_ppl)
        results["train_times"].append(train_time)
        results["valid_losses"].append(valid_loss)
        results["valid_ppls"].append(valid_ppl)
        results["valid_times"].append(valid_time)

    test_loss, test_ppl, test_time = validate_transformer(model, test_loader, criterion)
    print(f"== Test Loss: {test_loss:.3f} | Test PPL: {test_ppl:7.3f} | Test Time: {test_time:.3f} ==")
    results["test_loss"] = test_loss
    results["test_ppl"] = test_ppl
    results["test_time"] = test_time

    results["test_bleu1"], results["test_bleu2"] = compute_bleu(model, test_loader)
    print("Test BLEU-1:", results["test_bleu1"], "Test BLEU-2:", results["test_bleu2"])

    json.dump(results, open(f"transformer-adam-dropout_{str(use_dropout)}-strategy_{strategy}.json", "w"), indent=4)
