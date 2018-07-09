'''
Train a meta-learner on SNLI (try the one from the blog post) (so it learns to update from SNLI 
-- 400k), and then use the meta-learner to update a LSTM on SST (which has 40k data).
Author: Yuhui Zhang
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
import numpy as np

TEXT = data.Field(sequential=True, include_lengths=True)
LABEL = data.Field(sequential=False)

train, val, test = datasets.SNLI.splits(TEXT, LABEL)

TEXT.build_vocab(train, vectors="glove.840B.300d")
vocab = TEXT.vocab
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), 
    # sort_key=lambda x: len(x.premise) + len(x.hypothesis), 
    batch_size=32,
    # sort_within_batch=True, 
    repeat=False)

class Config:
    emb_dim = 300
    hidden_size = 300
    depth = 1
    dropout = 0.3
    bidir = False
    n_labels = 3
    emb_update = False
    lr = 1e-3
    batch_size = 128


class Model(nn.Module):
    def __init__(self, vocab, config):
        super(Model, self).__init__()
        self.config = config
        self.encoder = nn.LSTM(
            config.emb_dim,
            config.hidden_size,
            config.depth,
            # dropout=config.dropout,
            bidirectional=config.bidir
        )
        self.out = nn.Linear(config.hidden_size, config.n_labels)
        self.fc1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False
    
    def forward(self, input):
        input_premise, length_premise = input.premise
        input_hypothesis, length_hypothesis = input.hypothesis
        embed_premise = self.embed(input_premise)
        embed_hypothesis = self.embed(input_hypothesis)
        output_premise = self.autolen_rnn(embed_premise, length_premise)
        output_hypothesis = self.autolen_rnn(embed_hypothesis, length_hypothesis)
        output = torch.cat([output_premise, output_hypothesis,
                            output_premise - output_hypothesis,
                            output_premise * output_hypothesis], dim=1)
        output = F.tanh(self.fc1(output))
        output = self.out(output)
        return output

    def autolen_rnn(self, inputs, lengths):
        _, idx = lengths.sort(0, descending=True)
        _, revidx = idx.sort(0, descending=False)
        packed_emb = nn.utils.rnn.pack_padded_sequence(inputs[:, idx, :], lengths[idx])
        output, (h, c) = self.encoder(packed_emb)
        output = h[0, revidx, :]
        # The following code work the same but is more difficult, we use hidden state as final output
        # output = nn.utils.rnn.pad_packed_sequence(output)[0]
        # output = output[:, idx, :]
        # output = output[lengths - 1, torch.range(0, len(lengths) - 1).type(torch.LongTensor), :]
        return output


config = Config()
model = Model(vocab, config) 
print model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad is True], lr=config.lr)
cnt = 0
for epoch in range(1):
    for batch in train_iter:

        # print(batch.fields)
        # print((batch.premise[0]).shape)
        # print(batch.hypothesis)
        
        outputs = model(batch)
        labels = batch.label - 1
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc = (outputs.max(1)[1] == labels).type(torch.FloatTensor).mean()

        if cnt % 100 == 0: print cnt, loss, acc
        cnt += 1

print cnt



