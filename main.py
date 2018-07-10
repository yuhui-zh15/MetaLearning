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

class MetaLearner(nn.Module):
    """ Bare Meta-learner class
        Should be added: intialization, hidden states, more control over everything
    """
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([[0.001, 0]]))

    def forward(self, forward_model, backward_model):
        """ Forward optimizer with a simple linear neural net
        Inputs:
            forward_model: PyTorch module with parameters gradient populated
            backward_model: PyTorch module identical to forward_model (but without gradients)
              updated at the Parameter level to keep track of the computation graph for meta-backward pass
        """
        f_model_iter = get_params(forward_model)
        b_model_iter = get_params(backward_model)
        for f_param_tuple, b_param_tuple in zip(f_model_iter, b_model_iter): # loop over parameters
            # Prepare the inputs, we detach the inputs to avoid computing 2nd derivatives (re-pack in new Variable)
            (module_f, name_f, param_f) = f_param_tuple
            (module_b, name_b, param_b) = b_param_tuple
            inputs = Variable(torch.stack([param_f.grad.data, param_f.data], dim=-1))
            # print self.weights
            # Optimization step: compute new model parameters, here we apply a simple linear function
            dW = F.linear(inputs, self.weights).squeeze()
            param_b = param_b + dW
            # Update backward_model (meta-gradients can flow) and forward_model (no need for meta-gradients).
            module_b._parameters[name_b] = param_b
            param_f.data = param_b.data

def get_params(module, memo=None, pointers=None):
    """ Returns an iterator over PyTorch module parameters that allows to update parameters
        (and not only the data).
    ! Side effect: update shared parameters to point to the first yield instance
        (i.e. you can update shared parameters and keep them shared)
    Yields:
        (Module, string, Parameter): Tuple containing the parameter's module, name and pointer
    """
    if memo is None:
        memo = set()
        pointers = {}
    for name, p in module._parameters.items():
        if p.requires_grad is False: continue
        if p not in memo:
            memo.add(p)
            pointers[p] = (module, name)
            yield module, name, p
        elif p is not None:
            prev_module, prev_name = pointers[p]
            module._parameters[name] = prev_module._parameters[prev_name] # update shared parameter pointer
    for child_module in module.children():
        for m, n, p in get_params(child_module, memo, pointers):
            yield m, n, p

def train(forward_model, backward_model, optimizer, meta_optimizer, train_data, meta_epochs, loss_func):
    """ Train a meta-learner
    Inputs:
      forward_model, backward_model: Two identical PyTorch modules (can have shared Tensors)
      optimizer: a neural net to be used as optimizer (an instance of the MetaLearner class)
      meta_optimizer: an optimizer for the optimizer neural net, e.g. ADAM
      train_data: an iterator over an epoch of training data
      meta_epochs: meta-training steps
    To be added: intialization, early stopping, checkpointing, more control over everything
    """
    for meta_epoch in range(meta_epochs): # Meta-training loop (train the optimizer)
        optimizer.zero_grad()
        losses = []
        acces = []
        cnt = 0
        for batch in train_data:   # Meta-forward pass (train the model)
            forward_model.zero_grad()         # Forward pass
            output = forward_model(batch)
            label = batch.label - 1
            loss = loss_func(output, label)  # Compute loss
            losses.append(loss)
            acces.append((output.max(1)[1] == label).type(torch.FloatTensor).mean())
            loss.backward(retain_graph=True)                   # Backward pass to add gradients to the forward_model
            optimizer(forward_model,          # Optimizer step (update the models)
                      backward_model)
            
            cnt += 1
            if cnt % 100 == 0: 
                meta_loss = sum(losses)             # Compute a simple meta-loss
                meta_loss.backward()                # Meta-backward pass
                meta_optimizer.step()              # Meta-optimizer step
                print cnt, meta_loss / len(losses), sum(acces) / len(acces), optimizer.weights
                losses = []
                acces = []

config = Config()
forward_model = Model(vocab, config) 
backward_model = Model(vocab, config)
optimizer = MetaLearner()
# meta_optimizer = optim.Adam([param for param in optimizer.parameters() if param.requires_grad is True], lr=config.lr)
print 'we are trying to optimize ', [param for param in optimizer.parameters() if param.requires_grad is True]
meta_optimizer = optim.Adam([param for param in optimizer.parameters() if param.requires_grad is True], lr=config.lr)
criterion = nn.CrossEntropyLoss()
train(forward_model, backward_model, optimizer, meta_optimizer, train_iter, 1, criterion)

# print model

# optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad is True], lr=config.lr)
# cnt = 0
# for epoch in range(1):
#     for batch in train_iter:

#         # print(batch.fields)
#         # print((batch.premise[0]).shape)
#         # print(batch.hypothesis)
        
#         outputs = model(batch)
#         labels = batch.label - 1
#         optimizer.zero_grad()
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         acc = (outputs.max(1)[1] == labels).type(torch.FloatTensor).mean()

#         if cnt % 100 == 0: print cnt, loss, acc
#         cnt += 1

# print cnt



