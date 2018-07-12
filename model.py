import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)
    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)

class CNNModel(nn.Module):
    def __init__(self, vocab, config):
        super(CNNModel, self).__init__()
        self.config = config
        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False
        self.conv = nn.Conv2d(in_channels=1, out_channels=config.filter_num, kernel_size=(config.kernel_size, config.emb_dim), stride=(1, config.emb_dim))

        self.fc1 = nn.Linear(4 * config.filter_num, config.filter_num)
        self.fc2 = nn.Linear(config.filter_num, config.n_labels)
        
    def forward(self, input):
        input_premise, length_premise = input.premise
        input_hypothesis, length_hypothesis = input.hypothesis
        embed_premise = self.embed(input_premise)
        embed_hypothesis = self.embed(input_hypothesis)
        output_premise = self.autolen_cnn(embed_premise)
        output_hypothesis = self.autolen_cnn(embed_hypothesis)
        output = torch.cat([output_premise, output_hypothesis,
                            output_premise - output_hypothesis,
                            output_premise * output_hypothesis], dim=1)
        output = F.tanh(self.fc1(output))
        output = self.fc2(output)
        return output

    def autolen_cnn(self, inputs):
        output = inputs.transpose(0, 1).unsqueeze(1) # [batch_size, in_kernel, seq_length, embed_dim]
        output = F.relu(self.conv(output)) # conv -> [batch_size, out_kernel, seq_length, 1]
        output = output.squeeze(3).max(2)[0] # max_pool -> [batch_size, out_kernel]
        return output

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
        # self.encoder.flatten_parameters()
        output, (h, c) = self.encoder(packed_emb)
        output = h[0, revidx, :]
        return output