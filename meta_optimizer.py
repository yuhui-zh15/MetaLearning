from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from utils import preprocess_gradients
from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D

class MetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.meta_model = model

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(3, hidden_size)
        self.ln1 = LayerNorm1D(hidden_size)

        # self.lstms = []
        # for i in range(num_layers):
        #     self.lstms.append(LayerNormLSTMCell(hidden_size, hidden_size))

        self.lstms = LayerNormLSTMCell(hidden_size, hidden_size)

        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)


    def cuda(self):
        super(MetaOptimizer, self).cuda()
        # for i in range(len(self.lstms)):
        #     self.lstms[i].cuda()

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            # for i in range(len(self.lstms)):
            #     self.hx[i] = Variable(self.hx[i].data)
            #     self.cx[i] = Variable(self.cx[i].data)
            self.hx = Variable(self.hx.data)
            self.cx = Variable(self.cx.data)
        else:
            # self.hx = []
            # self.cx = []
            # for i in range(len(self.lstms)):
            #     self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
            #     self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
            #     if use_cuda:
            #         self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()
            self.hx = Variable(torch.zeros(1, self.hidden_size))
            self.cx = Variable(torch.zeros(1, self.hidden_size))
            if use_cuda:
                self.hx, self.cx = self.hx.cuda(), self.cx.cuda()

    def forward(self, x):
        # Gradients preprocessing
        x = F.tanh(self.ln1(self.linear1(x)))

        # for i in range(len(self.lstms)):
        #     if x.size(0) != self.hx[i].size(0):
        #         self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
        #         self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

        #     self.hx[i], self.cx[i] = self.lstms[i](x, (self.hx[i], self.cx[i]))
        #     x = self.hx[i]

        if x.size(0) != self.hx.size(0):
            self.hx = self.hx.expand(x.size(0), self.hx.size(1))
            self.cx = self.cx.expand(x.size(0), self.cx.size(1))

        self.hx, self.cx = self.lstms(x, (self.hx, self.cx))
        x = self.hx

        x = self.linear2(x)
        return x.squeeze()

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            if type(module) == torch.nn.modules.sparse.Embedding: continue
            for name, p in module._parameters.items():
                grads.append(p.grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = preprocess_gradients(torch.cat(grads))

        inputs = Variable(torch.cat((flat_grads, flat_params.data.unsqueeze(1)), 1))

        # Meta update itself
        flat_params = flat_params + self(inputs)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

class FastMetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(FastMetaOptimizer, self).__init__()
        self.meta_model = model

        self.linear1 = nn.Linear(6, 2)
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        # Gradients preprocessing
        x = F.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))
            if use_cuda:
                self.f = self.f.cuda()
                self.i = self.i.cuda()

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            if type(module) == torch.nn.modules.sparse.Embedding: continue
            for name, p in module._parameters.items():
                grads.append(p.grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = torch.cat(grads)

        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads), flat_params.data.unsqueeze(1), loss.unsqueeze(1)), 1))
        inputs = torch.cat((inputs, self.f, self.i), 1)
        self.f, self.i = self(inputs)

        # Meta update itself
        flat_params = torch.t(self.f) * flat_params - torch.t(self.i) * Variable(flat_grads)
        flat_params = flat_params.view(-1)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.


class MetaModel:

    def __init__(self, model):
        self.model = model

    def reset(self):
        for module in self.model.children():
            if type(module) == torch.nn.modules.sparse.Embedding: continue
            for name, p in module._parameters.items():
                module._parameters[name] = Variable(p.data)

    def get_flat_params(self):
        params = []

        for module in self.model.children():
            if type(module) == torch.nn.modules.sparse.Embedding: continue
            for name, p in module._parameters.items():
                params.append(p.view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for i, module in enumerate(self.model.children()):
            if type(module) == torch.nn.modules.sparse.Embedding: continue
            for name, p in module._parameters.items():
                p_shape = p.size()
                p_flat_size = reduce(mul, p_shape, 1)
                module._parameters[name] = flat_params[offset: offset + p_flat_size].view(*p_shape)
                offset += p_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
