#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Implements a very basic version of LSTM."""
import torch
import torch.nn as nn
from torch.autograd import Variable


class LstmCell(nn.Module):
    """
    A reimplementation of the LSTM cell. It takes 176s to run the time
    sequence prediction example; the built-in LSTMCell takes 133s. So it's
    slower, but at least transparent.

    As a reminder: input size is batch_size x input_features.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LstmCell, self).__init__()
        # TODO: add forget_bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_i = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.w_h = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))

        if bias:
            self.b_i = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self, initrange=0.1):
        """Initializes the parameters uniformly to between -/+ initrange."""
        for weight in self.parameters():
            weight.data.uniform_(-initrange, initrange)

    def save_parameters(self, out_dict=None, prefix=''):
        """
        Saves the parameters into a dictionary that can later be e.g. savez'd.
        If prefix is specified, it is prepended to the names of the parameters,
        allowing for hierarchical saving / loading of parameters of a composite
        model.
        """
        if out_dict is None:
            out_dict = {}
        for name, p in self.named_parameters():
            out_dict[prefix + name] = p.data.cpu().numpy()
        return out_dict

    def load_parameters(self, data_dict, prefix=''):
        """Loads the parameters saved by save_parameters()."""
        for name, value in data_dict.items():
            real_name = name[len(prefix):]
            setattr(self, real_name, nn.Parameter(torch.from_numpy(value)))

    def forward(self, input, hidden):
        h_t, c_t = hidden

        ifgo = input.matmul(self.w_i) + h_t.matmul(self.w_h)

        if self.bias:
            ifgo += self.b_i + self.b_h

        i, f, g, o = ifgo.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t

    def init_hidden(self, batch_size):
        return (Variable(torch.Tensor(batch_size, self.hidden_size).zero_()),
                Variable(torch.Tensor(batch_size, self.hidden_size).zero_()))


class Lstm(nn.Module):
    """
    Several layers of LstmCells. Input is batch_size x num_steps x input_size,
    which is different from the Pytorch LSTM (the first two dimensions are
    swapped).
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = [LstmCell(input_size, hidden_size)
                       for _ in range(num_layers)]

    def forward(self, input, hidden):
        outputs = []
        for input_t in input.chunk(input.size(1), dim=1):
            values = input_t.squeeze(1)  # From input to output
            for i in range(self.num_layers):
                h_t, c_t = self.layers[i](values, hidden[i])
                values = h_t
                hidden[i] = h_t, c_t
            outputs.append(values)
        outputs = torch.stack(outputs, 1)
        return outputs, hidden

    def init_hidden(self, batch_size):
        return [(Variable(torch.Tensor(batch_size, self.hidden_size).zero_()),
                 Variable(torch.Tensor(batch_size, self.hidden_size).zero_()))
                for _ in range(self.num_layers)]
