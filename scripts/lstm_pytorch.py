#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Implements a very basic version of LSTM."""
import torch
import torch.nn as nn


class LstmCell(nn.Module):
    """
    A reimplementation of the LSTM cell. It takes 176s to run the time
    sequence prediction example; the built-in LSTMCell takes 133s. So it's
    slower, but at least transparent.

    As a reminder: input size is batch_size x input_features.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LstmCell, self).__init__()
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
