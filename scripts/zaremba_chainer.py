#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Implements the small model from Zaremba (2014). Its main purpose is to
reproduce the numbers exactly.
"""

import argparse
import math
import sys
import time

import numpy as np
import chainer
from chainer import Chain
import chainer.links as L
import chainer.functions as F

from pytorch_lm.data import Corpus
from pytorch_lm.lstm_chainer import Lstm
from pytorch_lm.loss import SequenceLoss


class SmallZarembaModel(Chain):
    """"Implements the small model from Zaremba (2014)."""
    def __init__(self, vocab_size):
        super(SmallZarembaModel, self).__init__()
        self.hidden_size = 200
        self.input_size = 200
        self.num_layers = 2

        self.encoder = L.EmbedID(vocab_size, self.input_size)
        self.rnn = Lstm(self.input_size, self.hidden_size, self.num_layers)
        self.decoder = L.Linear(self.hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        uniform = chainer.initializers.Uniform(initrange)
        uniform(self.encoder.W.data)
        uniform(self.decoder.W.data)
        uniform(self.decoder.b.data)

    def __call__(self, input, hidden, trace=False):
        # print('INPUT', input.size())
        emb = self.encoder(input)
        # print('EMB', emb.size())
        # self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(F.reshape(output, (-1, output.shape[2])))
        if trace:
            print('EMB', F.copy(emb, -1).data)
            print('RNN_OUT', F.copy(output, -1).data)
        return (
            F.reshape(decoded, (output.shape[0], output.shape[1], decoded.shape[1])),
            hidden
        )

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)

    def save_parameters(self, out_dict=None, prefix=''):
        if out_dict is None:
            out_dict = {}
        self.rnn.save_parameters(out_dict, prefix=prefix + 'RNN/')
        out_dict[prefix + 'embedding'] = F.copy(self.encoder.W, -1).data
        # .T is required because stupid Linear stores the weights transposed
        out_dict[prefix + 'softmax_w'] = F.copy(self.decoder.W, -1).data.T
        out_dict[prefix + 'softmax_b'] = F.copy(self.decoder.b, -1).data
        return out_dict

    def load_parameters(self, data_dict, prefix=''):
        def set_data(parameter, value, is_cuda):
            t = torch.from_numpy(value)
            if is_cuda:
                t = t.cuda()
            parameter.data = t
        device_id = self._device_id if self._device_id is not None else -1

        self.rnn.load_parameters(data_dict, prefix=prefix + 'RNN/')
        self.encoder.W.data = F.copy(data_dict[prefix + 'embedding'], device_id).data
        # .T is required because stupid Linear stores the weights transposed
        self.decoder.W.data = F.copy(data_dict[prefix + 'softmax_w'], device_id).data.T
        self.decoder.b.data = F.copy(data_dict[prefix + 'softmax_b'], device_id).data
