#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Implements a very basic version of LSTM."""
import chainer
from chainer import Link, Chain, ChainList
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L

class LstmCell(Chain):
    """
    A reimplementation of the LSTM cell, for comparison with the Pytorch and
    TensorFlow implementations.

    As a reminder: input size is batch_size x input_features.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LstmCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        with self.init_scope():
            self.w_i = chainer.Parameter(
                shape=(input_size, 4 * hidden_size), name='w_i')
            self.w_h = chainer.Parameter(
                shape=(hidden_size, 4 * hidden_size), name='w_h')

            if bias:
                self.b_i = chainer.Parameter(shape=(4 * hidden_size))
                self.b_h = chainer.Parameter(shape=(4 * hidden_size))

            self.reset_parameters()

    def reset_parameters(self, initrange=0.1):
        """Initializes the parameters uniformly to between -/+ initrange."""
        uniform = chainer.initializers.Uniform(initrange)
        for weight in self.params():
            uniform(weight.data)

    def save_parameters(self, out_dict=None, prefix=''):
        """
        Saves the parameters into a dictionary that can later be e.g. savez'd.
        If prefix is specified, it is prepended to the names of the parameters,
        allowing for hierarchical saving / loading of parameters of a composite
        model.
        """
        if out_dict is None:
            out_dict = {}
        for name, p in self.namedparams():
            # The name starts with a /
            out_dict[prefix + name[1:]] = F.copy(p, -1).data
        return out_dict

    def load_parameters(self, data_dict, prefix=''):
        """Loads the parameters saved by save_parameters()."""
        device_id = self._device_id if self._device_id is not None else -1
        for name, value in data_dict.items():
            real_name = name[len(prefix):]
            param = getattr(self, real_name)
            param.data = F.copy(value, device_id).data

    def __call__(self, input, hidden):
        h_t, c_t = hidden

        ifgo = F.matmul(input, self.w_i) + F.matmul(h_t, self.w_h)

        if self.bias:
            print('SHAPES', ifgo.shape, self.b_i.shape, self.b_h.shape)
            ifgo += F.broadcast_to(self.b_i + self.b_h, shape=ifgo.shape)

        i = F.sigmoid(ifgo[:, :self.hidden_size])
        f = F.sigmoid(ifgo[:, self.hidden_size:2*self.hidden_size])
        g = F.tanh(ifgo[:, 2*self.hidden_size:3*self.hidden_size])
        o = F.sigmoid(ifgo[:, 3*self.hidden_size:])
        c_t = f * c_t + i * g
        h_t = o * F.tanh(c_t)

        return h_t, c_t

    def init_hidden(self, batch_size=0, np_arrays=None):
        """
        Returns the Variables for the hidden states. If batch_size is specified,
        all states are initialized to zero. If np_arrays is, it should be a
        2-tuple of numpy arrays, which are wrapped in Variables.
        """
        if batch_size and np_arrays:
            raise ValueError('Only one of {batch_size, np_arrays) is allowed.')
        if not batch_size and not np_arrays:
            raise ValueError('Either batch_size or np_arrays must be specified.')
        xp = self.xp  # numpy or cupy
        with cuda.get_device_from_id(self._device_id):
            if batch_size != 0:
                ret = (chainer.Variable(xp.zeros((batch_size, self.hidden_size),
                                                 dtype=self.w_i.dtype)),
                       chainer.Variable(xp.zeros((batch_size, self.hidden_size),
                                                 dtype=self.w_i.dtype)))
            else:
                ret = (chainer.Variable(np_arrays[0]),
                       chainer.Variable(np_arrays[0]))
        return ret
