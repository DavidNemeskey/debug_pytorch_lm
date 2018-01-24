#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Implements a very basic version of LSTM."""
import tensorflow as tf

class LstmCell(object):
    def __init__(self, input_size, hidden_size, batch_size, bias=True):
        super(LstmCell, self).__init__()
        # TODO: add forget_bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bias = bias

        self.w_i = tf.get_variable('w_i', (input_size, 4 * hidden_size))
        self.w_h = tf.get_variable('w_h', (hidden_size, 4 * hidden_size))
        self.weights = [self.w_i, self.w_h]

        if bias:
            self.b_i = tf.get_variable('b_i', (4 * hidden_size))
            self.b_h = tf.get_variable('b_h', (4 * hidden_size))
            self.weights += [self.b_i, self.b_h]

    def save_parameters(self, session, out_dict=None):
        if out_dict is None:
            out_dict = {}
        for w in self.weights:
            out_dict[w.name.rsplit(':', 1)[0]] = session.run([w])[0]
        return out_dict

    def load_parameters(self, session, data_dict):
        """Loads the parameters saved by save_parameters()."""
        for w in self.weights:
            name = w.name.rsplit(':', 1)[0]
            if name in data_dict:
                session.run(w.assign(data_dict[name]))

    def __call__(self, input, hidden):
        h_t, c_t = hidden

        ifgo = tf.matmul(input, self.w_i) + tf.matmul(h_t, self.w_h)

        if self.bias:
            ifgo += self.b_i + self.b_h
        i = tf.sigmoid(ifgo[:, :self.hidden_size])
        f = tf.sigmoid(ifgo[:, self.hidden_size:2*self.hidden_size])
        g = tf.tanh(ifgo[:, 2*self.hidden_size:3*self.hidden_size])
        o = tf.sigmoid(ifgo[:, 3*self.hidden_size:])
        c_t = f * c_t + i * g
        h_t = o * tf.tanh(c_t)

        return h_t, c_t

    def init_hidden(self):
        return (
            tf.zeros([self.batch_size, self.hidden_size]),
            tf.zeros([self.batch_size, self.hidden_size])
        )
