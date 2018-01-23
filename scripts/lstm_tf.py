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
            out_dict[w.name.rsplit(':', 1)[0]] = session.run([w])
        return out_dict

    def load_parameters(self, session, data_dict):
        """Loads the parameters saved by save_parameters()."""
        for w in self.weights:
            name = w.name.rsplit(':', 1)[0]
            print('name', name)
            if name in data_dict:
                print('in dict')
                session.run(w.assign(value))

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


def test_cell():
    """Tests the LstmCell class."""
    import numpy as np

    input_size, hidden_size = 3, 2
    batch_size = 4

    with tf.Graph().as_default() as graph:
        input_data = tf.placeholder(tf.float32, [batch_size, input_size], 'input_data')
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope('Model', initializer=initializer):
            lstm_cell = LstmCell(input_size, hidden_size, batch_size)
        init_state = lstm_cell.init_hidden()
        final_state = lstm_cell(input_data, init_state)

        init = tf.global_variables_initializer()

    input_np = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]], dtype=np.float32)
    print('input', input_np)
    with tf.Session(graph=graph) as session:
        session.run(init)
        hidden = session.run(init_state)
        print('hidden', hidden)
        fetches = [final_state]
        feed_dict = {input_data: input_np,
                     init_state: hidden}
        final_state = session.run(fetches, feed_dict)
        print(final_state)
