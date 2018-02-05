#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import contextlib
import os
import unittest

import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable

import pytorch_lm.lstm_pytorch as pti
import pytorch_lm.lstm_tf as tfi


class TestLstms(unittest.TestCase):
    """Tests that the two LSTM cell implementations work alike."""
    def setUp(self):
        self.input_size = 2
        self.hidden_size = 3
        self.batch_size = 2
        self.num_layers = 2

        weight_file = os.path.join(os.path.dirname(__file__), 'lstm.npz')
        if not os.path.isfile(weight_file):
            pti_lstm = pti.Lstm(self.input_size, self.hidden_size, self.num_layers)
            pti_d = pti_lstm.save_parameters(prefix='Lstm/')
            np.savez(weight_file, **pti_d)
        self.weights = dict(np.load(weight_file))

    @contextlib.contextmanager
    def __create_lstms(self, input_size=None, hidden_size=None,
                       num_layers=None, batch_size=None):
        """
        Helper method. Creates three objects:
        - a pytorch Lstm
        - a tensorflow Lstm
        - a session for the latter.
        """
        input_size = input_size or self.input_size
        hidden_size = hidden_size or self.hidden_size
        num_layers = num_layers or self.num_layers
        batch_size = batch_size or self.batch_size
        pti_lstm = pti.Lstm(input_size, hidden_size, num_layers)

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Lstm', initializer=initializer):
                tfi_lstm = tfi.Lstm(input_size, hidden_size, num_layers, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)
            yield pti_lstm, tfi_lstm, session

    def __assert_parameters_equals(self, pti_lstm, tfi_lstm, session):
        """Checks if the state vectors of the two LSTM networks are the same."""
        equals = []
        for l, pti_cell in enumerate(pti_lstm.layers):
            tfi_cell = tfi_lstm.layers[l]
            for name, _ in pti_lstm.named_parameters():
                pti_value = getattr(pti_cell, name).data.cpu().numpy()
                tfi_value = session.run(getattr(tfi_cell, name))
                equals.append(np.allclose(pti_value, tfi_value))
            self.assertTrue(all(equals))

    def test_data_from_pytorch_to_tf(self):
        """Tests data transfer from the pytorch LSTM to the tf one."""
        with self.__create_lstms(batch_size=4) as (pti_lstm, tfi_lstm, session):
            pti_d = pti_lstm.save_parameters(prefix='Lstm/')
            tfi_lstm.load_parameters(session, pti_d)

            self.__assert_parameters_equals(pti_lstm, tfi_lstm, session)

    def test_data_from_tf_to_pytorch(self):
        """Tests data transfer from the tf LSTM to the pytorch one."""
        with self.__create_lstms(batch_size=4) as (pti_lstm, tfi_lstm, session):
            tfi_d = tfi_lstm.save_parameters(session)
            pti_lstm.load_parameters(tfi_d, prefix='Lstm/')

            self.__assert_parameters_equals(pti_lstm, tfi_lstm, session)

    def test_sequence_tagging(self):
        """Tests sequence tagging (i.e. the output)."""
        with self.__create_lstms() as (pti_lstm, tfi_lstm, session):
            tfi_lstm.load_parameters(session, self.weights)
            pti_lstm.load_parameters(self.weights, prefix='Lstm/')

            # Input
            input_np = np.array(
                [
                    [[1, 2], [2, 4], [2, 3]],
                    [[1, 1], [2, 2], [3, 3]]
                ],
                dtype=np.float32
            )
            pti_input = Variable(torch.FloatTensor(input_np))
            tfi_input = tf.placeholder(tf.float32, input_np.shape)
            # Target (arithmetic mean)
            target_np = np.array([[1.5, 3, 2.5], [1, 2, 3]], dtype=np.float32)
            pti_target = Variable(torch.FloatTensor(target_np))
            tfi_target = tf.placeholder(tf.float32, target_np.shape)

            # Initial states
            pti_hidden = pti_lstm.init_hidden(2)
            tfi_init_state = tfi_lstm.init_hidden()
            tfi_output, tfi_final_state = tfi_lstm(tfi_input, tfi_init_state)

            # Pytorch
            pti_lstm.zero_grad()
            pti_output, pti_final_state = pti_lstm.forward(pti_input, pti_hidden)
            print(pti_target.size(), pti_output.size())
#            pti_loss = (pti_h - Variable(torch.FloatTensor([0, 1]))).norm(2)
#            pti_loss.backward(retain_graph=True)
#            pti_h_np, pti_c_np = (v.data.cpu().numpy() for v in (pti_h, pti_c))
#            pti_loss_np = pti_loss.data[0]
#            pti_grads_dict = {name: p.grad.data.cpu().numpy()
#                              for name, p in pti_cell.named_parameters()}


if __name__ == '__main__':
    unittest.main()
