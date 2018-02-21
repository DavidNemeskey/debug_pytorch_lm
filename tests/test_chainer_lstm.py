#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import contextlib
import importlib
import os
import unittest

try:
    chainer = importlib.import_module('chainer')
    import chainer.functions as F
    from chainer import Variable
except ImportError:
    chainer = None
import numpy as np
import tensorflow as tf

import pytorch_lm.lstm_chainer as chi
import pytorch_lm.lstm_tf as tfi


@unittest.skipIf(chainer is None, "Chainer not found.")
class TestLstms(unittest.TestCase):
    """Tests that the chainer LSTM implementation is the same as the TF one."""
    def setUp(self):
        self.input_size = 5
        self.hidden_size = 4
        self.batch_size = 3
        self.num_layers = 2

        weight_file = os.path.join(os.path.dirname(__file__), 'lstm.npz')
        if not os.path.isfile(weight_file):
            chi_lstm = chi.Lstm(self.input_size, self.hidden_size, self.num_layers)
            chi_d = chi_lstm.save_parameters(prefix='Lstm/')
            np.savez(weight_file, **chi_d)
        self.weights = dict(np.load(weight_file))

    @contextlib.contextmanager
    def __create_lstms(self, input_size=None, hidden_size=None,
                       num_layers=None, batch_size=None, cuda=False):
        """
        Helper method. Creates three objects:
        - a chainer Lstm
        - a tensorflow Lstm
        - a session for the latter.
        """
        input_size = input_size or self.input_size
        hidden_size = hidden_size or self.hidden_size
        num_layers = num_layers or self.num_layers
        batch_size = batch_size or self.batch_size
        chi_lstm = chi.Lstm(input_size, hidden_size, num_layers)
        if cuda:
            chi_lstm.to_gpu()

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Lstm', initializer=initializer):
                tfi_lstm = tfi.Lstm(input_size, hidden_size, num_layers, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)
            yield chi_lstm, tfi_lstm, session

    @unittest.skipUnless(chainer.cuda.available,
                         'CUDA test: a GPU is not available.')
    def test_cuda(self):
        """Tests if the Chainer network works on the GPU."""
        with self.__create_lstms(cuda=False) as (chi_lstm, tfi_lstm, session):
            self.assertTrue(next(chi_lstm.params())._cpu)
            chi_lstm.load_parameters(chi_lstm.save_parameters())
            self.assertTrue(next(chi_lstm.params())._cpu)

        with self.__create_lstms(cuda=True) as (chi_lstm, tfi_lstm, session):
            self.assertFalse(next(chi_lstm.params())._cpu)
            chi_lstm.load_parameters(chi_lstm.save_parameters())
            self.assertFalse(next(chi_lstm.params())._cpu)

    def __assert_parameters_equals(self, chi_lstm, tfi_lstm, session):
        """Checks if the state vectors of the two LSTM networks are the same."""
        equals = []
        for l, chi_cell in enumerate(chi_lstm.layers):
            tfi_cell = tfi_lstm.layers[l]
            for name, _ in chi_cell.namedparams():
                chi_value = F.copy(getattr(chi_cell, name[1:]), -1).data
                tfi_value = session.run(getattr(tfi_cell, name[1:]))
                equals.append(np.allclose(chi_value, tfi_value))
            self.assertTrue(all(equals))

    def __test_data_from_chainer_to_tf(self, cuda):
        with self.__create_lstms(batch_size=4, cuda=cuda) as (
            chi_lstm, tfi_lstm, session
        ):
            chi_d = chi_lstm.save_parameters(prefix='Lstm/')
            tfi_lstm.load_parameters(session, chi_d)

            self.__assert_parameters_equals(chi_lstm, tfi_lstm, session)

    def test_data_from_chainer_to_tf(self):
        """Tests data transfer from the chainer LSTM to the tf one."""
        with self.subTest(name='cpu'):
            self.__test_data_from_chainer_to_tf(False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__test_data_from_chainer_to_tf(True)

    def __test_data_from_tf_to_chainer(self, cuda):
        with self.__create_lstms(batch_size=4, cuda=cuda) as (
            chi_lstm, tfi_lstm, session
        ):
            tfi_d = tfi_lstm.save_parameters(session)
            chi_lstm.load_parameters(tfi_d, prefix='Lstm/')

            self.__assert_parameters_equals(chi_lstm, tfi_lstm, session)

    def test_data_from_tf_to_chainer(self):
        """Tests data transfer from the tf LSTM to the chainer one."""
        with self.subTest(name='cpu'):
            self.__test_data_from_tf_to_chainer(False)
        with self.subTest(name='gpu'):
            if not chainer.cuda.available:
                self.skipTest('CUDA test: a GPU is not available.')
            self.__test_data_from_tf_to_chainer(True)

    def __test_sequence_tagging(self, cuda):
        """Tests sequence tagging (i.e. the output)."""
        with self.__create_lstms(cuda=cuda) as (chi_lstm, tfi_lstm, session):
            tfi_lstm.load_parameters(session, self.weights)
            chi_lstm.load_parameters(self.weights, prefix='Lstm/')

            # Input
            input_np = np.array(
                [
                    [[1, 2, 1, 2, 1.5], [2, 4, 2, 4, 3], [2, 3, 2, 3, 2.5]],
                    [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                    [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
                ],
                dtype=np.float32
            )
            chi_input = Variable(input_np)
            if cuda:
                chi_input = chi_input.to_gpu()
            tfi_input = tf.placeholder(tf.float32, input_np.shape)

            # Target (arithmetic mean)
            target_np = np.array([[1.5, 3, 2.5], [1, 2, 3], [1, 2, 3]], dtype=np.float32)
            chi_target = Variable(target_np)
            if cuda:
                chi_target = chi_target.to_gpu()
            tfi_target = tf.placeholder(tf.float32, target_np.shape)

            # Initial states
            chi_hidden = chi_lstm.init_hidden(self.batch_size)
            tfi_hidden_np = [[F.copy(v, -1).data for v in l] for l in chi_hidden]
            tfi_init_state = tfi_lstm.init_hidden()
            tfi_output, tfi_final_state = tfi_lstm(tfi_input, tfi_init_state)

            optimizer = chainer.optimizers.SGD(lr=1.0)
            optimizer.setup(chi_lstm)

            # Chainer
            chi_lstm.cleargrads()
            chi_output, chi_final_state = chi_lstm(chi_input, chi_hidden)
            chi_loss = F.sqrt(F.sum((F.mean(chi_output, 2) - chi_target) ** 2))
            chi_loss.backward()
            chi_final_state_np = [[F.copy(a, -1).data for a in l]
                                  for l in chi_final_state]
            chi_loss_np = np.asscalar(F.copy(chi_loss, -1).data)
            chi_grads_dict = {l: {name: F.copy(p.grad, -1).data
                                  for name, p in layer.namedparams()}
                              for l, layer in enumerate(chi_lstm.layers)}

            # Tensorflow
            tfi_loss = tf.norm(tf.reduce_mean(tfi_output, axis=2) - tfi_target, 2)
            tfi_grad_vars = [w for l in tfi_lstm.layers for w in l.weights]
            tfi_gradients = tf.gradients(tfi_loss, tfi_grad_vars)
            tfi_optimizer = tf.train.GradientDescentOptimizer(1)
            tfi_train_op = tfi_optimizer.apply_gradients(zip(tfi_gradients, tfi_grad_vars))
            tfi_feed_dict = {tfi_input: input_np,
                             tfi_target: target_np,
                             tuple(tfi_init_state): tuple(tfi_hidden_np)}
            tfi_output_np, tfi_final_state_np, tfi_loss_np, tfi_grads_list, _ = session.run(
                [tfi_output, tfi_final_state, tfi_loss, tfi_gradients, tfi_train_op],
                tfi_feed_dict)
            tfi_grads_dict = {
                l: {
                    name: tfi_grads_list[tfi_grad_vars.index(getattr(layer, name[1:]))]
                    for name in chi_grads_dict[l]
                }
                for l, layer in enumerate(tfi_lstm.layers)
            }

            # The tests
            with self.subTest(name='state'):
                for layer in range(len(chi_final_state_np)):
                    for hc in range(2):
                        self.assertTrue(np.allclose(
                            chi_final_state_np[layer][hc],
                            tfi_final_state_np[layer][hc],
                        ))

            with self.subTest(name='loss'):
                self.assertTrue(np.allclose(tfi_loss_np, chi_loss_np))

            with self.subTest(name='gradients'):
                for layer, chi_grads in chi_grads_dict.items():
                    for name, chi_value in chi_grads.items():
                        self.assertTrue(np.allclose(
                            chi_value, tfi_grads_dict[layer][name]))

            # Apply the loss in Chainer
            optimizer.update()
            # For simple SGD, the 3 optimizer lines are equivalent to
            # for p in chi_lstm.params():
            #     p.data -= p.grad
            self.__assert_parameters_equals(chi_lstm, tfi_lstm, session)

    def test_sequence_tagging_cpu(self):
        """Tests sequence tagging (i.e. the output) on the CPU."""
        self.__test_sequence_tagging(False)

    @unittest.skipUnless(chainer.cuda.available,
                         'CUDA test: a GPU is not available.')
    def test_sequence_tagging_gpu(self):
        """Tests sequence tagging (i.e. the output) on the GPU."""
        self.__test_sequence_tagging(True)


if __name__ == '__main__':
    unittest.main()
