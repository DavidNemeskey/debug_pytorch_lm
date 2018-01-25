import contextlib
import unittest

import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable

import lstm_pytorch as pti
import lstm_tf as tfi


class TestLstmCells(unittest.TestCase):
    """Tests that the two LSTM cell implementations work alike."""
    @contextlib.contextmanager
    def __create_cells(self, input_size, hidden_size, batch_size):
        """
        Helper method. Creates three objects:
        - a pytorch LstmCell
        - a tensorflow LstmCell
        - a session for the latter.
        """
        pti_cell = pti.LstmCell(input_size, hidden_size)

        with tf.Graph().as_default() as graph:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('Model', initializer=initializer):
                tfi_cell = tfi.LstmCell(input_size, hidden_size, batch_size)
            init = tf.global_variables_initializer()

        with tf.Session(graph=graph) as session:
            session.run(init)
            yield pti_cell, tfi_cell, session

    def __assert_cell_state_equals(self, pti_cell, tfi_cell, session):
        """Checks if the state vectors of the two cells are the same."""
        equals = []
        for name, _ in pti_cell.named_parameters():
            pti_value = getattr(pti_cell, name).data.cpu().numpy()
            tfi_value = session.run(getattr(tfi_cell, name))
            equals.append(np.allclose(pti_value, tfi_value))
        self.assertTrue(all(equals))
        # np.allclose(pti_value, tfi_d[name]for name, pti_value in pti_d.items()]

    def test_data_from_pytorch_to_tf(self):
        with self.__create_cells(3, 2, 4) as (pti_cell, tfi_cell, session):
            pti_d = pti_cell.save_parameters(prefix='Model/')
            tfi_cell.load_parameters(session, pti_d)

            self.__assert_cell_state_equals(pti_cell, tfi_cell, session)

    def test_data_from_tf_to_pytorch(self):
        """Tests data transfer from the tf cell to the pytorch one."""
        # input_size, hidden_size = 3, 2
        # batch_size = 4

        with self.__create_cells(3, 2, 4) as (pti_cell, tfi_cell, session):
            tfi_d = tfi_cell.save_parameters(session)
            pti_cell.load_parameters(tfi_d, prefix='Model/')

            self.__assert_cell_state_equals(pti_cell, tfi_cell, session)

    def __run_input(self, pti_cell, tfi_cell, session, input_np,
                    input_size, batch_size, pti_hidden=None, tfi_hidden=None):
        """Runs input_np through both cells, and returns the new states."""
        pti_input = Variable(torch.FloatTensor(input_np))
        tfi_input = tf.placeholder(tf.float32, [batch_size, input_size])

        tfi_init_state = tfi_cell.init_hidden()
        tfi_final_state = tfi_cell(tfi_input, tfi_init_state)

        if pti_hidden is None:
            pti_hidden = pti_cell.init_hidden(batch_size)
        if tfi_hidden is None:
            tfi_hidden = session.run(tfi_init_state)

        tfi_feed_dict = {tfi_input: input_np,
                         tfi_init_state: tfi_hidden}
        tfi_new_state = session.run(tfi_final_state, tfi_feed_dict)
        pti_new_state = pti_cell.forward(pti_input, pti_hidden)

        return pti_new_state, tfi_new_state

    def test_output(self):
        """Tests if the outputs to an input are the same."""
        input_size, batch_size = 3, 4

        with self.__create_cells(input_size, 2, batch_size) as (
            pti_cell, tfi_cell, session
        ):
            tfi_d = tfi_cell.save_parameters(session)
            pti_cell.load_parameters(tfi_d, prefix='Model/')

            input_np = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]],
                                dtype=np.float32)
            pti_hidden, (tfi_h, tfi_c) = self.__run_input(
                pti_cell, tfi_cell, session, input_np, input_size, batch_size)
            pti_h, pti_c = (v.data.cpu().numpy() for v in pti_hidden)

            self.assertTrue(np.allclose(tfi_h, pti_h))
            self.assertTrue(np.allclose(tfi_c, pti_c))

    def test_output_twice(self):
        """Same as test_output(), but after two consecutive inputs."""
        input_size, batch_size = 3, 4

        with self.__create_cells(input_size, 2, batch_size) as (
            pti_cell, tfi_cell, session
        ):
            tfi_d = tfi_cell.save_parameters(session)
            pti_cell.load_parameters(tfi_d, prefix='Model/')

            input_np = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]],
                                dtype=np.float32)
            pti_hidden, tfi_hidden = self.__run_input(
                pti_cell, tfi_cell, session, input_np, input_size, batch_size)
            pti_hidden, (tfi_h, tfi_c) = self.__run_input(
                pti_cell, tfi_cell, session, input_np, input_size, batch_size,
                pti_hidden, tfi_hidden)
            pti_h, pti_c = (v.data.cpu().numpy() for v in pti_hidden)

            self.assertTrue(np.allclose(tfi_h, pti_h))
            self.assertTrue(np.allclose(tfi_c, pti_c))

    def test_loss2(self):
        """Tests whether the states are the same after reading an input."""
        input_size, batch_size = 3, 4
        with self.__create_cells(input_size, 2, batch_size) as (
            pti_cell, tfi_cell, session
        ):
            tfi_d = tfi_cell.save_parameters(session)
            pti_cell.load_parameters(tfi_d, prefix='Model/')

            input_np = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]],
                                dtype=np.float32)
            pti_input = Variable(torch.FloatTensor(input_np))
            tfi_input = tf.placeholder(tf.float32, [batch_size, input_size])

            tfi_init_state = tfi_cell.init_hidden()
            tfi_final_state = tfi_cell(tfi_input, tfi_init_state)

            pti_hidden = pti_cell.init_hidden(batch_size)
            tfi_hidden = session.run(tfi_init_state)

            tfi_feed_dict = {tfi_input: input_np,
                             tfi_init_state: tfi_hidden}
            pti_h, pti_c = pti_cell.forward(pti_input, pti_hidden)
            pti_h, pti_c = pti_cell.forward(pti_input, (pti_h, pti_c))

            pti_cell.zero_grad()
            pti_loss = (pti_h - Variable(torch.FloatTensor([0, 1]))).norm(2)
            pti_loss.backward()
            pti_loss_np = pti_loss.data.cpu().numpy()
            pti_grads_dict = {name: p.grad.data.cpu().numpy()
                              for name, p in pti_cell.named_parameters()}

            tfi_loss = tf.norm((tfi_final_state[0] - tf.constant([0, 1.0])))
            tfi_grads_np, tfi_loss_np = session.run(
                [tf.gradients(tfi_loss, tfi_cell.weights), tfi_loss],
                tfi_feed_dict)
            tfi_grads_np, tfi_loss_np = session.run(
                [tf.gradients(tfi_loss, tfi_cell.weights), tfi_loss],
                tfi_feed_dict)
            tfi_grads_dict = {
                name: tfi_grads_np[tfi_cell.weights.index(getattr(tfi_cell, name))]
                for name in pti_grads_dict
            }

            self.assertTrue(np.allclose(tfi_loss_np, pti_loss_np))
            for name, pti_value in pti_grads_dict.items():
                self.assertTrue(np.allclose(pti_value, tfi_grads_dict[name]))
            # print(session.run(tfi_loss, tfi_feed_dict))
            # optimizer = tf.train.GradientDescentOptimizer(self._lr)
            # self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def test_loss(self):
        """Tests whether the states are the same after reading an input."""
        input_size, batch_size = 3, 4
        with self.__create_cells(input_size, 2, batch_size) as (
            pti_cell, tfi_cell, session
        ):
            tfi_d = tfi_cell.save_parameters(session)
            pti_cell.load_parameters(tfi_d, prefix='Model/')

            input_np = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]],
                                dtype=np.float32)
            pti_input = Variable(torch.FloatTensor(input_np))
            tfi_input = tf.placeholder(tf.float32, [batch_size, input_size])

            tfi_init_state = tfi_cell.init_hidden()
            tfi_final_state = tfi_cell(tfi_input, tfi_init_state)

            pti_hidden = pti_cell.init_hidden(batch_size)
            tfi_hidden = session.run(tfi_init_state)

            tfi_feed_dict = {tfi_input: input_np,
                             tfi_init_state: tfi_hidden}
            pti_h, pti_c = pti_cell.forward(pti_input, pti_hidden)
            # pti_h, pti_c = pti_cell.forward(pti_input, (pti_h, pti_c))

            pti_cell.zero_grad()
            pti_loss = (pti_h - Variable(torch.FloatTensor([0, 1]))).norm(2)
            pti_loss.backward()
            pti_loss_np = pti_loss.data.cpu().numpy()
            pti_grads_dict = {name: p.grad.data.cpu().numpy()
                              for name, p in pti_cell.named_parameters()}

            tfi_loss = tf.norm((tfi_final_state[0] - tf.constant([0, 1.0])))
            tfi_grads_np, tfi_loss_np = session.run(
                [tf.gradients(tfi_loss, tfi_cell.weights), tfi_loss],
                tfi_feed_dict)
            tfi_grads_np, tfi_loss_np = session.run(
                [tf.gradients(tfi_loss, tfi_cell.weights), tfi_loss],
                tfi_feed_dict)
            tfi_grads_dict = {
                name: tfi_grads_np[tfi_cell.weights.index(getattr(tfi_cell, name))]
                for name in pti_grads_dict
            }

            print(tfi_loss_np, pti_loss_np)
            self.assertTrue(np.allclose(tfi_loss_np, pti_loss_np))
            for name, pti_value in pti_grads_dict.items():
                self.assertTrue(np.allclose(pti_value, tfi_grads_dict[name]))
            # print(session.run(tfi_loss, tfi_feed_dict))
            # optimizer = tf.train.GradientDescentOptimizer(self._lr)
            # self._train_op = optimizer.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    unittest.main()
