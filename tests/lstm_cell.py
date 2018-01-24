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

    def test_output(self):
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
            tfi_h, tfi_c = session.run(tfi_final_state, tfi_feed_dict)
            pti_h, pti_c = (v.data.cpu().numpy() for v in
                            pti_cell.forward(pti_input, pti_hidden))
            self.assertTrue(np.allclose(tfi_h, pti_h))
            self.assertTrue(np.allclose(tfi_c, pti_c))


if __name__ == '__main__':
    unittest.main()
