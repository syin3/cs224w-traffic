from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from lib import utils

class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """
    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, network_type, graphEmbedFile, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian"):
        """
        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        # print(num_nodes, num_proj, num_units)
        # 207 None 64: when creating cell
        # 207 1 64: when creating cell_with_projection
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = (network_type=='gconv')
        self._graphEmbedFile = graphEmbedFile
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            # supports have now two matrices for the two directions
            # all of them are of form D^{-1}W
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        # print('This is the number of matrices: ', len(supports))
        # 2
        # There are 2 matrices for bi-directional random walk
        # Hence either one or two matrices will be in list of supports
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """

        # print(inputs.get_shape())
        # print(state.get_shape())
        # (64, 414)
        # (64, 13248)

        with tf.variable_scope(scope or "dcgru_cell"):
            # print('111111111111111111111111111\n')
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                output_size = 2 * self._num_units
                # We start with bias of 1.0 to not reset and not update.
                if self._use_gc_for_ru:
                    fn = self._gconv
                else:
                    # What is _fc?? fully connecgted
                    fn = self._fc
                # "value" is the output
                value = tf.nn.sigmoid(fn(inputs, state, output_size, bias_start=1.0))
                value = tf.reshape(value, (-1, self._num_nodes, output_size))

                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))

                # print('333333333333333333333333333\n')

            with tf.variable_scope("candidate"):
                c = fn(inputs, r * state, self._num_units)
                if self._activation is not None:
                    # activation is tanh
                    c = self._activation(c)
            # print('444444444444444444444444444444\n')
            # This is on Page 3 of paper
            # H(t) = u * H(t-1) + (1-u) * c
            # print(u.get_shape())
            # print(state.get_shape())
            # print(c.get_shape()) # second dim: 2 * self._num_units
            # (64, 13248)
            # (64, 13248)
            # (13248, 64)
            output = new_state = u * state + (1 - u) * c
            # print('555555555555555555555555555555\n')
            # When there is projection, we need an extra step
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        # What is _fc??
        node2vec = np.load('./data/embeddings/{}.npy'.format(self._graphEmbedFile))
        n2vMatrix = np.repeat(node2vec[np.newaxis, :, :], 64, axis=0)
        # print(n2vMatrix.shape)
        n2vMatrix = tf.reshape(tf.constant(n2vMatrix, dtype=tf.float32), (64, -1))
        # print(n2vMatrix.get_shape())
        # x = np.stack((x, n2vMatrix), axis=3)
        # x = np.stack((y, n2vMatrix), axis=3)
        # y_t = np.dstack((y_t, n2vMatrix))
        # print(x.shape)
        # exit()
        # (64, 12, 207, 2)
        n2vDim = int(n2vMatrix.get_shape()[1].value / 207)

        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))

        n2vMatrix = tf.reshape(n2vMatrix, (batch_size * self._num_nodes, -1))
        inputDim = int(inputs.get_shape()[1].value)
        
        # need to control this line
        # when training/ predicting without node2vec, comment out
        # when training/ predicting with node2vec, comment in
        inputs = tf.concat([inputs, n2vMatrix], axis=-1)
        # print(inputs.get_shape())
        # exit()

        inputs_and_state = tf.concat([inputs, state], axis=-1)
        # print(inputs_and_state.get_shape())
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        # print(inputs_and_state.get_shape(), weights.get_shape())
        # print(inputs_and_state)
        # print(weights)

        # value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        value = tf.matmul(inputs_and_state, weights)
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        # print('finished')
        # output dimension is (batch_size * self._num_nodes, output_size)
        return tf.reshape(value, [batch_size, self._num_nodes * output_size])

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """

        # print(inputs)
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)

        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))

        # print(inputs.eval())
        # print(state.eval())
        # print(inputs.get_shape())
        # print(state.get_shape())
        # exit()
        # You must feed a value for placeholder tensor

        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        # obtain transpose with a specified permutation: https://www.tensorflow.org/api_docs/python/tf/transpose
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                # supports is full of matrices
                # This is the diffusion convolution
                for support in self._supports:
                    # print(support)
                    # print(support.eval())
                    # print(x0.eval())
                    # exit()
                    # There is value in support
                    # But still a place holder for x0: You must feed a value for placeholder tensor


                    # This is where the convolution takes place.
                    # Matrices of laplacian or random walk or dual random walk multiplies the x0
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        # concats to matrix x, will multiply together
                        x = self._concat(x, x2)
                        # if self._max_diffusion_step = 2, then it's really meaningless to have this substitutino
                        # x1, x0 = x2, x1

            # two matrices in self._supports
            # K = 2
            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            # print(num_matrices, '*************')
            # num_matrices = 5
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            # This is the diffusion convolutional Layer
            # https://www.tensorflow.org/api_docs/python/tf/get_variable
            # Gets an existing variable with these parameters or create a new one.
            # so maybe the weights are the same for each input of RNN cells
            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            # print(x.get_shape(), '**********')
            # (13248, 640) or (13248, 330)
            # print(input_size, num_matrices, output_size)
            # print(weights.get_shape(), '**********')
            # 640, 330 paired with 64, 128 in encoding
            # 640, 325 paired with 64, 128 in encoding

            # Finally multiplies the weights
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            # https://www.tensorflow.org/api_docs/python/tf/get_variable
            # Same as weights, if the first time, we create variable biases
            # if not first time and biases already exists, we just bring it here
            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)

        # Shuyi:
        # Clearly, x is the result of this function: _gconv
        # An explanation for __call__ is: https://www.daniweb.com/programming/software-development/threads/39004/what-does-call-method-do
        # In short, the obejct of class DCGRUCell is first created, and then called as a function
        # The creation is in dcrnn_model.py, objects are created at Line 45, 47
        # They are cell and cell_with_projection.
        # The following two instances are just creating a list of repeated elements, of a certain length
        # E.g. [1] * 5 = [1,1,1,1,1]
        # encoding_cells = [cell] * num_rnn_layers
        # decoding_cells = [cell] * (num_rnn_layers - 1) + [cell_with_projection]
        # So the output of  is [output, new_state] and new_state is used by building RNN cells
        # After searching, I found that the objects are called as function in Line 85 and 86,
        # where they are certainly operated by rnn.static_rnn and rnn_decoder
        #   _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
        #   outputs, final_state = legacy_seq2seq.rnn_decoder(labels, enc_state, decoding_cells,
                                                                  # loop_function=_loop_function)
        # We now see what inputs here are:
        # they are self._inputs defined at Line 38.
        # They are in fact: place holders
        # self._inputs = tf.placeholder(tf.float32, shape=(batch_size, seq_len, num_nodes, input_dim), name='inputs')
        # Placeholders are defined and explained at this site: https://www.tensorflow.org/api_docs/python/tf/placeholder
        # They need to be fed before evaluated.
        # self._inputs is later unstacked and then fed into Line 85

        # Are we really only playing with the convolution??
            # print(type(x))
            # print(x.get_shape())
            # print('------------------\n')
            # (13248, 128) or (13248, 64)

        # Now we roughly know how the convolution works. Where and how does the back propagation come in?

        # An activation is right away called after this 
        # print(x.eval())
        # exit()
        # x is a placeholder: You must feed a value for placeholder tensor
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
        # After this, an activation function is called
