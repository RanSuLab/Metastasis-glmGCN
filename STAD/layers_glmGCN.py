from inits import *
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(123)
tf.set_random_seed(123)
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape,seed=123)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class SparseGraphLearn(object):
    """Sparse Graph learning layer."""
    def __init__(self, input_dim, output_dim, num,edge, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.num_nodes = placeholders['num_nodes']
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.edge = edge
        self.num=num
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        weights_initializler = initializers.xavier_initializer(seed=11)
        biases_initializer = init_ops.zeros_initializer()
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable("weights", shape=[input_dim, output_dim],
                                                   initializer=weights_initializler)
            self.vars['a'] = tf.get_variable("a", shape=[ output_dim,1],
                                                   initializer=weights_initializler)
            if self.bias:
                self.vars['bias'] = tf.get_variable("bias", shape=[output_dim], initializer=biases_initializer)

    def __call__(self, inputs,adj):
        x = inputs
        if self.sparse_inputs:
            x = sparse_dropout(x, 0.3, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 0.3, seed=123)
        N = self.num
        h = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        edge_v = tf.abs(tf.gather(h,self.edge[0]) - tf.gather(h,self.edge[1]))
        edge_v = tf.squeeze(self.act(dot(edge_v, self.vars['a'])))
        edge_v=edge_v+np.log(adj[1])*30
        sgraph = tf.SparseTensor(indices=tf.transpose(self.edge), values=edge_v, dense_shape=[N, N])
        sgraph = tf.sparse_softmax(sgraph)
        return h, sgraph

class GraphConvolution(object):
    '''Implement graph classification'''
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.output_dim=output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        weights_initializler = initializers.xavier_initializer(seed=11)
        biases_initializer = init_ops.zeros_initializer()
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable("weights", shape=[input_dim, output_dim],
                                                    initializer=weights_initializler)
            if self.bias:
                self.vars['bias'] = tf.get_variable("bias", shape=[output_dim], initializer=biases_initializer)

        if self.logging:
            self._log_vars()


    def _call(self, inputs, adj):
        x = inputs
        inputs_unstacked = tf.unstack(x)
        def fn(x_slice):
           pre_sup = dot(x_slice, self.vars['weights'], sparse=self.sparse_inputs)
           output = dot(adj, pre_sup, sparse=True)
           if self.bias:
             output += self.vars['bias']
           return output
        outputs = tf.stack([fn(i)for i in inputs_unstacked])

        return self.act(outputs)

    def __call__(self, inputs, adj):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs, adj)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs
    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
