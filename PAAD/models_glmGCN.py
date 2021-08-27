from layers_glmGCN import *
from metrics import *
import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(123)
tf.set_random_seed(123)
class glmGCN(object):
    def __init__(self, placeholders, edge, adj,num,input_dim, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.loss1 = 0
        self.loss2 = 0
        self.inputs = placeholders['features']
        self.secondinput=placeholders['secondinput']
        self.edge = edge
        self.adj=adj
        self.num=num
        self.input_dim = input_dim
        self.num_nodes = placeholders['num_nodes']
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        learning_rate1 = tf.train.exponential_decay(
            learning_rate = FLAGS.lr1, global_step=placeholders['step'], decay_steps=100, decay_rate=0.9, staircase=True)
        learning_rate2 = tf.train.exponential_decay(
            learning_rate = FLAGS.lr2, global_step=placeholders['step'], decay_steps=100, decay_rate=0.9, staircase=True)
        self.optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate1)
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)

        self.layers0 = SparseGraphLearn(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden_gl,
                                        num=self.num,
                                        edge=self.edge,
                                        placeholders=self.placeholders,
                                        act=tf.nn.sigmoid,
                                        dropout=True,
                                        sparse_inputs=True)

        self.layers1 = GraphConvolution(input_dim=1,
                                            output_dim=FLAGS.hidden_gcn,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging)

        self.layers2 = GraphConvolution(input_dim=FLAGS.hidden_gcn,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging)

        self.build()
        self.pro = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for var in self.layers0.vars.values():
            self.loss1 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers1.vars.values():
            self.loss2 += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Graph Learning loss
        D = tf.matrix_diag(tf.ones(self.placeholders['num_nodes']))*-1
        D = tf.sparse_add(D, self.S)*-1
        D = tf.matmul(tf.transpose(self.x), D)

        self.loss1 += tf.trace(tf.matmul(D, self.x)) * FLAGS.losslr1
        self.loss1 -= tf.trace(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.S), tf.sparse_tensor_to_dense(self.S))) * FLAGS.losslr2
        N = self.num
        adj = tf.SparseTensor(indices=tf.constant(self.adj[0], dtype=tf.int64),
                              values=tf.constant(self.adj[1], dtype=tf.float32), \
                              dense_shape=[N, N])
        S_A = tf.sparse_add(self.S, tf.pow(tf.sparse_tensor_to_dense(adj), 30) * -1)
        self.loss1 -= tf.trace(tf.matmul(tf.transpose(S_A), S_A)) * FLAGS.losslr3
        self.loss2 += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

        self.loss = self.loss1 + self.loss2

    def _accuracy(self):
        self.accuracy, self.y_pred, self.y_actual, self.y_score = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def build(self):
        self.x, self.S = self.layers0(self.inputs, self.adj)
        keep_probability = tf.placeholder_with_default(1.0, shape=[])
        nn = tf.layers.batch_normalization(self.secondinput)

        x1 = self.layers1(nn, self.S)
        layer1 = tf.layers.batch_normalization(x1)
        layer2 = self.layers2(layer1, self.S)
        layer2 = tf.layers.batch_normalization(layer2)

        # Flatten inputs
        flattened = tf.layers.flatten(layer2)
        dense1 = tf.contrib.layers.fully_connected(flattened, 4096, activation_fn=tf.nn.relu)
        dense1 = tf.nn.dropout(dense1, keep_prob=keep_probability, seed=3)
        dense2 = tf.contrib.layers.fully_connected(dense1, 2048, activation_fn=tf.nn.relu)
        dense2 = tf.nn.dropout(dense2, keep_prob=keep_probability, seed=3)
        dense3 = tf.contrib.layers.fully_connected(dense2, 1024, activation_fn=tf.nn.relu)
        dense3 = tf.nn.dropout(dense3, keep_prob=keep_probability, seed=3)
        dense4 = tf.contrib.layers.fully_connected(dense3, 512, activation_fn=tf.nn.relu)
        dense4 = tf.nn.dropout(dense4, keep_prob=keep_probability, seed=3)
        dense = tf.contrib.layers.fully_connected(dense4, 2, activation_fn=tf.identity)
        #Save the features extracted
        self.layer = dense
        self.outputs = tf.nn.softmax(dense)

        # Build metrics
        self._loss()
        self._accuracy()
        #Set the hierarchical learning rate
        self.vars1 = tf.trainable_variables()[0:2 ]
        self.vars2 = tf.trainable_variables()[2:]
        # Automatically calculate and optimize the derivative of parameters
        self.opt_op1 = self.optimizer1.minimize(self.loss1, var_list=self.vars1)
        self.opt_op2 = self.optimizer2.minimize(self.loss2, var_list=self.vars2)
        self.opt_op = tf.group(self.opt_op1, self.opt_op2)

    def predict(self):
        return tf.nn.softmax(self.outputs)