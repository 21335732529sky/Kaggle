import tensorflow as tf
import itertools
import numpy as np
#from Dataset import Dataset
import warnings

warnings.filterwarnings('ignore')


class AutoEncoder:
    def __init__(self, input_dim, num_of_units):
        self.graph = tf.Graph()

        self._build_graph(self.graph, input_dim, num_of_units)

        self.sess = tf.InteractiveSession(graph=self.graph)
        self.sess.run(tf.global_variables_initializer())
    def _build_graph(self, graph, input_dim, num_of_units):
        with graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, input_dim])
            self.dropout = tf.placeholder(tf.float32)

            self.weights = [self._weight_variable([input_dim, num_of_units])]
            self.biases = [self._bias_variable([num_of_units])]

            self.weights.append(self._weight_variable([num_of_units, input_dim]))
            self.biases.append(self._bias_variable([input_dim]))

            self.hidden = tf.nn.dropout(tf.matmul(self.inputs, self.weights[0]) + self.biases[0],
                                        keep_prob=self.dropout)
            self.out = tf.nn.dropout(tf.matmul(self.hidden, self.weights[1]) + self.biases[1],
                                     keep_prob=self.dropout)

            self.error = tf.reduce_mean(tf.square(self.out - self.inputs)) / 2
            self.optimizer = tf.train.AdamOptimizer().minimize(self.error)

    def _weight_variable(self, shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def train(self, x, iteration, batch, verbose=False, dropout=1.0):
        fold = x.shape[0] // batch
        for i, f in itertools.product(range(iteration), range(fold)):
            _ = self.sess.run(self.optimizer,
                              feed_dict={self.inputs: x[f*batch:(f+1)*batch],
                                         self.dropout: dropout})
            if f == 0 and verbose:
                error = self.sess.run(self.error,
                                      feed_dict={self.inputs: x,
                                                 self.dropout: dropout})
                print('Error = {:.5f}'.format(error))


    def encode(self, x):
        return self.sess.run(self.hidden, feed_dict={self.inputs: x, self.dropout: 1.0})


#for test
if __name__ == '__main__':
    d = Dataset('D:HomeCredit/bureau.csv', '', omit=[['SK_ID_CURR', 'SK_ID_BUREAU'], []],
                validate_size=0.1, target='SK_ID_CURR').train_data()

    ae = AutoEncoder(d[0].shape[1], int(d[0].shape[1]/3))

    ae.train(d[0].values, 2, 20, verbose=True)
