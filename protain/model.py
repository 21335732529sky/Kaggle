import tensorflow as tf

class Resnet:
    def __init__(self, resnet_layers=3, dropout=None, alpha=0.001, input_dim=[512, 512, 4], output_dim=28, channels=[4, 8, 16, 16]):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph(resnet_layers, dropout, alpha, input_dim, output_dim, channels)
        self.sess = tf.InteractiveSession(graph=self.graph)
        self.sess.run(tf.global_variables_initializer())


    @staticmethod
    def _resnet_block(i, channel_in, channel_out, input_):
        w0 = tf.get_variable('f_resnet_{}_0'.format(i), dtype=tf.float32, shape=[3, 3, channel_in, channel_out])
        h0 = tf.nn.relu(tf.nn.conv2d(input_, w0, [1, 1, 1, 1], padding='SAME'))

        w1 = tf.get_variable('f_resnet_{}_1'.format(i), dtype=tf.float32, shape=[3, 3, channel_out, channel_out])
        h1 = tf.nn.relu(tf.nn.conv2d(h0, w1, [1, 1, 1, 1], padding='SAME'))
        w2 = tf.get_variable('f_resnet_{}_2'.format(i), dtype=tf.float32, shape=[3, 3, channel_out, channel_out])
        h2 = tf.nn.relu(tf.nn.conv2d(h1, w2, [1, 1, 1, 1], padding='SAME') + tf.identity(h0))

        return h2

    def _build_graph(self, resnet_layers, dropout, alpha, input_dim, output_dim, channels):
        self.x = tf.placeholder(tf.float32, [None, *input_dim])
        self.y_ = tf.placeholder(tf.float32, [None, output_dim])
        self.layers = [self._resnet_block(0, channels[0], channels[1], self.x)]
        for i in range(1, resnet_layers):
            self.layers.append(tf.nn.max_pool(self._resnet_block(i, channels[i], channels[i+1], self.layers[-1]), (1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME'))

        self.flatten = tf.layers.flatten(self.layers[-1])
        self.w_fc = tf.get_variable('W_fc', dtype=tf.float32, shape=[self.flatten.get_shape().as_list()[1], output_dim])
        self.b_fc = tf.get_variable('b_fc', dtype=tf.float32, shape=[output_dim])

        self.output = tf.matmul(self.flatten, self.w_fc) + self.b_fc
        self.preds = tf.sigmoid(self.output)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.y_))
        self.optimizer = tf.train.AdamOptimizer(alpha).minimize(self.loss)

    def train(self, x, y):
        feed_dict = {self.x: x, self.y_: y}
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def predict(self, x):
        feed_dict = {self.x: x}
        return self.sess.run(self.preds, feed_dict=feed_dict)
