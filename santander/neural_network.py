import tensorflow as tf
import pandas as pd
import numpy as np

class NNModel:
	def __init__(self, layers=(200, 2), lr=1e-3):
		self._build_model(layers, lr)

	def _build_model(self, layers, lr):
		self.x = tf.placeholder(tf.float32, shape=[None, layers[0]])
		self.y_ = tf.placeholder(tf.float32, shape=[None, layers[-1]])
		self.keep_prob = tf.placeholder(tf.float32)

		self.layers = [self.x]

		for i in range(1, len(layers) - 1):
			w = tf.get_variable('W_hidden_{}'.format(i), dtype=tf.float32, shape=[layers[i], layers[i + 1]])
			b = tf.get_variable('b_hidden_{}'.format(i), dtype=tf.float32, shape=[layers[i + 1]])
			hidden = tf.dropout(tf.relu(tf.matmul(self.layers[-1], w) + b), keep_prob=self.keep_prob)
			self.layers.append(hidden)