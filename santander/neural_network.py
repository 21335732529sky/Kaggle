import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from utils import ParallelExecutorBase

class NNModel:
	def __init__(self, layers=(200, 2), lr=1e-3):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self._build_model(layers, lr)
		self.sess = tf.InteractiveSession(graph=self.graph)
		self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

	def _build_model(self, layers, lr):
		self.x = tf.placeholder(tf.float32, shape=[None, layers[0]])
		self.y_ = tf.placeholder(tf.float32, shape=[None, layers[-1]])
		self.keep_prob = tf.placeholder(tf.float32)

		self.layers = [self.x]

		for i in range(1, len(layers) - 1):
			w = tf.get_variable('W_hidden_{}'.format(i), dtype=tf.float32, shape=[layers[i - 1], layers[i]])
			b = tf.get_variable('b_hidden_{}'.format(i), dtype=tf.float32, shape=[layers[i]])
			hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(self.layers[-1], w) + b), keep_prob=self.keep_prob)
			self.layers.append(hidden)

		w = tf.get_variable('W_hidden_{}'.format(len(layers)), dtype=tf.float32, shape=[layers[-2], layers[-1]])
		b = tf.get_variable('b_hidden_{}'.format(len(layers)), dtype=tf.float32, shape=[layers[-1]])
		self.layers.append(tf.matmul(self.layers[-1], w) + b)

		self.predictions = tf.nn.softmax(self.layers[-1])
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.layers[-1], labels=self.y_))
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_, axis=-1), tf.argmax(self.layers[-1], axis=-1)), tf.float32))
		print(tf.argmax(self.y_, axis=-1))
		self.auc, self.update_op = tf.metrics.auc(tf.argmax(self.y_, axis=-1), tf.gather(self.predictions, 1, axis=-1))

		self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

	def train(self, x, y, keep_prob=0.7):
		feed_dict = {self.x: x, self.y_: y, self.keep_prob: keep_prob}
		loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

		return loss

	def evaluate(self, x, y):
		feed_dict = {self.x: x, self.y_: y, self.keep_prob: 1.0}
		auc, loss = self.sess.run([self.update_op, self.loss], feed_dict=feed_dict)

		return auc, loss

	def predict(self, x):
		feed_dict = {self.x: x, self.keep_prob: 1}
		preds = self.sess.run(self.predictions, feed_dict=feed_dict)

		return preds

def data_generator(X, Y, batch=32):
	count = 0
	index = list(range(X.shape[0]))
	pos = [i for i in index if Y[i] == 1]
	neg = [i for i in index if Y[i] == 0]
	while count < X.shape[0]:
		ch = [random.randint(0, 1) for _ in range(batch)]
		index_slice = [random.choice(pos) if i == 1 else random.choice(neg) for i in ch]
		bx = [X[i] for i in index_slice]
		by = [Y[i] for i in index_slice]
		by = [np.eye(2)[i] for i in by]
		count += batch
		yield bx, by, count

	return

def normalize(df, skip=2):
	for key in df.columns[skip:]:
		df.loc[:, key] = StandardScaler().fit_transform(df.loc[:, key].values.reshape(-1, 1))

	return df

def calc_poly(df, skip=0):
	double = df.iloc[:, skip:] ** 2
	triple = df.iloc[:, skip:] ** 3
	double.columns = [name + '^2' for name in double.columns]
	triple.columns = [name + '^3' for name in triple.columns]

	return [double, triple]

class ModelMaker(ParallelExecutorBase):
	def work(self, item):
		model = NNModel(layers=(600, 300, 100, 2))
		epoch = item['epoch']
		train_x, train_y = item['train_data']
		val_x, val_y = item['val_data']
		print(f"[ID: {os.getpid()}] start training (No.{item['i']})")
		for e in range(epoch):
			for bx, by, _ in data_generator(train_x, train_y, batch=item['batch_size']):
				loss = model.train(bx, by)
			val_auc, _ = model.evaluate(val_x, val_y)
			print(f'[ID: {os.getpid()}] epoch {e + 1} end. AUC = {val_auc:.6f}')

		print(f'[ID: {os.getpid()}] end training')
		preds = model.predict(val_x)
		print(preds.shape)
		preds = [e[1] for e in preds]
		np.save(f"predictions/preds_{item['i']}", np.array(preds))
		print(f'[ID: {os.getpid()}] saved prediction.')

	def make_items(self, train_data=(), val_data=(), epoch=5, batch_size=32, num_iter=30):
		item = {'train_data': train_data, 'val_data': val_data, 'epoch': epoch, 'batch_size': 32}
		for i in range(num_iter):
			item['i'] = i
			yield dict(item)

def main():
	print(len(os.listdir('predictions')))
	df = pd.read_csv('train.csv')
	test_df = pd.read_csv('test.csv')
	print('calculating round ...')
	rounds_train = calc_poly(df, skip=2)
	df = df.join([*rounds_train])
	df = normalize(df)
	train_x = df.iloc[:, 2:].values
	train_y = df.iloc[:, 1].values.flatten()
	train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
													  test_size=.1,
													  stratify=train_y,
													  shuffle=True,
													  random_state=7777)
	val_y = [np.eye(2)[i] for i in val_y]
	
	ens = ModelMaker()
	ens.run(3,
			train_data=(train_x, train_y),
			val_data=(val_x, val_y),
			epoch=1,
			batch_size=32,
			num_iter=100)

	preds = sum(np.load(f'predictions/{name}') for name in os.listdir('predictions')) / len(os.listdir('predictions'))
	val_y = [e[1] for e in val_y]
	precision, recall, _ = precision_recall_curve(val_y, preds)
	auc_score = auc(recall, precision)
	print(auc_score)

	'''
	model = NNModel(layers=(600, 300, 100, 2))
	epoch = 10
	batch_size = 32
	for i in range(epoch):
		print(f'epoch {i}:')
		v_auc = 0
		v_loss = 0
		t_loss = 0
		bar = tqdm(data_generator(train_x, train_y, batch=batch_size))
		for bx, by, c in bar:
			t_loss = model.train(bx, by)
			if (c // batch_size) % 30 == 0:
				v_auc, v_loss = model.evaluate(val_x, val_y)
			bar.set_description(f'train_loss = {t_loss:.5f} val_loss = {v_loss:.5f} AUC = {v_auc:.5f}')
	'''

if __name__ == '__main__':
	main()

