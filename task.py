from sklearn.base import BaseEstimator
import time
import tensorflow as tf
import numpy as np

class TF_NN(BaseEstimator):

	def __init__(self, n_hidden, learning_rate=0.01, activation=tf.nn.relu, random_seed=None, batch=20, momentum=0.9, epochs=500):
		self.learning_rate = learning_rate
		self.random_seed = random_seed
		self.g = tf.Graph()
		self.training_cost = []
		self.n_hidden = n_hidden
		self.batch = batch
		self.activation = activation
		self.epochs = epochs
		self.momentum = momentum

	def create_batch_generator(self, X, y, batch_size, shuffle):
		X_copy = np.array(X)
		y_copy = np.array(y)

		if shuffle:
			np.random.seed(self.random_seed)
			data = np.column_stack((X_copy, y_copy))
			np.random.shuffle(data)
			X_copy = data[:, :-1]
			y_copy = data[:, -1]

		for i in range(0, X.shape[0], batch_size):
			yield(X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])

	def neuron_layer(self, input, output, activation, name):
		### for reusing the same variable for example when we use cross validation
		### by default fully_connected use xavier_initializer()
		try:
			h = tf.contrib.layers.fully_connected(inputs=input, num_outputs=output, activation_fn=activation,
												  weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1, scope='l2'), scope=name)
		except:
			h = tf.contrib.layers.fully_connected(inputs=input, num_outputs=output, activation_fn=activation,
												  weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1, scope='l2'), scope=name,
												  reuse=True)
		return h

	def neuron_layer_scratch(self, input, output, activation, name):
		pass

	def build(self, X, y, x_dim, n):
		##### placeholders for input
		self.tf_x = tf.placeholder(dtype=tf.float32, shape=(None, x_dim), name='tf_x')
		self.tf_y = tf.placeholder(dtype=tf.float32, shape=(None), name='tf_y')

		##### I need to put all the building of the NN in a variable scope in order to avoid problem
		# of adam and momentum optimizer: when you use them twice (like in cross validation) the break
		try:
			with tf.variable_scope('DNN') as scope:
				self.neural_net()
		except:
			with tf.variable_scope('DNN', reuse=True) as scope:
				self.neural_net()

	def neural_net(self):
		with tf.variable_scope('neural_layer'):
			hidden = self.neuron_layer(self.tf_x, self.n_hidden[0], activation=self.activation, name='hidden1')
			if len(self.n_hidden) == 2:
				hidden = self.neuron_layer(hidden, self.n_hidden[1], activation=self.activation, name='hidden2')
			self.final_layer = self.neuron_layer(hidden, 1, None, 'output')

		with tf.variable_scope('loss'):
			sqr_errors = tf.square(self.tf_y - self.final_layer)
			mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')

			#### let's add regularization (see weights_regularizer in the function neuron_layer())
			with tf.variable_scope('loss_reg'):
				self.loss = mean_cost + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'DNN'))

		with tf.variable_scope('gradient'):
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='GradientDescent')
			optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, name='MomentumOptimizer')
			#optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='AdamOptimizer')
			self.optimizer = optimizer.minimize(self.loss)

	def fit(self, X, y, shuffle=True):
		## build the model
		with self.g.as_default():
			tf.set_random_seed(self.random_seed)
			self.build(X, y, X.shape[1], X.shape[0])
			self.init_op = tf.global_variables_initializer()

		## run the model
		self.sess = tf.Session(graph=self.g)
		self.train(X, y, shuffle)

	def train(self, X, y, shuffle):
		## initialize all variables
		self.sess.run(self.init_op)
		for epoch in range(self.epochs):
			epoch_cost = []
			batch_generator = self.create_batch_generator(X, y, batch_size=self.batch, shuffle=shuffle)

			for bat_X, bat_y in batch_generator:
				feed = {self.tf_x: bat_X, self.tf_y: bat_y}
				_, batch_cost = self.sess.run([self.optimizer, self.loss], feed_dict=feed)
				epoch_cost.append(batch_cost)

			self.training_cost.append(np.mean(epoch_cost))
			#if not (epoch+1) % 50: print('---- Epoch %2d: Avg. Training Loss: %.4f' % (epoch+1, np.mean(epoch_cost)))

	def predict(self, X):
		with tf.name_scope("predict"):
			y_pred = self.sess.run(self.final_layer, feed_dict={self.tf_x: X})
		return y_pred

	# def plotTrainingCost(self):
		# plt.plot(range(1, len(self.training_cost) + 1), self.training_cost)
		# plt.tight_layout()
		# plt.xlabel('Epochs')
		# plt.ylabel('Training Cost')
		# plt.show()


if __name__ == "__main__":
	inizio = time.time()
	seed = 0

	####### BABY TEST
	X_train = np.arange(20).reshape((20, 1))
	X_test = np.arange(10).reshape((10, 1))
	y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0,
						9.3, 9, 10, 10.5, 11.9, 11.7, 12.5, 12, 12.2, 14])

	model = TF_NN([15,10], learning_rate=0.0001, activation=tf.tanh, random_seed=0, batch=2, epochs=100)


	model.fit(X_train, y_train)
	#model.plotTrainingCost()
	y_pred = model.predict(X_train)
	print(y_pred)


	"""
	plt.plot(X_train, y_train)
	plt.scatter(X_train, y_pred)
	plt.show()
	"""

	gcloud ml - engine jobs submit training $JOB_NAME - -job - dir $OUTPUT_PATH - -runtime - version 1.2 - -module - name trainer.task	- -package - path trainer /	--region $REGIO







	print('\nCalcolato Tutto in', (time.time() - inizio) / 60., 'min')
