import numpy as np

class NN:

	def __init__(self, input_neurons, hidden_neurons, output_neurons):
		np.random.seed(0)
		self.W1 = np.random.randn(input_neurons, hidden_neurons)
		self.W2 = np.random.randn(hidden_neurons, output_neurons)
		self.B1 = np.zeros((1, hidden_neurons))
		self.B2 = np.zeros((1, output_neurons))

	def sigmoid(self, x):
		y = 1/(1 + np.exp(-x))
		return y

	def sigmoid_deriv(self, x):
		y = self.sigmoid(x) *  (1-self.sigmoid(x))
		return y


	def foward(self, X):
		self.x = X
		self.A1 = np.dot(X, self.W1)
		self.A1 += self.B1
		self.Z1 = self.sigmoid(self.A1)

		self.A2 = np.dot(self.Z1, self.W2)
		self.A2 += self.B2
		self.Z2 = self.sigmoid(self.A2)

		return self.Z2

	def backprop(self, Y):
		self.error = Y - self.Z2
		self.d_output = self.error * self.sigmoid_deriv(self.Z2)

		self.error_hidden = self.d_output.dot(self.W2.T)
		self.d_hidden = self.error_hidden * self.sigmoid_deriv(self.Z1)

		return self.d_output, self.error_hidden, self.d_hidden

	def update(self, lr):
		self.W2 += self.Z1.T.dot(self.d_output) * lr
		self.B2 += np.sum(self.d_output, axis=0, keepdims=True) * lr
		self.W1 += self.x.T.dot(self.d_hidden) * lr
		self.B1 += np.sum(self.d_hidden, axis=0, keepdims=True) *lr

	def cost(self):
		return np.mean(np.abs(self.d_output))

