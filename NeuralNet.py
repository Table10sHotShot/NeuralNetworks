#!/usr/bin/python3

import numpy as np

class NN(object):

	def __init__(self, configuration):

		self.weights = []

		#layer configurations do not include bias unit
		self.configuration = configuration
		self.num_layers = len(configuration)
		
		#list of weight matrices
		#each weight matrix of form wij
		for layer_id in range(1, self.num_layers):
			#add 1 for bias unit, weight initialized between 0 and 0.2
			self.weights.append( np.matrix( np.random.rand( self.configuration[layer_id-1]+1, self.configuration[layer_id] ) / 5 ) )
#			self.weights.append( np.ones((self.configuration[layer_id-1]+1, self.configuration[layer_id])) / 2 )

	def _update(self, input_matrix):

		#input matrix in form examples x features

		self.a = []
		self.z = []
		
		#append next layer activations to a
		#next layer activations of form examples x features in current layer
		self.a.append( np.hstack( (np.ones((input_matrix.shape[0],1)), input_matrix) ) )

		for weight_matrix in self.weights:
			self.z.append( self.a[-1]*weight_matrix )
			self.a.append( np.hstack( (np.ones( (self.a[-1].shape[0], 1) ), 1/(1+np.exp(-self.z[-1]))) ) )

		self.a[-1] = self.z[-1]

		del self.z[-1]

	def back_propagate(self, targets, lambda_reg):
		
		num_examples = int(targets.shape[0])

		delta = []
		delta.append(self.a[-1] - targets)
		
		#calculate dE/dz for all layers
		for layer_id in range(1, self.num_layers-1):
			weight = self.weights[-layer_id][1:,:]
			dsigmoid = np.multiply(self.a[-(layer_id+1)][:,1:], 1-self.a[-(layer_id+1)][:,1:])
			delta.append(np.multiply(delta[-1]*weight.T, dsigmoid))
		
		#calculate dE/dW for all layers
		Delta=[]
		for layer_id in range(self.num_layers-1):
			Delta.append(self.a[layer_id].T * delta[-(layer_id+1)])

		self.weight_gradient = []
		for layer_id in range(len(Delta)):
			self.weight_gradient.append(
					np.nan_to_num(
						1/num_examples * Delta[layer_id]) +
						np.vstack( (np.zeros((1,self.weights[layer_id].shape[1])), lambda_reg/num_examples*self.weights[layer_id][1:,:]) )
					)
	
	def train(self, input_matrix, targets, alpha, lambda_reg, iterations=1000):

		for i in range(iterations):
			self._update(input_matrix)
			self.back_propagate(targets, lambda_reg)
			self.weights = [self.weights[layer_id] - alpha*self.weight_gradient[layer_id] for layer_id in range(len(self.weights))]
	
	def predict(self, input_matrix):

		self._update(input_matrix)
		return self.a[-1]

if __name__ == '__main__':

#	inputs = [[np.random.randint(20), np.random.randint(20), np.random.randint(20)] for _ in range(3200)]
	inputs = [[0,1], [1,0]]
#	inputs = [1,2,3]
#	targets = [entry[0] + entry[1] + entry[2] for entry in inputs]
	targets = [0, 1]
#	targets = [6]
	matrix_inputs = np.matrix(inputs)
	matrix_targets = np.matrix(targets).T

	nn = NN([2,8,1])
#	nn.train(np.matrix([[3,2],[3,4]]), np.matrix([5,7]).T, 0.08, 0.08, 2000)
	nn.train(matrix_inputs, matrix_targets, 0.05, 0, 30000)
	print(nn.predict(np.matrix([1,0])))
