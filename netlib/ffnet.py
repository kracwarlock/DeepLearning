"""
FFnet
"""

#import pdb
import time
import numpy as np
import cudamat as cm
from cudamat import learn as cl
#from cudamat.examples import util

class FFNet:

	def __init__(self, epsilon, momentum, num_epochs, batch_size, num_batches, dim_in, dim_out, num_hid):

		# training parameters
		self.epsilon = epsilon
		self.momentum = momentum
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.num_batches = num_batches

		# model parameters
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.num_hid = num_hid

		# initialize weights
		self.w_w1 = cm.CUDAMatrix(dim_in ** -0.5 * np.random.randn(dim_in, num_hid))
		self.w_b1 = cm.CUDAMatrix(np.zeros((num_hid, 1)))
		self.w_w2 = cm.CUDAMatrix(num_hid ** -0.5 * np.random.randn(num_hid, dim_out))
		self.w_b2 = cm.CUDAMatrix(np.zeros((dim_out, 1)))

		# initialize weight update matrices
		self.wu_w1 = cm.empty(self.w_w1.shape).assign(0)
		self.wu_b1 = cm.empty(self.w_b1.shape).assign(0)
		self.wu_w2 = cm.empty(self.w_w2.shape).assign(0)
		self.wu_b2 = cm.empty(self.w_b2.shape).assign(0)

		# initialize temporary storage
		self.h = cm.empty((self.num_hid, self.batch_size))
		self.out = cm.empty((self.dim_out, self.batch_size))
		self.delta = cm.empty((self.num_hid, self.batch_size))

	def reinitTestStorage(self, kkk):
		# Initalize temporary storage.
		self.h = cm.empty((self.num_hid, kkk))
		self.out = cm.empty((self.dim_out, kkk))

	#def getWeights(self):

	#def setWeights(self, wmat):

	def train(self, dev_train, dev_lbl):

		# Train neural network.
		start_time = time.time()
		for epoch in range(self.num_epochs):
		    print "Epoch " + str(epoch + 1)
		    err = []

		    for batch in range(self.num_batches):
		        # get current minibatch
		        inp = dev_train.slice(batch*self.batch_size,(batch + 1)*self.batch_size)
		        target = dev_lbl.slice(batch*self.batch_size,(batch + 1)*self.batch_size)

		        # forward pass
		        cm.dot(self.w_w1.T, inp, target = self.h)

		        self.h.add_col_vec(self.w_b1)
		        self.h.apply_sigmoid()

		        cm.dot(self.w_w2.T, self.h, target = self.out)

		        self.out.add_col_vec(self.w_b2)
		        self.out.apply_sigmoid()

		        # back prop errors
		        self.out.subtract(target) # compute error

		        # gradients for w_w2 and w_b2
		        self.wu_w2.add_dot(self.h, self.out.T, beta = self.momentum)
		        self.wu_b2.add_sums(self.out, axis = 1, beta = self.momentum)

		        # compute delta
		        cm.dot(self.w_w2, self.out, target = self.delta)

		        # delta = delta * h * (1 - h)
		        cl.mult_by_sigmoid_deriv(self.delta, self.h)

		        # gradients for w_w1 and w_b1
		        self.wu_w1.add_dot(inp, self.delta.T, beta = self.momentum)
		        self.wu_b1.add_sums(self.delta, axis = 1, beta = self.momentum)

		        # update weights
		        self.w_w1.subtract_mult(self.wu_w1, self.epsilon/self.batch_size)
		        self.w_b1.subtract_mult(self.wu_b1, self.epsilon/self.batch_size)
		        self.w_w2.subtract_mult(self.wu_w2, self.epsilon/self.batch_size)
		        self.w_b2.subtract_mult(self.wu_b2, self.epsilon/self.batch_size)

		        # calculate error on current minibatch 
		        err.append(np.abs(self.out.asarray())>0.5)

		    print "Training misclassification rate: " + str(np.mean(err))
		    print "Time: " + str(time.time() - start_time)

	def test(self, dev_test, dev_lbl):

		# forward pass
		cm.dot(self.w_w1.T, dev_test, target = self.h)

		self.h.add_col_vec(self.w_b1)
		self.h.apply_sigmoid()

		cm.dot(self.w_w2.T, self.h, target = self.out)

		self.out.add_col_vec(self.w_b2)
		self.out.apply_sigmoid()

		# compute error
		self.out.subtract(dev_lbl)

		print "Testing misclassification rate: " + str(np.mean(np.abs(self.out.asarray())>0.5))
