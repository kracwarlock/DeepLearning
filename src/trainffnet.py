from netlib import ffnet
import cudamat as cm
from cudamat.examples import util
import numpy as np

def main():

	# initialize CUDA
	cm.cublas_init()

	# training parameters
	epsilon = 0.01
	momentum = 0.9
	num_epochs = 30
	batch_size = 128
	num_batches = 92

	# model parameters
	dim_in = 784
	dim_out = 1
	num_hid = 1024

	# load data
	util.load('data/mnist49.dat', globals())
	global dat_train
	global dat_test
	global lbl_train
	global lbl_test

	# Put training data onto the GPU.
	dat_train = dat_train/255.
	dat_train = dat_train - (np.mean(dat_train, 1)+10**-8)[:, np.newaxis]
	dev_train = cm.CUDAMatrix(dat_train)
	dev_lbl = cm.CUDAMatrix(lbl_train)

	net = ffnet.FFNet(epsilon, momentum, num_epochs, batch_size, num_batches, dim_in, dim_out, num_hid)
	net.train(dev_train, dev_lbl)

	# Load test data onto the GPU.
	dat_test = dat_test/255.
	dat_test = dat_test - np.mean(dat_test, 1)[:, np.newaxis]
	dev_test = cm.CUDAMatrix(dat_test)
	dev_lbl = cm.CUDAMatrix(lbl_test)

	net.reinitTestStorage(dat_test.shape[1])
	net.test(dev_test, dev_lbl)

	cm.cublas_shutdown()

if __name__ == '__main__':
	main()
