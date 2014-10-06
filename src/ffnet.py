"""
FFnet
"""

from random import Random

class FFnet:

	def __init__(self, layers):
		self.W = []
		self.layers = layers
		self.layercount = len(layers)
		for i in xrange(0, self.layercount):
			col = [0]*layers[j+1]
			mat = [col]*layers[j]
			self.W.append(mat);

	def initWeights(self, seed=None):
		if seed is not None:
			rand = Random(seed)
		else:
			rand = Random()
		mat = []
		for rc in self.W:
			mat = mat + [1.0]*len(rc[0])
			for i in xrange(1,len(rc)):
				mat = mat + [rand.gauss(0.0,1) for j in xrange(len(rc[i]))]
		self.setWeights(mat)

	def getWeights(self):
		mat = []
		for rc in self.W:
			for arr in rc:
				mat = mat + arr
		return mat

	def setWeights(self, wmat):

	def forwprop(self, inp):

	def backprop(self, err):


#import cudamat as cm                                                                    #import numpy as np 

#cm.cuda_set_device(0)
#cm.init()

#z2 = W1 * x + repmat(b1,1,m);
#a2 = f(z2);
#z3 = W2 * a2 + repmat(b2,1,m);
#h = f(z3)

#cm.shutdown()

