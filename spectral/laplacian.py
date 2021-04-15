import numpy as np
import math

# This code adds the root directory to the path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error import Error
import consts

"""
Desc:   This file handles all the work related to computing the Lnorm matrix.
"""


def W_matrix(X):
	"""
	Computes the W matrix as defined in the assignment

	:param X: Numpy matrix, shape(n, d)
	:return: W - Numpy matrix, shape(n, n)

	"""
	n = X.shape[0]
	W = [[exponent(X[i], X[j]) for j in range(n)] for i in range(n)]
	return np.array(W)


def exponent(v1, v2):
	"""
	Computes the exponent as described in the assignment

	:param v1: Numpy vector, shape(d, )
	:param v2: Numpy vector, shape(d, )

	"""
	exp = (-1) * (np.linalg.norm(v1 - v2)) / 2
	return math.exp(exp)


def D_matrix(W):
	"""
	Computes the D^(-1/2) matrix as defined in the assignment

	:param W: Numpy matrix, shape(n, n)
	:return: Numpy array, shape(n, ). represents diagonal values of the D^(-1/2) matrix

	"""
	D = W.sum(axis=1)

	for d in D:
		if d == 0:
			Error('Division By Zero', __file__)

	D_half = [1 / math.sqrt(d) for d in D]
	return D_half


def Lnorm_matrix(X):
	"""
	Computes the Lnorm matrix as defined in the assignment

	:param X: Numpy matrix, shape(n, d)
	:return: Numpy matrix, shape(n, n)

	"""
	W = W_matrix(X)
	n = W.shape[0]
	D = np.diag(D_matrix(W))  # this is the actual D^(-1/2) matrix
	return np.eye(n) - D.dot(W.dot(D))
