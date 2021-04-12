import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from error import Error

"""
Desc:   This file handles all the work related to computing the eigenvalues and eigenvectors
		of the Lnorm matrix.
"""


def QR(A):
	"""
	The modified Gram-Schmidt algorithm as defined in the assignment

	:param A: Numpy matrix, shape(n, n)
	:return: Q, R - Numpy matrices, shape(n, n)

	"""
	n = A.shape[0]
	R = np.zeros((n, n))
	Q = np.zeros((n, n))
	Norm = np.zeros(n)
	U = A.copy()

	for i in range(n):
		Norm[i] = np.linalg.norm(U[:, i])

		if Norm[i] == 0:
			Error('Division By Zero', __file__)

		Q[:, i] = U[:, i] / Norm[i]
		R[i][i + 1:n] = Q[:, i].dot(U[:, i + 1:n])
		U[:, i + 1:n] = U[:, i + 1:n] - np.transpose(np.array([Q[:, i]])).dot(np.array([R[i, i + 1:n]]))

	np.fill_diagonal(R, Norm)
	return Q, R


def QR_Iterations(A, epsilon=0.0001):
	"""
	The QR Iteration algorithm as defined in the assignment

	:param A: Numpy matrix, shape(n, n)
	:param epsilon: Float value, as described in the assignment
	:return: Q, R - Numpy matrices, shape(n, n)

	"""
	n = A.shape[0]
	Abar = A.copy()
	Qbar = np.eye(n, dtype=float)

	for i in range(n):
		Q, R = QR(Abar)
		Abar = R.dot(Q)

		diff_mat = abs(Qbar) - abs(Qbar.dot(Q))
		if (abs(diff_mat) <= epsilon).all():
			return Abar, Qbar
		Qbar = Qbar.dot(Q)

	return Abar, Qbar
