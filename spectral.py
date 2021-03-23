import laplacian
import eigenvalues
import numpy as np
import math

"""
Desc:   This is the main file regarding the spectral clustering algorithm.
		Here all the code for the algorithm is being glued together.
		
Note:   This code doesn't handle the part that sends the vectors of T to kmeans.
		That part is handled in main.py; this choice was made due to the fact
		that k is needed for the separate run of kmeans.  
"""


def eigengap(eigen_values):  # TODO - max and index will probably be slow
	"""
	The Eigengap Heuristic as described in the assignment

	:param eigen_values: sorted list of eigenvalues
	:return: int, first index of max value, as described in the assignment

	"""
	n = len(eigen_values)
	delta = [abs(eigen_values[i] - eigen_values[i + 1]) for i in range(int(math.ceil(n / 2)))]
	return delta.index(max(delta))


def eigenvalues_list(A):
	"""
	Transfers A to a sorted list of eigenvalue

	:param A: The diagonal matrix of eigenvalues
	:return: np array
	"""
	n = len(A)
	dtype = [('key', float), ('value', int)]
	eigen_values = [(A[i][i], i) for i in range(n)]
	eigen_values = np.array(eigen_values, dtype=dtype)
	eigen_values = np.sort(eigen_values, order='key')
	return eigen_values


def U_matrix(Q, k, eigen_values):
	"""
	Computes the U matrix as defined in the assignment

	:param Q: matrix, shape(n, n)
	:param k: int, amount of centroids
	:param eigen_values: array like, shape(k, )
	:return: computed U matrix, shape(n, k)

	"""
	U = [Q[:, eigen_values[j][1]] for j in range(k)]

	# TODO - what's happening here?
	# U = np.array(U)
	# U = transpose(U)
	return np.array(U)


def T_matrix(U):
	"""
	Computes the T matrix as defined in the assignment

	:param U: matrix, shape(n, k)
	:return: computed T matrix, shape(n, k)

	"""
	n = U.shape[0]
	k = U.shape[1]
	T = [[U[i][j] / np.linalg.norm(U[i]) for j in range(k)] for i in range(n)]
	return np.array(T)  # TODO - why transfer to array?


def main(X, k=None):
	"""
	The main function of the spectral algorithm code

	:param X: observations, shape(n, )
	:param k: int, amount of centroids
	:return:    The T matrix (vectors that will be transferred to C), and
				the computed k value

	"""
	n = len(X)

	Lnorm = laplacian.Lnorm_matrix(X)
	A, Q = eigenvalues.QR_Iterations(Lnorm)

	eigen_values = eigenvalues_list(A)
	if k is None:
		k = eigengap([eigen_values[i][0] for i in range(n)])

	U = U_matrix(Q, k, eigen_values)
	T = T_matrix(U)
	return T, k
