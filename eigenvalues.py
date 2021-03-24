from numpy import transpose, diagonal
from numpy.linalg import norm
import numpy as np
import sklearn.datasets
import laplacian
import time

"""
Desc:   This file handles all the work related to computing the eigenvalues and eigenvectors
		of the Lnorm matrix.
"""

# TODO- should we work with copies everywhere?


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
		Norm[i] = norm(U[:, i])
		Q[:, i] = U[:, i] / Norm[i]
		R[i][i + 1:n] = Q[:, i].dot(U[:, i + 1:n])  # TODO - what's happening here?
		U[:, i + 1:n] = U[:, i + 1:n] - transpose(np.array([Q[:, i]])).dot(np.array([R[i, i + 1:n]]))

	np.fill_diagonal(R, Norm)
	return Q, R


# TODO - how the value of epsilon was decided?
def QR_Iterations(A, epsilon=0.0001):  # doesn't converge all the way
	"""
	The QR Iteration algorithm as defined in the assignment

	:param A: Numpy matrix, shape(n, n)
	:param epsilon: Float value, as described in the assignment
	:return: Q, R - Numpy matrices, shape(n, n)

	"""

	n = A.shape[0]
	Abar = A.copy()
	Qbar = np.eye(n, dtype=float)
	Sum = 0

	for i in range(max(n, 10)):  # TODO - why not just n?
		t1 = time.time()
		Q, R = QR(Abar)
		Sum += time.time() - t1
		Abar = R.dot(Q)

		diff_mat = Qbar - Qbar.dot(Q)  # TODO - shouldn't there be an absolute value?
		if (abs(diff_mat) <= epsilon).all():  # TODO - as stated, absolute value in the wrong place
			return Abar, Qbar

		Qbar = Qbar.dot(Q)

	print("QR total time is: ", Sum)
	return Abar, Qbar


"""
DEBUG FUNCTIONS
"""

def debug_general():
	A = np.array([(5.0, 3.0, 1.0, 4.0), (3.0, 6.0, 0.0, 2.5), (1.0, 0.0, 3.0, 1.7), (4.0, 2.5, 1.7, 10.0)])
	# A = np.full((3,3),2)
	(Q, R) = QR(A)
	print("Q is:")
	print(Q)
	print("R is:")
	print(R)
	print("The QR multiplication is:")
	print(Q.dot(R))
	print("")
	(Abar, Qbar) = QR_Iterations(A)
	print(Abar)
	print("")
	# print("")
	# print(Qbar)
	# print("")
	# print(U_matrix(A))
	# print(T_matrix(U_matrix(A)))
	print(np.linalg.eigvalsh(A))
	# (Abar,Qbar) = QR_Iterations(A)
	# print(Abar)
	# A = np.array([(5.0,3.0),(3.0,6.0)])
	# print("")
	# [Q,R] = QR(A)
	# print(np.matmul(np.transpose(Q),Q))
	# print(R)
	# print(np.matmul(Q,R))
	return 0


# print(Create_vectors_to_cluster(np.random.rand(4,400)))

def debug_QR():
	lst = []
	for i in range(50):
		A = np.random.rand(i, i)
		(Q, R) = QR(A)
		lst.append((abs(A - np.matmul(Q, R)) < 0.00000001).all())
	print(lst)


def debug_QRIterations():
	lst = []
	for i in range(1):
		# A = np.random.rand(50,3)*0.1
		t1 = time.time()
		A = sklearn.datasets.make_blobs(n_samples=200, n_features=12, centers=12)[0]
		Lnorm = laplacian.Lnorm_matrix(A)
		# T = T_matrix(Lnorm, False, 0)
		t2 = time.time()
		print("The time is approximatly ", t2 - t1, " seconds")
		# (Abar,Qbar) = QR_Iterations(Lnorm)
		# eig_sorted_1 = [Abar[j][j] for j in range(Lnorm.shape[0])]
		# eig_sorted_1.sort()
		# eig_sorted_2 = np.linalg.eigvalsh(Lnorm)
		# eig_sorted_2.sort()
		# lst.append(max([abs(eig_sorted_1[j] - eig_sorted_2[j]) for j in range(len(eig_sorted_1))]))
		# print(np.array(T).shape[1])


# print("")
# print(lst)
# debug_QRIterations()
# debug_QR()
