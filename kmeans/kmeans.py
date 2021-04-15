import numpy as np
import mykmeanssp as mk

"""
Desc:   This is the main file regarding the kmeans algorithm.
"""


MAX_ITER = 300

observations = []
clusters = []
centroids = []


def norm_func(x, u):
	"""
	computes norm of two vectors, as needed in the assignment

	:param x: array like vector, shape(d, )
	:param u: array like vector, shape(d, )
	:return: float - norm value

	"""
	return np.power(np.linalg.norm(x - u, axis=0), 2)


def k_means_pp(k, n):
	"""
	This function was defined in h.w 2
	Overall:    this is the Kmeans++ part that chooses the initial centroids
			    for the algorithm

	:param k: int
	:param n: int

	"""
	np.random.seed(0)
	nums = [i for i in range(n)]
	rand = np.random.choice(nums, 1)
	centroids.append(int(rand[0]))
	min_arr = [norm_func(x, observations[centroids[-1]]) for x in observations]

	for j in range(1, k):
		latest_centroid = observations[centroids[-1]]

		new_dist = np.power(observations - latest_centroid, 2).sum(axis=1)
		for i in range(n):
			temp = min(min_arr[i], new_dist[i])
			min_arr[i] = temp

		s = sum(min_arr)
		probs = [m / s for m in min_arr]
		u = np.random.choice(nums, 1, p=probs)
		centroids.append(int(u[0]))


def main(X, k, n, d):
	"""
	The main function of the kmeans++ algorithm

	:param X: The observations, shape(n, d)
	:param k: int, amount of clusters
	:param n: int, amount of observations
	:param d: int, dimension of each observation
	:return: list, shape(n, d+1) - the observations. At the end of each
			observation (at index d) the cluster of the observation appears.

	"""
	global observations
	observations = X
	k_means_pp(k, n)
	return mk.kmeans([observations.tolist(), centroids, k, n, d, MAX_ITER])
