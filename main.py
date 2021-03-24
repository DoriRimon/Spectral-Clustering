import spectral
import random
from sklearn.datasets import make_blobs

"""
Desc:   This is the main file - the glue of the program

Note:   Throughout the documentation we used names that were defined in the assignment
		such as n, k, d and matrices names such as W, D, Lnorm..
		The goal was to make the documentation easy to understand and coherent with the
		assignment.
"""

# TODO - error handling
# TODO - edge values
# TODO - need bottlenecks to C
# TODO - remove unrelated imports
# TODO - remove debug code
# TODO - remove unnecessary prints
# TODO - check all the shapes in the documentation
# TODO - document C code properly
# TODO - handle existing files when creating data files
# TODO - circular imports (tasks -> main -> mykmeansssp)

# TODO - find actual (K, n) of the maximum capacity

TWO_DIM_MAX_CAPACITY = (5, 100)  # Our (K, n) of the maximum capacity for 2 dimensional vectors
THREE_DIM_MAX_CAPACITY = (10, 200)  # Our (K, n) of the maximum capacity for 3 dimensional vectors
X = centers = None


def create_data(n, d, k, Random):
	"""
	Creates the data using the sklearn.datasets.make blobs API

	:param n: int, number of points
	:param d: int, dimension of each point
	:param k: int , amount of centers
	:param Random: boolean, the Random variable described in the assignment

	"""
	global X, centers

	if d == 2:
		max_k, max_n = TWO_DIM_MAX_CAPACITY
	else:
		max_k, max_n = THREE_DIM_MAX_CAPACITY

	if Random:
		n = random.randint(int(max_n / 2), max_n)
		k = random.randint(int(max_k / 2), max_k)

	X, centers = make_blobs(n_samples=n, n_features=d, centers=k)
	return k


def build_data_text_file():
	"""
	Builds the data.txt file

	"""
	with open("data.txt", "w") as data:
		for i in range(len(centers)):
			point = X[i]
			data.write(','.join([str(num) for num in point]) + ',' + str(centers[i]) + '\n')


def build_clusters_text_file(k, spectral_res, kmeans_res):
	"""
	Builds the clusters.txt file

	"""
	n = len(spectral_res)
	spectral_indexes = {i: [] for i in range(k)}
	kmeans_indexes = {i: [] for i in range(k)}
	for i in range(n):
		spectral_i = spectral_res[i][-1]
		kmeans_i = kmeans_res[i][-1]
		spectral_indexes[spectral_i].append(i)
		kmeans_indexes[kmeans_i].append(i)

	with open("clusters.txt", "w") as clusters:
		clusters.write(str(k) + '\n')
		for i in range(k):
			clusters.write(','.join(spectral_indexes[i]) + '\n')
		for i in range(k):
			clusters.write(','.join(kmeans_indexes[i]) + '\n')


def build_clusters_pdf_file(K, k, n, d, spectral_res, kmeans_res):
	"""
	Builds the clusters.pdf file

	"""
	pass


def print_max_capacity():
	"""
	Prints the max capacity of the program, as defined in the assignment

	"""
	print('maximum capacity for a run of under 5 minutes:')
	print(' --> given vectors of 2 dimensions:')
	print('     K = {}. n = {}'.format(TWO_DIM_MAX_CAPACITY[0], TWO_DIM_MAX_CAPACITY[1]))
	print(' --> given vectors of 3 dimensions:')
	print('     K = {}. n = {}'.format(THREE_DIM_MAX_CAPACITY[0], THREE_DIM_MAX_CAPACITY[1]))


def main(k, n, Random):
	"""
	The main function of the code

	:param k: int , amount of centers
	:param n: int, number of points
	:param Random: boolean, the Random variable described in the assignment

	"""
	# import is done here inorder to prevent calling the kmeans module (from tasks.py) before build.
	import kmeans

	k, n = int(k), int(n)
	print_max_capacity()

	d = random.randint(2, 3)
	K = create_data(n, d, k, Random)  # this returned K is the one that was used in the data generation

	build_data_text_file()

	"""
	Both The Spectral and the Kmeans algorithms end up sending vectors to the kmeans module.
	The only difference is which vectors.
	Here those vectors (spectral_observations & kmeans_observations) are being computed.
	
	Note:   The if else section here is needed because
	        --> if Random=True, k should be computed by the
	            eigengap heuristic, and that k will be of use in the following code.
	        --> else, the actual k is needed in the when computing U in the spectral module.
	"""
	if Random:
		spectral_observations, k = spectral.main(X)
	else:
		spectral_observations, _ = spectral.main(X, k)  # else => k = K

	kmeans_observations = X

	spectral_res = kmeans.main(spectral_observations, k, n, k)  # TODO - send correct params
	kmeans_res = kmeans.main(kmeans_observations, k, n, d)

	build_clusters_text_file(k, spectral_res, kmeans_res)
	build_clusters_pdf_file(K, k, n, d, spectral_res, kmeans_res)
