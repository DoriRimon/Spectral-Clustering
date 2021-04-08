import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib
import numpy as np
from math import comb


def remove_file(filename):
	"""
	Helping method to remove files

	:param filename: str, the name of the file
	"""
	try:
		os.remove(filename)
	except OSError:
		pass


def build_data_text_file(X, centers):
	"""
	Builds the data.txt file

	"""
	remove_file("data.txt")
	with open("data.txt", "w") as data:
		for i in range(len(centers)):
			point = X[i]
			data.write(','.join([str(num) for num in point]) + ',' + str(centers[i]) + '\n')


def build_clusters_text_file(k, spectral_res, kmeans_res):
	"""
	Builds the clusters.txt file

	"""
	remove_file("clusters.txt")
	n = len(spectral_res)
	spectral_indexes = {i: [] for i in range(k)}
	kmeans_indexes = {i: [] for i in range(k)}
	for i in range(n):
		spectral_i = int(spectral_res[i][-1])
		kmeans_i = int(kmeans_res[i][-1])
		spectral_indexes[spectral_i].append(i)
		kmeans_indexes[kmeans_i].append(i)

	with open("clusters.txt", "w") as clusters:
		clusters.write(str(k) + '\n')
		for i in range(k):
			clusters.write(','.join([str(num) for num in spectral_indexes[i]]) + '\n')
		for i in range(k):
			clusters.write(','.join([str(num) for num in kmeans_indexes[i]]) + '\n')


def jaccard(data, result, k):
	"""
	Computes the Jaccard Measure as defined in the assignment

	:param data: array like, shape(n, ) - the data centers
	:param result: matrix, shape(n, d+1) - the clustered observations
	:param k: int, amount of clusters
	:return: 0 <= float <= 1, the jaccard measure

	"""
	data_hash = {i: data[i] for i in range(len(data))}
	n = len(result)
	result_hash = dict()
	for i in range(n):
		result_i = int(result[i][-1])
		result_hash[i] = result_i

	counter = 0
	for i in range(n):
		for j in range(i+1, n):
			if data_hash[i] == data_hash[j] and result_hash[i] == result_hash[j]:
				counter += 1

	result_clusters = {i: 0 for i in range(k)}
	for i in range(n):
		spectral_i = int(result[i][-1])
		result_clusters[spectral_i] += 1

	divider = sum(comb(amount, 2) for amount in result_clusters.values())
	return counter / divider


def build_clusters_pdf_file(K, k, n, d, spectral_res, kmeans_res, centers):
	"""
	Builds the clusters.pdf file

	"""

	remove_file("clusters.pdf")
	print("Jaccard measure for Spectral Clustering: ", jaccard(centers, spectral_res, k))
	print("Jaccard measure for Kmeans: ", jaccard(centers, kmeans_res, k))

	fig = plt.figure()
	cmap = matplotlib.cm.get_cmap('brg')
	colors = [cmap(i) for i in np.linspace(0, 1, k)]
	ax = None
	if d == 2:
		ax = fig.add_subplot()
		for i in range(n):
			ax.scatter(spectral_res[i][0], spectral_res[i][1], c=colors[spectral_res[i][d]])
	elif d == 3:
		ax = fig.add_subplot(projection='3d')
		for i in range(n):
			ax.scatter(spectral_res[i][0], spectral_res[i][1], spectral_res[i][2], c=colors[spectral_res[i][d]])

	plt.savefig("clusters.pdf")
