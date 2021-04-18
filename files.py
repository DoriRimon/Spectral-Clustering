import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

"""
Desc:   This file handles all the work related to the output files of the program
"""


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


def jaccard(data, result):
	"""
	Computes the Jaccard Measure as defined in the assignment

	:param data: array like, shape(n, ) - the data centers
	:param result: matrix, shape(n, d+1) - the clustered observations
	:return: 0 <= float <= 1, the jaccard measure

	"""
	data_hash = {i: data[i] for i in range(len(data))}
	n = len(result)
	result_hash = dict()
	for i in range(n):
		result_i = int(result[i][-1])
		result_hash[i] = result_i

	counter = 0
	divider = 0

	for i in range(n):
		for j in range(i + 1, n):
			if data_hash[i] == data_hash[j] and result_hash[i] == result_hash[j]:
				counter += 1
			if data_hash[i] == data_hash[j] or result_hash[i] == result_hash[j]:
				divider += 1

	return counter / divider


def build_clusters_pdf_file(K, k, n, d, spectral_res, kmeans_res, centers):
	"""
	Builds the clusters.pdf file

	"""

	remove_file("clusters.pdf")
	jaccard_spectral = jaccard(centers, spectral_res)
	jaccard_kmeans = jaccard(centers, kmeans_res)
	print("Jaccard measure for Spectral Clustering: ", jaccard_spectral)
	print("Jaccard measure for Kmeans: ", jaccard_kmeans)

	fig = None
	cmap = matplotlib.cm.get_cmap('brg')
	colors = [cmap(i) for i in np.linspace(0, 1, k)]
	(ax1, ax2) = (None, None)
	if d == 2:
		# fig, (ax1, ax2) = plt.subplots(1, 2)
		fig = plt.figure(figsize=(10, 10))
		ax1 = fig.add_subplot(2, 2, 1)
		ax2 = fig.add_subplot(2, 2, 2)
	elif d == 3:
		fig = plt.figure(figsize=(10, 10))
		ax1 = fig.add_subplot(2, 2, 1, projection='3d')
		ax2 = fig.add_subplot(2, 2, 2, projection='3d')
	for i in range(n):
		coordinates = [kmeans_res[i][k] for k in range(d)]
		ax1.scatter(*coordinates, color=colors[int(kmeans_res[i][d])])
		ax1.set_title('K-means')
		ax2.scatter(*coordinates, color=colors[int(spectral_res[i][k])])
		ax2.set_title('Normalized Spectral Clustering')
		s = f'Data was generated from the values:\nn = {n}, k = {K}\n \
		    The k that was used for both algorithms was {k}\n \
		    The Jaccard measure for Spectral Clustering: {jaccard_spectral}\n \
		    The Jaccard measure for K-means: {jaccard_kmeans}'
		fig.text(0.5, 0.3, s, ha='center', fontsize='16')
	plt.savefig("clusters.pdf")
