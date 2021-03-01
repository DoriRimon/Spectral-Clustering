import numpy as np
import argparse
import pandas
import sklearn
from EigenValues import Create_vectors_to_cluster
import mykmeanssp as mk

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int)
parser.add_argument("N", type=int)
parser.add_argument("Random", type=bool)
#parser.add_argument("MAX_ITER", type=int)
#parser.add_argument("filename", type=str)

args = parser.parse_args()
k, N, Random, d, MAX_ITER = args.k, args.N, args.Random, np.random.randint(2,4), 300

assert(k > 0 and N > 0 and k < N) #should be replaced

observations = []
clusters = []
centroids = []

#df = pandas.read_csv(filename, header=None)
#observations = df.to_numpy(dtype=np.float_)


def norm_func(x, u):
	return np.power(np.linalg.norm(x - u, axis=0), 2)

def Spectral_Clustering_Vectors(X):
        T = Create_vectors_to_cluster(X, Random, k)
        observations = T #each row of T is an observations

def k_means_pp(k_min,k_max, spectral): #recives the bounderies for under 5 minutes preformance
        K = 0 #changed here compare to ex2
        if(not Random):
             K = k
        else:
             K = np.random.randint(k_min,k_max+1)
        
        X = sklearn.datasets.make_blobs(n_samples = N, n_features = d, centers = K) #these are the vectors we need to cluster

        if (spectral):
             Spectral_Clustering_Vectors(X) #initiate observations for spectral clustering
        else:
             observations = X #right format?
        
	np.random.seed(0)
	nums = [i for i in range(N)]
	rand = np.random.choice(nums, 1)
	centroids.append(int(rand[0]))
	min_arr = [norm_func(x, observations[centroids[-1]]) for x in observations]

	for j in range(1, K):
		latest_centroid = observations[centroids[-1]]

		new_dist = np.power(observations - latest_centroid, 2).sum(axis=1)
		for i in range(N):
			temp = min(min_arr[i], new_dist[i])
			min_arr[i] = temp

		s = sum(min_arr)
		probs = [m / s for m in min_arr]
		u = np.random.choice(nums, 1, p=probs)
		centroids.append(int(u[0]))

	return K #change here compare to ex2

def print_centroids():
	print(','.join(list(map(str, centroids))))

def main(k_min,k_max): #there are to many files in the end of the program, need to slightly change interface
        clusters_rep = open("clusters.txt",mode = 'w') #check the first line (K value)
        data_rep = open("data.txt", mode = 'w')
        #data_rep = create_data_file(data_rep,X)
        
        K = k_means_pp(k_min,k_max,True) #spectral
	#print_centroids()
	[obs_spec,clusters_spec] = mk.kmeans([observations.tolist(), centroids, K, N, d, MAX_ITER]) #getting from C the observations,clusters
	#clusters_spectral = open("Clusters_rep.txt",'r')
	#clusters_rep.write(K,"/n") #first line OK?
	#file_cpy(clusters_spectral,cluster_rep)

	k_means_pp(k_min,k_max,False) #kmeans
	#print_centroids()
	[obs_reg,clusters_reg] = mk.kmeans([observations.tolist(), centroids, K, N, d, MAX_ITER])
	#clusters_kmeans = open("Clusters_rep.txt",'r')
	#file_cpy(clusters_kmeans,clusters_rep)

########################################################### Currently not in use   #######################################################################
def file_cpy(fin,fout): #new function for the final project, need to match to files that are already open
                lines = fin.readlines()
                lines = [l for l in lines if "ROW" in l]
                fout.writelines(lines)

def create_data_file(data_rep,X): #new function for the final project, need to fix (add cluster number in the end)
        for i in range(N):
                if(i > 0):
                        data_rep.write("\n")
                for j in range(d):
                        data_rep.write(X[i][j])
##########################################################################################################################################################	

main()






	
########################################################### PLAN FOR NEW INTERFACE #######################################################################
#the interface should be changed. the m function in C will return two python databases from which we will generate the final files,
#instead of creating more intermidiate files. the best way to do it is to simply return from the C program the clusters and observations lists,
#then extract from this information the needed files.
##########################################################################################################################################################
