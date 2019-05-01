import sys
import math
import time

import numpy as np
np.random.seed(0)

import scipy.spatial.distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TIME_DEBUG = False
DEBUG = False

def k_means(data, K):
	"""
	Implements the K-means algorithm following the slides' notation.
	"""
	MAX_ITERATIONS = 50
	N = len(data)
	C,C_with_labels,m = {}, {}, [None]*N
	iters = 0

	# initialize cluster representatives
	random_idxs = np.random.randint(0, N, size=K)
	# print('K = {}, random_idxs = {}'.format(K,random_idxs))
	# return(0,0,0)

	for i in range(len(random_idxs)):
		point = data[random_idxs[i],2:]
		point_idx, point_label = data[random_idxs[i],0], data[random_idxs[i],1]
		C.setdefault(i, []).append((point[0], point[1]))
		C_with_labels.setdefault(i, []).append(
			[point_idx, point_label, point[0], point[1]]
		)

	# in case there is a centroid is not the closest to ANY point, then there will be no points in its array and the mean will be nan. to prevent this, I store the old clusters and check if the a cluster in the updated clusters is empty, then just assign it the old value i.e. leave it unchanged
	C_old = C.copy()

	while (iters < MAX_ITERATIONS):
		cluster_means = dict()
		for c in C:
			if (len(C[c]) > 0): # in case there is no point belonging to a cluster
				cluster_means[c] = get_cluster_mean(C[c])
				C_old[c] = C[c]
			else: # use the old centroid and do not update its value
				cluster_means[c] = get_cluster_mean(C_old[c])

		# print('C =')
		# for c in C: print(C[c], len(C[c]));
		# print ('cluster_means = {}'.format(cluster_means))
		# input('...\n')

		# reassign points in D to closest cluster mean
		C = {c: [] for c in C}
		C_with_labels = {c: [] for c in C}
		for i in range(N):
			point = data[i,2:]
			point = (point[0], point[1])
			point_idx, point_label = data[i,0], data[i,1]

			# get the closest cluster to this point
			closest_dist = float('inf')
			closest_cluster = None
			for c in cluster_means:
				dist = distance(point, cluster_means[c])
				if dist < closest_dist:
					closest_dist = dist
					closest_cluster = c

			# print ('Closest cluster/centroid to the point {} is {}:{}'.format(point, closest_cluster, cluster_means[closest_cluster]))

			# assign this point to the closest cluster
			C[closest_cluster].append(point)
			C_with_labels[closest_cluster].append(
				[point_idx, point_label, point[0], point[1]]
			)
			# update m s.t. m_i is cluster ID of ith point in D
			m[i] = closest_cluster

			# print ('Closest centroid to the point {} is {}'.format(point, m[i]))
			# input('...')


		# if there is no change in the centroids, then stop
		new_cluster_means = {c: get_cluster_mean(C[c]) for c in C}
		if (centroids_did_not_change(new_cluster_means, cluster_means)):	
			if DEBUG:
				print('Old cluster_means = {}'.format(cluster_means))
				print('New cluster_means = {}'.format(new_cluster_means))
				print('No change in the centroids, breaking out of the loop.')
			break


		# if this is the last iteration then break out of the loop after the assignment of points to the cluster and before recomputing the new centroids
		if iters == MAX_ITERATIONS-1:
			if DEBUG:
				print('this is the last ({}) iteration, break out of the loop after the assignment of points to the cluster and before recomputing the new centroids'.format(iters))
			break


		iters += 1

	if DEBUG:
		print('iters = ',iters)

	for c in C_with_labels: C_with_labels[c] = np.asarray(C_with_labels[c])
	
	return (C, C_with_labels, m)


def centroids_did_not_change(new, old):
	did_not_change = True
	for c in new:
		if (c not in old) or (not np.array_equal(new[c], old[c])):
			did_not_change = False
	return did_not_change


def get_cluster_mean(cluster_points):
	"""
	cluster_points: all the points that belong to this cluster_points
	"""
	return np.mean(cluster_points, axis=0)


def distance(p1, p2):
	"""
	Computes the Euclidean Distance between the two points p1 and p2
	"""
	# print ('p1 = {} p2 {}'.format(p1,p2))
	return np.sqrt(((p2[0]-p1[0])**2) + ((p2[1]-p1[1])**2))


def wc_ssd(C):
	"""
	Computes the within cluster sum of squared distances
	"""
	sum_ = 0
	for c_k in C.values(): # for each of the k clusters in C
		r_k = get_cluster_mean(c_k) # kth cluster's centroid
		sum_c_k = 0
		for x_i in c_k: # for every point in this cluster, compute the distance sqaure
			sum_c_k += distance(x_i, r_k)**2
		sum_ += sum_c_k
	return sum_


def silhoutte_coefficient(data_org, C_with_labels, m):
	"""
	Computes the Silhoutte Coefficient of the clustering C using the cluster assignments m
	"""
	# compute SC using numpy vectorization
	S = []
	pairwise_dists = scipy.spatial.distance.pdist(data_org[:,2:], 'euclidean')
	pairwise_dists = scipy.spatial.distance.squareform(pairwise_dists)
	# for i in range(len(data_org)):
	for i in data_org[:,0].astype(int):
		membership_cluser = C_with_labels[m[i]]
		cluster_neighbor_idxs = membership_cluser[:,0]
		cluster_neighbor_idxs = cluster_neighbor_idxs.astype(int)
		# print(cluster_neighbor_idxs)
		# mask = np.isin(org_idx_mappings_2[:,0], cluster_neighbor_idxs)
		# print(mask)
		# print(org_idx_mappings_2[mask])
		# cluster_neighbor_idxs = org_idx_mappings_2[mask][:,1]
		# print(cluster_neighbor_idxs)

		other_clusters_neighbor_idxs = []
		for c in C_with_labels:
			if (len(C_with_labels[c]) > 0) and (c != m[i]):
				other_cluster = C_with_labels[c]
				other_clusters_neighbor_idxs.extend(other_cluster[:,0])
		other_clusters_neighbor_idxs = list(map(int, other_clusters_neighbor_idxs))
		# mask = np.isin(org_idx_mappings_2[:,0], other_clusters_neighbor_idxs)
		# other_clusters_neighbor_idxs = org_idx_mappings_2[mask][:,1]

		intra_cluster_distances = pairwise_dists[i, cluster_neighbor_idxs]
		inter_cluster_distances = pairwise_dists[i, other_clusters_neighbor_idxs]
		# input('...')
		A = intra_cluster_distances.mean()
		B = inter_cluster_distances.mean()

		S_i = (B-A) / max(A,B)
		S.append(S_i)
	SC_new = sum(S) / len(S)

	return SC_new


# might need to change this to make it adaptable to hierarchical clustring
def nmi(C, m, labels):
	"""
	Computes the Normalized Mutual Information Gain for the clustering C and the class labels
	Source: https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf 

	NMI(C,G) = I(C,G) / (H(C) + H(G))
	C = clusters
	G = class labels
	I(C,G) 	= mutual information gain b/w C and G
			= H(G) = H(G|C) (entropy of class labels within each cluster), (if a label is not in a cluster, then give it 0 probablity)
	H(G) 	= entropy of class labels 
	H(C)	= entropy of cluster labels 
	"""

	# calculate H(G) - the entropy of the class labels
	# print('labels = ', labels)
	class_labels,cnts = np.unique(labels,return_counts = True)
	total_num_examples = np.sum(cnts)
	# print('class_labels = {}, cnts = {}'.format(class_labels, cnts))
	H_G=0.0
	for i in range(len(class_labels)):
		probability=float(cnts[i])/total_num_examples
		H_G += -(probability*math.log(probability,2))
	# print ('entropy H_G =',H_G)

	# calculate H(C) - the entropy of the clusters
	H_C = 0.0
	for c in C:
		probability = len(C[c]) / total_num_examples
		H_C += -(probability*math.log(probability,2))
	# print ('entropy H_C =',H_C)	

	# calculate I(C,G)
	H_G_Cs = [] # the conditional entropies for all the clusters
	for c in C:
		cluster = C[c]
		total_num_examples_c = len(C[c]) # the total number of examples/points in this cluster
		cluster_labels, cnts = np.unique(cluster[:,1], return_counts=True) # the labels that are in this cluster and their counts
		cluster_labels_cnts = {cluster_labels[i]: cnts[i] for i in range(len(cluster_labels))}

		# print('For cluster {}, cluster_labels = {}, cnts = {} and cluster_labels_cnts = {}'.format(c,cluster_labels, cnts, cluster_labels_cnts))

		H_G_C = 0.0
		for class_label in class_labels:
			if class_label in cluster_labels:
				# print('class label {} is present in this cluster'.format(class_label))
				probability = cluster_labels_cnts[class_label]/total_num_examples_c
				H_G_C += -(probability*math.log(probability,2))
			else:
				# print('class label {} is not present in this cluster'.format(class_label))
				probability = 0
				H_G_C += probability

		# multiply H_G_C with probability of this cluster (see example)
		H_G_C = (1/len(C))*H_G_C
		# print ('For cluster {}, entropy H_G_C = {}'.format(c, H_G_C))
		H_G_Cs.append(H_G_C)

	I_C_G = H_G - sum(H_G_Cs)
	NMI = I_C_G / (H_C + H_G)

	return NMI


def main():
	if len(sys.argv) != 3:
		print('Usage:\n\tpython3 kmeans.py dataFilename K')
		sys.exit()

	DATA_FILE_PATH = sys.argv[1]
	K = int(sys.argv[2])

	#embedded_data_org = np.genfromtxt(DATA_FILE_PATH, delimiter=',', max_rows=100)
	embedded_data_org = np.genfromtxt(DATA_FILE_PATH, delimiter=',')
	embedded_data_index = embedded_data_org[:,0]
	embedded_data_labels = embedded_data_org[:,1]
	embedded_data = embedded_data_org[:,2:]

	t0 = time.time()
	C, C_with_labels, m = k_means(embedded_data_org, K) # following the slides' notation
	if TIME_DEBUG:
		print('K-means took time {}'.format(time.time()-t0))

	#print('C = ', C)
	#print('C_with_labels =')
	#for c in C_with_labels: print('{}: {}'.format(c,C_with_labels[c][:10,:]), len(C_with_labels[c]));
	# print('m = {}'.format(m))
	# print('labels = {}'.format(embedded_data_labels))

	t0 = time.time()
	print('WC-SSD: {}'.format(round(wc_ssd(C), 3) ))
	if TIME_DEBUG:
		print('WC-SSD calculation took time {}'.format(time.time()-t0))
	
	t0 = time.time()
	print('SC: {}'.format(round(silhoutte_coefficient(embedded_data_org, C_with_labels, m), 3)))
	if TIME_DEBUG:
		print('SC calculatiion took time {}'.format(time.time()-t0))
	
	t0 = time.time()
	print('NMI: {}'.format(round(nmi(C_with_labels, m, embedded_data_labels), 3)))
	if TIME_DEBUG:
		print('NMI calculatiion took time {}'.format(time.time()-t0))


if __name__ == '__main__':
	main()

