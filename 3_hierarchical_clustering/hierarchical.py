import os, sys, datetime, statistics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# importing kmeans.py
sys.path.insert(0, '../2_k_means')

from kmeans import *

import numpy as np
np.random.seed(0)

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import single, fcluster, complete, average
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_FIGS = True
TIME_DEBUG = False

def make_cluster_representation(subsamples_org, m):
	
	subsamples_org_2 = np.delete(subsamples_org,0,1)
	idxs = list(range(len(subsamples_org_2)))
	subsamples_org_2 = np.insert(subsamples_org_2, 0, idxs, axis=1)
	subsamples_index = subsamples_org_2[:,0]
	subsamples_labels = subsamples_org_2[:,1]
	subsamples = subsamples_org_2[:,2:]
	
	#print(subsamples_org_2)

	C,C_with_labels = {}, {}
	for i in range(len(m)):
		point = subsamples_org_2[i,2:]
		point_idx, point_label = subsamples_org_2[i,0], subsamples_org_2[i,1]
		C.setdefault(m[i], []).append((point[0], point[1]))
		C_with_labels.setdefault(m[i], []).append(
			[point_idx, point_label, point[0], point[1]]
		)
	
	for c in C_with_labels: C_with_labels[c] = np.asarray(C_with_labels[c])
	
	"""
	print('C = ', C)
	print('C_with_labels =')
	for c in C_with_labels: print('{}: {}'.format(c,C_with_labels[c][:10,:]), len(C_with_labels[c]));
	"""

	return (subsamples_org_2, C, C_with_labels)


def main():
	if len(sys.argv) != 2:
		print('Usage:\n\tpython3 hierarchical_clustering.py dataFilename')
		sys.exit()

	DATA_FILE_PATH = sys.argv[1]

	# dataset components for dataset 1
	#dataset_1_org = np.genfromtxt(DATA_FILE_PATH, delimiter=',', max_rows=20)
	dataset_1_org = np.genfromtxt(DATA_FILE_PATH, delimiter=',')
	dataset_1_index = dataset_1_org[:,0]
	dataset_1_labels = dataset_1_org[:,1]
	dataset_1 = dataset_1_org[:,2:]

	# create the subsamples from the original data
	subsample_digits = [0,1,2,3,4,5,6,7,8,9]
	digits_idxs = dict()
	for d in subsample_digits:
		idxs = np.where(dataset_1_org[:,1] == d)[0]
		digits_idxs.setdefault(d, []).extend(idxs)

	subsamples_org = []
	for d in digits_idxs:
		# print (digits_idxs[d], len(digits_idxs[d]))
		rand_idxs = np.random.randint(0, len(digits_idxs[d]), size=10)
		# print('rand_idxs = {}'.format(rand_idxs))
		rand_samples_rows_idxs = [digits_idxs[d][rand_idx] for rand_idx in rand_idxs]
		# print('rand_samples_rows_idxs = {}'.format(rand_samples_rows_idxs))
		rows = dataset_1_org[rand_samples_rows_idxs,:]
		# print('rows = {}'.format(rows))
		subsamples_org.extend(rows)

	subsamples_org = np.asarray(subsamples_org)
	#print('subsamples_org = {}; len = {}'.format(subsamples_org, len(subsamples_org)))
	#print('---')
	subsamples_index = subsamples_org[:,0]
	subsamples_labels = subsamples_org[:,1]
	subsamples = subsamples_org[:,2:]
	#print(subsamples_index)
	#print(subsamples_labels)
	#print(subsamples)

	y = pdist(subsamples)

	######################## 3.1 START #########################
	print('=== PART 3.1 ===')
	Z_single = single(y)
	fig = plt.figure(figsize=(25, 10))
	dn = dendrogram(Z_single, leaf_rotation=90, leaf_font_size=8, show_contracted=True)
	file_name = 'dendo_{}_{}.png'.format('single_linkage',str(datetime.datetime.now()))
	if SAVE_FIGS:
		fig.savefig(file_name)
	######################## 3.1 END #########################

	######################## 3.2 START #########################
	print('=== PART 3.2 ===')
	Z_complete = complete(y)
	fig = plt.figure(figsize=(25, 10))
	dn = dendrogram(Z_complete, leaf_rotation=90, leaf_font_size=8, show_contracted=True)
	file_name = 'dendo_{}_{}.png'.format('complete_linkage',str(datetime.datetime.now()))
	if SAVE_FIGS:
		fig.savefig(file_name)
	
	Z_average = average(y)
	fig = plt.figure(figsize=(25, 10))
	dn = dendrogram(Z_average, leaf_rotation=90, leaf_font_size=8, show_contracted=True)
	file_name = 'dendo_{}_{}.png'.format('average_linkage',str(datetime.datetime.now()))
	if SAVE_FIGS:
		fig.savefig(file_name)
	######################## 3.2 END #########################
	
	######################## 3.3 START #########################
	print('=== PART 3.3 ===')
	K = [2,4,8,16,32]
	#K = [4]

	########### SINGLE LINKAGE ###########
	WC_SSDs, SCs = [], []
	for k in K:
		#print('Algo: {}, K = {}'.format('single_linkage', k))

		fcluster_assignments = fcluster(Z_single, t=k, criterion='maxclust')
		#print('fcluster_assignments = {}, {}'.format(fcluster_assignments, len(fcluster_assignments)))

		# create the data structures using these assignments to calculate wc_ssd and SC
		m = fcluster_assignments
		subsamples_org_2, C, C_with_labels = make_cluster_representation(subsamples_org, m)
		
		WC_SSDs.append(round(wc_ssd(C), 3))
		SCs.append(round(silhoutte_coefficient(subsamples_org_2,C_with_labels,m), 3))
	
	# format: plot_<part #>_<wc-ssd | SC>_<algo>_<cur datetime>.png
	algo = 'single_linkage'
	file_name = 'plot_p_{}_{}_{}_{}.png'.format(3,'wcssd', algo, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, WC_SSDs)
	ax.set(xlabel='K', ylabel='WC-SSD',
	       title='WC-SSD v.s. K for {}'.format(algo))
	ax.grid()
	if SAVE_FIGS:
		fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_{}_{}_{}.png'.format(3,'sc', algo, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, SCs)
	ax.set(xlabel='K', ylabel='SC',
	       title='SC v.s. K for {}'.format(algo))
	ax.grid()
	if SAVE_FIGS:
		fig.savefig(file_name)

	########### COMPLETE LINKAGE ###########
	WC_SSDs, SCs = [], []
	for k in K:
		#print('Algo: {}, K = {}'.format('complete_linkage', k))

		fcluster_assignments = fcluster(Z_complete, t=k, criterion='maxclust')
		#print('fcluster_assignments = {}, {}'.format(fcluster_assignments, len(fcluster_assignments)))

		# create the data structures using these assignments to calculate wc_ssd and SC
		m = fcluster_assignments
		subsamples_org_2, C, C_with_labels = make_cluster_representation(subsamples_org, m)
		
		WC_SSDs.append(round(wc_ssd(C), 3))
		SCs.append(round(silhoutte_coefficient(subsamples_org_2,C_with_labels,m), 3))
	
	# format: plot_<part #>_<wc-ssd | SC>_<algo>_<cur datetime>.png
	algo = 'complete_linkage'
	file_name = 'plot_p_{}_{}_{}_{}.png'.format(3,'wcssd', algo, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, WC_SSDs)
	ax.set(xlabel='K', ylabel='WC-SSD',
	       title='WC-SSD v.s. K for {}'.format(algo))
	ax.grid()
	if SAVE_FIGS:
		fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_{}_{}_{}.png'.format(3,'sc', algo, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, SCs)
	ax.set(xlabel='K', ylabel='SC',
	       title='SC v.s. K for {}'.format(algo))
	ax.grid()
	if SAVE_FIGS:
		fig.savefig(file_name)

	########### AVERAGE LINKAGE ###########
	WC_SSDs, SCs = [], []
	for k in K:
		#print('Algo: {}, K = {}'.format('average_linkage', k))

		fcluster_assignments = fcluster(Z_average, t=k, criterion='maxclust')
		#print('fcluster_assignments = {}, {}'.format(fcluster_assignments, len(fcluster_assignments)))

		# create the data structures using these assignments to calculate wc_ssd and SC
		m = fcluster_assignments
		subsamples_org_2, C, C_with_labels = make_cluster_representation(subsamples_org, m)
		
		WC_SSDs.append(round(wc_ssd(C), 3))
		SCs.append(round(silhoutte_coefficient(subsamples_org_2,C_with_labels,m), 3))
	
	# format: plot_<part #>_<wc-ssd | SC>_<algo>_<cur datetime>.png
	algo = 'average_linkage'
	file_name = 'plot_p_{}_{}_{}_{}.png'.format(3,'wcssd', algo, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, WC_SSDs)
	ax.set(xlabel='K', ylabel='WC-SSD',
	       title='WC-SSD v.s. K for {}'.format(algo))
	ax.grid()
	if SAVE_FIGS:
		fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_{}_{}_{}.png'.format(3,'sc', algo, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, SCs)
	ax.set(xlabel='K', ylabel='SC',
	       title='SC v.s. K for {}'.format(algo))
	ax.grid()
	if SAVE_FIGS:
		fig.savefig(file_name)

	######################## 3.3 END #########################
	
	######################## 3.5 START #########################
	print('=== PART 3.5 ===')
	
	########### SINGLE LINKAGE ###########
	algo = 'single_linkage'
	CHOSEN_K=8
	fcluster_assignments = fcluster(Z_single, t=CHOSEN_K, criterion='maxclust')
	m = fcluster_assignments
	subsamples_org_2, C, C_with_labels = make_cluster_representation(subsamples_org, m)
	NMI = round(nmi(C_with_labels, m, subsamples_labels), 3)
	print('NMI for algo/distance measure = {} with K = {}: {}'.format(algo, CHOSEN_K, NMI))

	########### COMPLETE LINKAGE ###########
	algo = 'complete_linkage'
	CHOSEN_K=8
	fcluster_assignments = fcluster(Z_complete, t=CHOSEN_K, criterion='maxclust')
	m = fcluster_assignments
	subsamples_org_2, C, C_with_labels = make_cluster_representation(subsamples_org, m)
	NMI = round(nmi(C_with_labels, m, subsamples_labels), 3)
	print('NMI for algo/distance measure = {} with K = {}: {}'.format(algo, CHOSEN_K, NMI))
	
	########### AVERAGE LINKAGE ###########
	algo = 'average_linkage'
	CHOSEN_K=8
	fcluster_assignments = fcluster(Z_average, t=CHOSEN_K, criterion='maxclust')
	m = fcluster_assignments
	subsamples_org_2, C, C_with_labels = make_cluster_representation(subsamples_org, m)
	NMI = round(nmi(C_with_labels, m, subsamples_labels), 3)
	print('NMI for algo/distance measure = {} with K = {}: {}'.format(algo, CHOSEN_K, NMI))

	######################## 3.5 END #########################


if __name__ == '__main__':
	main()
