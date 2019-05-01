import os, sys, datetime, statistics, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
np.random.seed(0)

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

from kmeans import *

TIME_DEBUG = False

def visualize_clustered_examples(dataset_org, num_examples_to_visualize, filename=None, plot=False):
	if (plot==True) and (filename==None):
		raise Exception('Must provide filename if plot set to True')

	CLUSTER_IDS = np.unique(dataset_org[:,1]).astype(int)
	# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e', '#1f77b4', '#81D394']
	# colors = np.random.rand(len(CLUSTER_IDS),1)
	colors=iter(cm.rainbow(np.linspace(0,1,len(CLUSTER_IDS))))

	random_idxs = np.random.randint(0, len(dataset_org), size=num_examples_to_visualize)
	digits_examples = dict()
	for d in CLUSTER_IDS:
		digits_examples[d] = [[],[]]

	for random_idx in random_idxs:
		example_x_coord = dataset_org[random_idx, 2]
		example_y_coord = dataset_org[random_idx, 3]
		example_label = dataset_org[random_idx,1]
		digits_examples[example_label][0].append(example_x_coord)
		digits_examples[example_label][1].append(example_y_coord)

	fig, ax = plt.subplots()

	for d in CLUSTER_IDS:
		x_coords = digits_examples[d][0]
		y_coords = digits_examples[d][1]
		color=next(colors)
		ax.scatter(x_coords, y_coords, c=color, label=d)
	lgd = ax.legend()

	if (plot==True):
		plt.savefig(filename)


def main():
	if len(sys.argv) != 2:
		print('Usage:\n\tpython3 analysis.py dataFilename')
		sys.exit()

	DATA_FILE_PATH = sys.argv[1]

	# dataset components for dataset 1
	# dataset_1_org = np.genfromtxt(DATA_FILE_PATH, delimiter=',', max_rows=200)
	dataset_1_org = np.genfromtxt(DATA_FILE_PATH, delimiter=',')
	dataset_1_index = dataset_1_org[:,0]
	dataset_1_labels = dataset_1_org[:,1]
	dataset_1 = dataset_1_org[:,2:]

	# dataset components for dataset 2
	dataset_2_digits = [2,4,6,7]
	mask  = np.isin(dataset_1_org[:,1], dataset_2_digits)
	dataset_2_org = dataset_1_org[mask]
	dataset_2_org = np.delete(dataset_2_org,0,1)
	idxs = list(range(len(dataset_2_org)))
	dataset_2_org = np.insert(dataset_2_org, 0, idxs, axis=1)
	dataset_2_index = dataset_2_org[:,0]
	dataset_2_labels = dataset_2_org[:,1]
	dataset_2 = dataset_2_org[:,2:]

	# dataset components for dataset 3
	dataset_3_digits = [6,7]
	mask  = np.isin(dataset_1_org[:,1], dataset_3_digits)
	dataset_3_org = dataset_1_org[mask]
	dataset_3_org = np.delete(dataset_3_org,0,1)
	idxs = list(range(len(dataset_3_org)))
	dataset_3_org = np.insert(dataset_3_org, 0, idxs, axis=1)
	dataset_3_index = dataset_3_org[:,0]
	dataset_3_labels = dataset_3_org[:,1]
	dataset_3 = dataset_3_org[:,2:]

	K = [2,4,8,16,32]
	#"""
	########################## PART 1 START ###################################
	print('=== PART 2.1 ===')
	t0 = time.time()
	#### dataset 1 ####
	WC_SSDs, SCs = [], []
	for k in K:
		C, C_with_labels, m = k_means(dataset_1_org, k)
		WC_SSDs.append(round(wc_ssd(C), 3))
		SCs.append(round(silhoutte_coefficient(dataset_1_org,C_with_labels,m), 3))
	# format: plot_<part #>_<wc-ssd | SC>_<dataset #>_<cur datetime>.png
	file_name = 'plot_p_{}_wcssd_ds_{}_{}.png'.format(1,1, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, WC_SSDs)
	ax.set(xlabel='K', ylabel='WC-SSD',
	       title='WC-SSD v.s. K for dataset_1')
	ax.grid()
	fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_sc_ds_{}_{}.png'.format(1,1, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, SCs)
	ax.set(xlabel='K', ylabel='SC',
	       title='SC v.s. K for dataset_1')
	ax.grid()
	fig.savefig(file_name)

	#### dataset 2 ####
	WC_SSDs, SCs = [], []
	for k in K:
		C, C_with_labels, m = k_means(dataset_2_org, k)
		WC_SSDs.append(round(wc_ssd(C), 3))
		SCs.append(round(silhoutte_coefficient(dataset_2_org,C_with_labels,m), 3))
	file_name = 'plot_p_{}_wcssd_ds_{}_{}.png'.format(1,2, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, WC_SSDs)
	ax.set(xlabel='K', ylabel='WC-SSD',
	       title='WC-SSD v.s. K for dataset_2')
	ax.grid()
	fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_sc_ds_{}_{}.png'.format(1,2, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, SCs)
	ax.set(xlabel='K', ylabel='SC',
	       title='SC v.s. K for dataset_2')
	ax.grid()
	fig.savefig(file_name)

	#### dataset 3 ####
	WC_SSDs, SCs = [], []
	for k in K:
		C, C_with_labels, m = k_means(dataset_3_org, k)
		WC_SSDs.append(round(wc_ssd(C), 3))
		SCs.append(round(silhoutte_coefficient(dataset_3_org,C_with_labels,m), 3))
	file_name = 'plot_p_{}_wcssd_ds_{}_{}.png'.format(1,3, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, WC_SSDs)
	ax.set(xlabel='K', ylabel='WC-SSD',
	       title='WC-SSD v.s. K for dataset_3')
	ax.grid()
	fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_sc_ds_{}_{}.png'.format(1,3, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	ax.plot(K, SCs)
	ax.set(xlabel='K', ylabel='SC',
	       title='SC v.s. K for dataset_3')
	ax.grid()
	fig.savefig(file_name)
	if TIME_DEBUG:
		print('Part {} took time: {}'.format(1, time.time()-t0))
	########################## PART 1 END ###################################
	#"""

	########################## PART 3 START ###################################
	#"""
	print('=== PART 2.3 ===')
	t0 = time.time()
	# generate 10 random numbers and then seed the PRNG with those seeds and repeat the above steps every time
	random_seeds = np.random.randint(0, len(dataset_1_org), size=10)
	dataset_1_K_stats = dict() #holds the WCs and SCs for all the 10 random seeds and values of K. format: k: ([wcs], [scs]), k1: ([wcs], [scs])
	dataset_2_K_stats = dict()
	dataset_3_K_stats = dict()

	# TODO: verify if this is the correct way of doing this?
	for random_seed in random_seeds:
		np.random.seed(random_seed)
		# print('Random_seed = {}'.format(random_seed))

		for k in K:
			C, C_with_labels, m = k_means(dataset_1_org, k)
			WC_SSD = round(wc_ssd(C), 3)
			SC = round(silhoutte_coefficient(dataset_1_org,C_with_labels,m), 3)
			dataset_1_K_stats.setdefault(k, ([], []))[0].append(WC_SSD)
			dataset_1_K_stats.setdefault(k, ([], []))[1].append(SC)

			C, C_with_labels, m = k_means(dataset_2_org, k)
			WC_SSD = round(wc_ssd(C), 3)
			SC = round(silhoutte_coefficient(dataset_2_org,C_with_labels,m), 3)
			dataset_2_K_stats.setdefault(k, ([], []))[0].append(WC_SSD)
			dataset_2_K_stats.setdefault(k, ([], []))[1].append(SC)

			C, C_with_labels, m = k_means(dataset_3_org, k)
			WC_SSD = round(wc_ssd(C), 3)
			SC = round(silhoutte_coefficient(dataset_3_org,C_with_labels,m), 3)
			dataset_3_K_stats.setdefault(k, ([], []))[0].append(WC_SSD)
			dataset_3_K_stats.setdefault(k, ([], []))[1].append(SC)


	dataset_1_wc_ssd_stds, dataset_1_wc_ssd_means, dataset_1_sc_stds, dataset_1_sc_means = [], [], [], []
	for stats in dataset_1_K_stats.values():
		dataset_1_wc_ssd_stds.append(statistics.stdev(stats[0]))
		dataset_1_wc_ssd_means.append(statistics.mean(stats[0]))
		dataset_1_sc_stds.append(statistics.stdev(stats[1]))
		dataset_1_sc_means.append(statistics.mean(stats[1]))

	dataset_2_wc_ssd_stds, dataset_2_wc_ssd_means, dataset_2_sc_stds, dataset_2_sc_means = [], [], [], []
	for stats in dataset_2_K_stats.values():
		dataset_2_wc_ssd_stds.append(statistics.stdev(stats[0]))
		dataset_2_wc_ssd_means.append(statistics.mean(stats[0]))
		dataset_2_sc_stds.append(statistics.stdev(stats[1]))
		dataset_2_sc_means.append(statistics.mean(stats[1]))

	dataset_3_wc_ssd_stds, dataset_3_wc_ssd_means, dataset_3_sc_stds, dataset_3_sc_means = [], [], [], []
	for stats in dataset_3_K_stats.values():
		dataset_3_wc_ssd_stds.append(statistics.stdev(stats[0]))
		dataset_3_wc_ssd_means.append(statistics.mean(stats[0]))
		dataset_3_sc_stds.append(statistics.stdev(stats[1]))
		dataset_3_sc_means.append(statistics.mean(stats[1]))

	# format: plot_<part #>_<wc-ssd | SC>_<dataset #>_<cur datetime>.png
	file_name = 'plot_p_{}_{}_ds_{}_{}.png'.format(3,'wcssd',1, str(datetime.datetime.now()))
	# print(file_name)
	fig, ax = plt.subplots()
	line1, = ax.plot(K, dataset_1_wc_ssd_stds, label='stdev')
	line2, = ax.plot(K, dataset_1_wc_ssd_means, label='mean')
	ax.set(xlabel='K', ylabel='Value',
		title='WC-SSD stdev and mean v.s. K for dataset_1')
	ax.legend()
	ax.grid()
	fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_{}_ds_{}_{}.png'.format(3,'sc',1, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(K, dataset_1_sc_stds, label='stdev')
	line2, = ax.plot(K, dataset_1_sc_means, label='mean')
	ax.set(xlabel='K', ylabel='Value',
		title='SC stdev and mean v.s. K for dataset_1')
	ax.legend()
	ax.grid()
	fig.savefig(file_name)

	file_name = 'plot_p_{}_{}_ds_{}_{}.png'.format(3,'wcssd',2, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(K, dataset_2_wc_ssd_stds, label='stdev')
	line2, = ax.plot(K, dataset_2_wc_ssd_means, label='mean')
	ax.set(xlabel='K', ylabel='Value',
		title='WC-SSD stdev and mean v.s. K for dataset_2')
	ax.legend()
	ax.grid()
	fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_{}_ds_{}_{}.png'.format(3,'sc',2, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(K, dataset_2_sc_stds, label='stdev')
	line2, = ax.plot(K, dataset_2_sc_means, label='mean')
	ax.set(xlabel='K', ylabel='Value',
		title='SC stdev and mean v.s. K for dataset_2')
	ax.legend()
	ax.grid()
	fig.savefig(file_name)

	file_name = 'plot_p_{}_{}_ds_{}_{}.png'.format(3,'wcssd',3, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(K, dataset_3_wc_ssd_stds, label='stdev')
	line2, = ax.plot(K, dataset_3_wc_ssd_means, label='mean')
	ax.set(xlabel='K', ylabel='Value',
		title='WC-SSD stdev and mean v.s. K for dataset_3')
	ax.legend()
	ax.grid()
	fig.savefig(file_name)
	# plot for SCs now
	file_name = 'plot_p_{}_{}_ds_{}_{}.png'.format(3,'sc',3, str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(K, dataset_3_sc_stds, label='stdev')
	line2, = ax.plot(K, dataset_3_sc_means, label='mean')
	ax.set(xlabel='K', ylabel='Value',
		title='SC stdev and mean v.s. K for dataset_3')
	ax.legend()
	ax.grid()
	fig.savefig(file_name)
	if TIME_DEBUG:
		print('Part {} took time: {}'.format(3, time.time()-t0))
	#"""
	########################## PART 3 END ###################################

	########################## PART 4 START ###################################
	# """
	print('=== PART 2.4 ===')
	t0 = time.time()
	CHOSEN_K = 8
	# clustering dataset 1
	C, C_with_labels, m = k_means(dataset_1_org, CHOSEN_K)
	NMI = round(nmi(C_with_labels, m, dataset_1_labels), 3)
	print('NMI for dataset {} with K = {}: {}'.format(1, CHOSEN_K, NMI))
	# visualize 1000 randomly selected examples from the custering
	num_examples_to_visualize=1000
	clustered_examples = [[example[1], c, example[2], example[3]] for c in C_with_labels for example in C_with_labels[c]]
	clustered_examples = np.asarray(clustered_examples)
	file_name = 'plot_p_{}_ds_{}_K_{}_{}.png'.format(4,1, CHOSEN_K, datetime.datetime.now())
	visualize_clustered_examples(clustered_examples, num_examples_to_visualize, filename=file_name, plot=True)

	# clustering dataset 2
	CHOSEN_K = 4
	# clustering dataset 1
	C, C_with_labels, m = k_means(dataset_2_org, CHOSEN_K)
	NMI = round(nmi(C_with_labels, m, dataset_2_labels), 3)
	print('NMI for dataset {} with K = {}: {}'.format(2, CHOSEN_K, NMI))
	# visualize 1000 randomly selected examples from the custering
	num_examples_to_visualize=1000
	clustered_examples = [[example[1], c, example[2], example[3]] for c in C_with_labels for example in C_with_labels[c]]
	clustered_examples = np.asarray(clustered_examples)
	file_name = 'plot_p_{}_ds_{}_K_{}_{}.png'.format(4,2, CHOSEN_K, datetime.datetime.now())
	visualize_clustered_examples(clustered_examples, num_examples_to_visualize, filename=file_name, plot=True)

	# clustering dataset 3
	CHOSEN_K = 8
	# clustering dataset 1
	C, C_with_labels, m = k_means(dataset_3_org, CHOSEN_K)
	NMI = round(nmi(C_with_labels, m, dataset_3_labels), 3)
	print('NMI for dataset {} with K = {}: {}'.format(3, CHOSEN_K, NMI))
	# visualize 1000 randomly selected examples from the custering
	num_examples_to_visualize=1000
	clustered_examples = [[example[1], c, example[2], example[3]] for c in C_with_labels for example in C_with_labels[c]]
	clustered_examples = np.asarray(clustered_examples)
	file_name = 'plot_p_{}_ds_{}_K_{}_{}.png'.format(4,3, CHOSEN_K, datetime.datetime.now())
	visualize_clustered_examples(clustered_examples, num_examples_to_visualize, filename=file_name, plot=True)

	if TIME_DEBUG:
		print('Part {} took time: {}'.format(4, time.time()-t0))
	# """
	########################## PART 4 END ###################################


if __name__ == '__main__':
	main()
