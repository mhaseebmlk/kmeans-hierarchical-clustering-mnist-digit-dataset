import numpy as np
np.random.seed(0)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TIME_DEBUG = False
DEBUG = False

PATH_TO_DATA = '../data/'
RAW_DATA_FILE_NAME = 'digits-raw.csv'
EMBEDDED_DATA_FILE_NAME = 'digits-embedding.csv'

RAW_DATA_PATH = PATH_TO_DATA + RAW_DATA_FILE_NAME
EMBEDDED_DATA_FILE_PATH = PATH_TO_DATA + EMBEDDED_DATA_FILE_NAME

def img_to_matrix(pixels):
	N = 28
	j = 0
	matrix = []
	for i in range(N):
		row_i = pixels[j:j+N]
		matrix.append(row_i)
		j = j+N
	return matrix

def main():
	if DEBUG:
		print('Reading the data...')

	raw_data_org = np.genfromtxt(RAW_DATA_PATH, delimiter=',')
	raw_data_index = raw_data_org[:,0]
	raw_data_labels = raw_data_org[:,1]
	raw_data = raw_data_org[:,2:]

	embedded_data_org = np.genfromtxt(EMBEDDED_DATA_FILE_PATH, delimiter=',')
	embedded_data_index = embedded_data_org[:,0]
	embedded_data_labels = embedded_data_org[:,1]
	embedded_data = embedded_data_org[:,2:]

	NUM_DIGITS = 10
	digits_picked = []
	digits_visualizations = dict()
	# randomly pick a digit from each class and then visualize it
	for d in range(NUM_DIGITS):
		# get all the indexes where this digit class occurs
		idxs = np.where(raw_data_labels == d)[0]
		# randomly pick an index from the list
		rand_idx = idxs[np.random.randint(0, len(idxs))]
		digit_pixels = raw_data[rand_idx,:]
		plt.matshow(img_to_matrix(digit_pixels)) 
		file_name = 'visualization_{}.png'.format(d)
		plt.savefig(file_name)

	total_num_examples = len(embedded_data)
	random_idxs = np.random.randint(0, total_num_examples, size=1000)
	
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e', '#1f77b4', '#81D394']
	digits_examples = dict()
	digits_colors = dict()
	for d in range(NUM_DIGITS):
		digits_examples[d] = [[],[]]
		digits_colors[d] = colors[d]

	for random_idx in random_idxs:
		example_x_coord = embedded_data[random_idx, 0]
		example_y_coord = embedded_data[random_idx, 1]
		example_label = embedded_data_labels[random_idx]
		digits_examples[example_label][0].append(example_x_coord)
		digits_examples[example_label][1].append(example_y_coord)

	fig, ax = plt.subplots()
	for d in range(NUM_DIGITS):
		x_coords = digits_examples[d][0]
		y_coords = digits_examples[d][1]
		ax.scatter(x_coords, y_coords, c=digits_colors[d], label=d)
	# lagend, xlabel, ylabel, title?
	ax.legend()
	plt.savefig('scatter_plot_1.png')

if __name__ == '__main__':
	main()

