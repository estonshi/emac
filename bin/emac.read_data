#!/usr/bin/env python

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
	
	# parse cmd arguments
	parser = argparse.ArgumentParser(description = \
		"Uncompress emac dataset and save to HDF5 file.")
	parser.add_argument("-i", "--input", type=str, help="input emac dataset file")
	parser.add_argument("-o", "--output", type=str, default="photons.h5", help="output HDF5 file")
	parser.add_argument("-s", "--showindex", type=int, default=-1, help="only show pattern with index you give, not save to HDF5")
	args = parser.parse_args()

	# read emac dataset
	dataset = np.fromfile(args.input, dtype='uint32')
	
	# uncompressing
	print("\nUncompressing ...")
	num_data, size_x, size_y = dataset[0:3]
	point = 3

	if args.showindex >= 0:
		print("Show index="+str(args.showindex)+"\n")
		for i in range(num_data):
			all_count = dataset[point]
			if i != args.showindex:
				point = point + all_count
			else:
				pattern = np.zeros((size_x * size_y), dtype='uint32')
				d = dataset[point : point + all_count]
				one_count = d[2:2+d[1]]
				pattern[one_count] = 1
				mul_count = d[2+d[1]:].reshape((2,-1))
				pattern[mul_count[0]] = mul_count[1]
				pattern = pattern.reshape((size_x , size_y))
				plt.imshow(np.log(1+pattern))
				plt.show()
		sys.exit(0)

	patterns = np.zeros((num_data, size_x * size_y), dtype='uint32')
	for i in range(num_data):
		all_count = dataset[point]
		d = dataset[point : point + all_count]
		one_count = d[2:2+d[1]]
		patterns[i][one_count] = 1
		mul_count = d[2+d[1]:].reshape((2,-1))
		patterns[i][mul_count[0]] = mul_count[1]
		point = point + all_count
	patterns = patterns.reshape((num_data, size_x, size_y))
	print("\nDone. Save to h5 ...")

	# save file
	save_f = h5py.File(args.output, 'w')
	save_f.create_dataset('photons', data=patterns, chunks=True, compression="gzip")
	save_f.close()