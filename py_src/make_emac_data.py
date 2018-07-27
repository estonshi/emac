#!/usr/bin/env python

import numpy as np
import h5py
import argparse

if __name__ == '__main__':

	# parse cmd arguments
	parser = argparse.ArgumentParser(description = \
		"Make compressed emac dataset from original HDF5 file.")
	parser.add_argument("-i", "--input", type=str, help="input HDF5 file")
	parser.add_argument("-p", "--h5path", type=str, help="data path inside input HDF5 file")
	parser.add_argument("-o", "--output", type=str, default="photons.emac", help="output dataset file")
	parser.add_argument("-l", "--selection", type=str, default="all", help="only select part of data, even/odd/all")
	args = parser.parse_args()

	# read h5
	ori_data = h5py.File(args.input, 'r')[args.h5path]
	assert len(ori_data.shape) == 3
	if not (ori_data.dtype == np.dtype('int64') or ori_data.dtype == np.dtype('int32')):
		raise ValueError("The data type of input file is incorrect. It should be 'int64' or 'int32'")

	print("\nCompressing data ..")
	num_data, size_x, size_y = ori_data.shape

	if args.selection == "even":
		print("\nOnly use even index patterns ...")
		selection = range(0, num_data, 2)
	elif args.selection == "odd":
		print("\nOnly use odd index patterns ...")
		selection = np.array(range(0, num_data, 2)) + 1
		if num_data%2 == 1:
			selection = selection[:-1]
	else:
		print("\nUse all patterns ...")
		selection = range(num_data)
	num_data = len(selection)
	data_array = [None] * num_data

	# compress data
	head_array = np.zeros(3, dtype='uint32')
	head_array[0] = num_data
	head_array[1] = size_x
	head_array[2] = size_y

	point = 0
	for i,ind in enumerate(selection):
		d = ori_data[ind].flatten()
		index_1 = np.where(d == 1)[0]
		index_mul = np.where(d > 1)[0]
		one_count = len(index_1)
		mul_count = d[index_mul]
		all_count = 2 + one_count + len(index_mul)*2
		mul = np.hstack([index_mul, mul_count])
		data_array[i] = np.hstack([[all_count, one_count], index_1, mul])
	data_array = np.hstack(data_array).astype('uint32')
	data_array = data_array.flatten()
	print("\nDone. Writing data ...")

	# write to binary file
	newf = open(args.output, 'w')
	head_array.tofile(newf)
	data_array.tofile(newf)
	newf.close()
	print("\nDone.\n")