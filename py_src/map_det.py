#!/usr/bin/env python

import numpy as np
import argparse
import configparser
import os
import sys
from spipy.analyse import q
from spipy.image import radp, preprocess

if __name__ == '__main__':
	
	# parse cmd arguments
	parser = argparse.ArgumentParser(description = \
		"Mappping detector pixels onto Ewald Sphere.")
	parser.add_argument("-c", "--config", type=str, help="path of config.ini")
	parser.add_argument("-s", "--saveto", type=str, default="", help="save results to a text file")
	args = parser.parse_args()

	# read config
	cf = configparser.ConfigParser()
	cf.read(args.config)
	detd = cf.getfloat("input", "detd")
	pix_s = cf.getfloat("input", "pix_s")
	lamb = cf.getfloat("input", "lambda")
	size = np.array(cf.get("input", "size").split(',')).astype(int)
	mask = cf.get("input", "mask")
	ron = cf.getint("adjust", "ron")
	downsampl = cf.getfloat("adjust", "downsampl")
	polarization = cf.get("adjust", "polarization")
	center = cf.get("input", "center")
	if len(args.saveto) == 0:
		det_q_output = cf.get("input", "det_q")
	else:
		det_q_output = args.saveto

	if center != "None":
		center = np.array(center.split(',')).astype(float)
	else:
		center = (size-1)/2.0
	if downsampl <= 0:
		print("\ndetect downsampling rate <= 0, use 1 as default")
		downsampl = 1.
	else:
		downsampl = float(downsampl)
	if not os.path.exists(os.path.dirname(det_q_output)):
		raise RuntimeError("Your output det_q path is invalid")
	if mask != "None" and not os.path.exists(mask):
		raise RuntimeError("Your mask path is invalid")

	datamask = np.zeros(size, dtype=int)
	# ron
	if ron > 0:
		cir_ron = radp.circle(2, ron) + center.astype(int)
		try:
			datamask[cir_ron[:,0], cir_ron[:,1]] = 1
		except:
			raise RuntimeError("Your given 'ron' or 'center' value (config.ini) is not good. Exit")
	# usermask
	if mask != "None":
		usermask = np.load(mask).astype(int)
		if usermask.shape[0] != size[0] or usermask.shape[1] != size[1]:
			raise RuntimeError("Your mask size does not match with data")
		datamask[np.where(usermask >= 1)] = 2

	# calculate q info
	q_coor, qmax, qunit, q_len = q.ewald_mapping(detd, lamb, pix_s, size, center)
	q_coor /= downsampl # (3, Nx, Ny)
	qunit *= downsampl
	q_len = int(qmax / qunit) * 2 + 3

	# calculate correction factor
	correction = preprocess.cal_correction_factor(size, polarization, detd, pix_s, center)

	# reshape and write to file
	qx = q_coor[0].flatten()
	qy = q_coor[1].flatten()
	qz = q_coor[2].flatten()
	correction = correction.flatten()
	datamask = datamask.flatten()
	det_mapper = np.vstack([qx,qy,qz,datamask,correction]).T
	
	print("Radius of reciprocal space : %d pixels" % q_len)

	np.savetxt(det_q_output, det_mapper, fmt = "%.2f %.2f %.2f %d %.3f")
	f = open(det_q_output, 'a+')
	f.write("%d\n" % q_len)
	f.close()
	
