#!/usr/bin/env python

import numpy as np
import argparse
import ConfigParser
import os
import sys
from spipy.analyse import q
from spipy.image import radp

if __name__ == '__main__':
	
	# parse cmd arguments
	parser = argparse.ArgumentParser(description = \
		"Mappping detector pixels onto Ewald Sphere.")
	parser.add_argument("-c", "--config", type=str, help="path of config.ini")
	args = parser.parse_args()

	# read config
	cf = ConfigParser.ConfigParser()
	cf.read(args.config)
	detd = cf.getfloat("input", "detd")
	pix_s = cf.getfloat("input", "pix_s")
	lamb = cf.getfloat("input", "lambda")
	size = np.array(cf.get("input", "size").split(',')).astype(int)
	mask = cf.get("input", "mask")
	ron = cf.getint("adjust", "ron")
	downsampl = cf.getfloat("adjust", "downsampl")
	center = cf.get("input", "center")
	det_q_output = cf.get("output", "det_q")

	if center != "None":
		center = np.array(center.split(',')).astype(float)
	else:
		center = size/2.0
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
	x, y = np.indices(size)
	x = x - center[0]
	y = y - center[0]
	z = np.zeros(x.shape) + detd
	r = np.linalg.norm([x, y], axis=0)

	# r_vec = np.array([x, y, z])
	# map r_vec to ewald sphere with r=detd and get qn_vec, assume the sample is at origin
	qx = np.zeros(size)
	qy = np.zeros(size)
	qz = np.zeros(size)

	nonzero = np.where(x != 0)
	a = y[nonzero]/x[nonzero]
	b = z[nonzero]/x[nonzero]
	qx[nonzero] = 1/np.sqrt(1+a**2+b**2) * np.sign(x[nonzero])
	qy[nonzero] = a * qx[nonzero]
	qz[nonzero] = b * qx[nonzero] - 1

	nonzero = np.where( (y != 0) & (x == 0) )
	a = z[nonzero]/y[nonzero]
	qy[nonzero] = 1/np.sqrt(1+a**2) * np.sign(y[nonzero])
	qz[nonzero] = a * qy[nonzero] - 1

	q_norm = np.linalg.norm([qx, qy, qz], axis=0)
	q_norm[np.where(q_norm == 0)] = 1e-10
	qx = qx * r / q_norm / downsampl
	qy = qy * r / q_norm / downsampl
	qz = qz * r / q_norm / downsampl

	# reshape and write to file
	qx = qx.flatten()
	qy = qy.flatten()
	qz = qz.flatten()
	datamask = datamask.flatten()
	det_mapper = np.vstack([qx,qy,qz,datamask]).T
	q_norm = np.int(np.linalg.norm([qx, qy, qz], axis=0).max() + 2)

	print("Radius of reciprocal space : %d" % q_norm)

	np.savetxt(det_q_output, det_mapper, fmt = "%.2f %.2f %.2f %d")
	f = open(det_q_output, 'a')
	f.write("%d" % q_norm)
	f.close()
	