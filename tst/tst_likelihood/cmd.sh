#!/bin/bash

set -e

nvprof ../../bin/emac.tst.maximization -f vol.bin -d det_q.mpr -b 30 -l 125 -m pat_mask.bin -q orientations.quat -z 14553 -g 1 -p test_pat.emac
