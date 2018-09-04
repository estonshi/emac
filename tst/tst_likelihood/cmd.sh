#!/bin/bash

set -e

../../bin/tst_likelihood -f vol.bin -d det_q.mpr -b 15 -l 125 -m pat_mask.bin -q orientations.quat -z 14553 -g 1 -p test_pat.emac
