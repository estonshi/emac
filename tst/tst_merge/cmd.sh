#!/bin/bash

set -e

../../bin/emac.tst.slice_merge -f vol.bin -d det_q.mpr -x 123 -y 123 -l 125 -m pat_mask.bin -q orientations.quat -z 14553 -g 1
