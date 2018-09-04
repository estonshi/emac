#!/bin/bash

set -e



cp ./py_src/map_det.py ./bin/map_det
chmod u+x ./bin/map_det

cp ./py_src/make_emac_data.py ./bin/make_data
chmod u+x ./bin/make_data

cp ./py_src/read_emac_data.py ./bin/read_data
chmod u+x ./bin/read_data

gcc ./src/gen_quat.c ./src/gen_quat_main.c -o ./bin/gen_quat -lm
gcc ./src/emac_data.c ./src/tst_emac_data.c -o ./bin/tst_emac_data -lm
gcc ./src/params.c ./src/gen_quat.c ./src/emac_data.c -o ./bin/params -lm