# emac

**emac** is a cuda program of [EMC methods](https://arxiv.org/pdf/0904.2581v1.pdf).

---

This code should run on a 64-bit **Linux** machine.

NVidia GPU is needed, with compute capability >= 3.0, CUDA version >= 8.0, and GPU global memory >= 4GB.

Patterns with size larger than (512,512) are not recomended.

This is a single-precision program. Keep the total photon number of a pattern < 2^23 to avoid accuracy loss.


---

1. Compile : run "compile.sh"

2. Create project : run "new_project.sh <folder>"

3. Run : All exec files are in "./bin" folder. Run "./bin/emac.xxx -h" for help.