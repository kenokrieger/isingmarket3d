#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

agents = np.empty((128, 128, 128))
for lid in range(128):
        file = f".data/lid={lid}.dat"
        agents[:, :, lid] = np.loadtxt(file)

agents += 1
agents /= 2

filled = np.array([
    [[1, 0, 1]]
])

for slice in range(16, 128, 16):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(agents[slice - 16:slice, slice - 16:slice, slice - 16:slice], facecolors="grey", shade=False, alpha=0.3)
    ax.view_init(8, 64)
    plt.savefig(f".images/lattice3d_{slice}.png", dpi=300)
    plt.close(fig)
