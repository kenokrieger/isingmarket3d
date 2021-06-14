#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt


for lid in range(256):
        file = f".data/lid={lid}.dat"
        agents = np.loadtxt(file)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(agents)
        plt.savefig(f".images/lattices/lid = {lid}.png")
        plt.close(fig)
