#! /usr/bin/env python3
"""Plots a live view of the magnetisation"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


LATTICE_SIZE = 128 ** 3

if __name__ == "__main__":
    fig, ax = plt.subplots()
    plt.title("Relative Magnetisation")


    def animate(i):
        magnetisation = np.loadtxt("magnetisation.dat", max_rows = 1)
        idx = np.arange(0, len(magnetisation))
        plt.cla()
        ax.plot(idx, magnetisation / LATTICE_SIZE)

    ani = FuncAnimation(fig, animate, interval=500)
    plt.show()
