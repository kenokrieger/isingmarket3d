#! /usr/bin/env python3
"""Plots a live view of the magnetisation"""
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig, ax = plt.subplots()

    magnetisation = np.loadtxt("magnetisation.dat")
    ax.plot(magnetisation)
    plt.title("Relative Magnetisation")
    plt.savefig("magnetisation.png")
