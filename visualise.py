#! /usr/bin/python3
from os.path import join

from matplotlib.pyplot import subplots, savefig, close
from numpy import loadtxt

saves_directory = ".images/lattices/"


def save_file(file, saves_directory):

    headers = []
    with open(file, 'r') as f:
        header = f.readline()
        while header[0] == '#':
            headers.append(header.replace('#', '').replace('\n', ''))
            header = f.readline()
    img_name = ", ".join(headers)

    agents = loadtxt(file)
    img_path = join(saves_directory, img_name + ".png")
    print("Plotting {}".format(img_name))

    fig, ax = subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(img_name)
    ax.imshow(agents, cmap="copper")
    savefig(img_path)
    close(fig)


if __name__ == "__main__":
    for lid in range(128):
        file = f".data/lid={lid}.dat"
        save_file(file, saves_directory)
