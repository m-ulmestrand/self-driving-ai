'''
If you are unsatisfied with the borders of a track, you can make changes, plot it and save it here.

Author: Mattias Ulmestrand
'''


from matplotlib import pyplot as plt
from draw_track import add_borders, remove_all_loops
import numpy as np


track_numbers = [1]

for i in track_numbers:
    name = f"racetrack{i}"
    track_nodes = np.load(f"tracks/{name}.npy")
    box_size = 100
    width = 5

    inner_line, outer_line = add_borders(track_nodes, width)

    for i in range(4, 12):
        remove_all_loops(inner_line, outer_line, i)

    fig, ax = plt.subplots()
    plt.scatter(track_nodes[:, 0], track_nodes[:, 1],
                color="black", s=10)

    plt.plot(outer_line[:, 0], outer_line[:, 1],
            color='black', linestyle='solid', marker='o', markersize='2')
    plt.plot(inner_line[:, 0], inner_line[:, 1],
            color='black', linestyle='solid', marker='o', markersize='2')

    plt.xlim(0, box_size)
    plt.ylim(0, box_size)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    save_track = input("Save track? ")

    if len(save_track) > 0:
        if save_track.lower() != "no" and save_track.lower() != "n":
            track = f'tracks/{name}'
            track_name = f'{track}.npy'

            np.save(f"{track}_inner_bound.npy", inner_line)
            np.save(f"{track}_outer_bound.npy", outer_line)
            print("Track saved as", track_name + ".")
        else:
            print("Track not saved.")
    else:
        print("Track not saved.")
