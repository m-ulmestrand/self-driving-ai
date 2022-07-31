'''
If you are unsatisfied with the borders of a track, you can make changes, plot it and save it here.

Author: Mattias Ulmestrand
'''


from matplotlib import pyplot as plt
from draw_track import add_borders
import numpy as np
import os.path


name = "racetrack15"
track_nodes = np.load(f"tracks/{name}.npy")
outer_line = np.zeros_like(track_nodes)
inner_line = np.zeros_like(track_nodes)
box_size = 100
width = 5


new_nodes = np.append(track_nodes, np.array([track_nodes[1]]), axis=0)
angle = 0
for i, node in enumerate(zip(new_nodes[:-2], new_nodes[1:-1], new_nodes[2:])):
    if not i % 2:
        add_borders(*node, inner_line, outer_line, i, angle, width)
outer_line[-1] = outer_line[0]
inner_line[-1] = inner_line[0]


fig, ax = plt.subplots()
plt.plot(track_nodes[:, 0], track_nodes[:, 1],
         color='black', linestyle='', marker='o', markersize='3')

plt.plot(outer_line[:, 0], outer_line[:, 1],
         color='black', linestyle='solid')
plt.plot(inner_line[:, 0], inner_line[:, 1],
         color='black', linestyle='solid')

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
