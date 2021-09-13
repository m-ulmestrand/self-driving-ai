from matplotlib import pyplot as plt
import numpy as np


name = "racetrack5"
track_nodes = np.load(f"{name}.npy")
outer_line = np.zeros_like(track_nodes)
inner_line = np.zeros_like(track_nodes)
width = 5


def add_borders(node1, node2, node3, i):
    x_diff1, y_diff1 = node2 - node1
    x_diff2, y_diff2 = node3 - node2
    angle1 = np.arctan2(y_diff1, x_diff1)
    angle2 = np.arctan2(y_diff2, x_diff2)
    distance = np.sqrt(x_diff1 ** 2 + y_diff1 ** 2)

    x_diff_norm = x_diff1 / distance
    y_diff_norm = y_diff1 / distance
    # Orthogonal to [x_diff, y_diff]
    width_vect = np.array([-y_diff_norm, x_diff_norm]) * width

    d_theta = angle2 - angle1
    surplus = width * np.sin(np.abs(d_theta))
    outer_line[i] = node1 + width_vect
    outer_line[i + 1] = np.array([x_diff_norm, y_diff_norm]) * (distance - surplus)
    inner_line[i] = node1 - width_vect
    inner_line[i + 1] = np.array([x_diff_norm, y_diff_norm]) * (distance + surplus)


new_nodes = np.append(track_nodes, np.array([track_nodes[1]]), axis=0)
for i, node in enumerate(zip(new_nodes[:-2], new_nodes[1:-1], new_nodes[2:])):
    add_borders(*node, i)
outer_line[-1] = outer_line[0]
inner_line[-1] = inner_line[0]


fig, ax = plt.subplots()
plt.plot(track_nodes[:, 0], track_nodes[:, 1],
         color='black', linestyle='', marker='o', markersize='3')

plt.plot(outer_line[:, 0], outer_line[:, 1],
         color='black', linestyle='solid')
plt.plot(inner_line[:, 0], inner_line[:, 1],
         color='black', linestyle='solid')
box_size = 100
plt.xlim(0, box_size)
plt.ylim(0, box_size)
ax.set_aspect('equal', adjustable='box')
plt.show()
np.save(f"{name}_inner_bound.npy", inner_line)
np.save(f"{name}_outer_bound.npy", outer_line)