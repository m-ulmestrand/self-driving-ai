'''
If you are unsatisfied with the borders of a track, you can make changes, plot it and save it here.

Author: Mattias Ulmestrand
'''


from matplotlib import pyplot as plt
import numpy as np
import os.path


name = "racetrack12"
track_nodes = np.load(f"tracks/{name}.npy")
outer_line = np.zeros_like(track_nodes)
inner_line = np.zeros_like(track_nodes)
box_size = 100
width = 5

def arctan_2pi(y, x):
    result = np.arctan2(y, x)
    return result + np.pi

def add_borders(node1, node2, node3, i, d_theta_prev):
    x_diff1, y_diff1 = node2 - node1
    x_diff2, y_diff2 = node3 - node2
    angle1 = arctan_2pi(y_diff1, x_diff1)
    angle2 = arctan_2pi(y_diff2, x_diff2)
    d_theta = angle2 - angle1
    diff_d_theta = (d_theta_prev - d_theta)
    if abs(diff_d_theta - 2*np.pi) < abs(diff_d_theta):
        d_theta += 2*np.pi
    elif abs(diff_d_theta + 2*np.pi) < abs(diff_d_theta):
        d_theta -= 2*np.pi
    d_theta_prev = d_theta
    half_angle = np.abs(d_theta) / 2
    distance = np.sqrt(x_diff1 ** 2 + y_diff1 ** 2)

    # The distance is the chord length of a circle segment with angle d_theta
    # Add a small number so that the expression doesn't blow up when d_theta = 0
    radius = distance / (2 * np.sin(half_angle) + 0.001)
    inner_dist = max(2 * (radius - width * np.sign(d_theta)) * np.sin(half_angle), 0)
    outer_dist = max(2 * (radius + width * np.sign(d_theta)) * np.sin(half_angle), 0)

    x_direction = x_diff1 / distance
    y_direction = y_diff1 / distance
    
    # Orthogonal to [x_diff, y_diff]
    width_vect = np.array([-y_direction, x_direction]) * width

    surplus = width * np.sin(d_theta)
    if np.abs(surplus) < distance:
        surplus = np.sign(surplus) * distance
    
    outer_line[i] = node1 - width_vect
    inner_line[i] = node1 + width_vect
    outer_line[i + 1] = outer_line[i] + np.array([x_direction, y_direction]) * outer_dist
    inner_line[i + 1] = inner_line[i] + np.array([x_direction, y_direction]) * inner_dist


new_nodes = np.append(track_nodes, np.array([track_nodes[1]]), axis=0)
angle = 0
for i, node in enumerate(zip(new_nodes[:-2], new_nodes[1:-1], new_nodes[2:])):
    if not i % 2:
        add_borders(*node, i, angle)
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
        track = 'tracks/{name}'
        track_name = f'{track}.npy'

        np.save(f"{track}_inner_bound.npy", inner_line)
        np.save(f"{track}_outer_bound.npy", outer_line)
        print("Track saved as", track_name + ".")
    else:
        print("Track not saved.")
else:
    print("Track not saved.")
