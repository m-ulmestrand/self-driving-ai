from matplotlib import pyplot as plt
import keyboard
import numpy as np
import os.path


box_size = 100              # Maximum bounds of track
mouse_x, mouse_y = 0, 0     # Mouse positions
nodes = np.zeros((0, 2))    # Positions of track nodes
d = 3                       # Distance between nodes

fig, ax = plt.subplots()
plt.plot()
plt.xlim(0, box_size)
plt.ylim(0, box_size)
ax.set_aspect('equal', adjustable='box')
fig.canvas.draw()
plt.show(block=False)


def keyboard_event():
    return True if keyboard.is_pressed('a') else False


def mouse_move(event):
    global mouse_x, mouse_y
    mouse_x, mouse_y = event.xdata, event.ydata


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


node_nr = 0
while plt.fignum_exists(fig.number):
    button_pressed = keyboard_event()
    if button_pressed:
        plt.connect('motion_notify_event', mouse_move)
        if mouse_x != 0.:
            if nodes.size == 0:
                nodes = np.append(nodes, np.array([[mouse_x, mouse_y]]), axis=0)
            else:
                x_diff = mouse_x - nodes[node_nr, 0]
                y_diff = mouse_y - nodes[node_nr, 1]
                d_squared = x_diff ** 2 + y_diff ** 2
                if d_squared >= d ** 2:
                    distance = np.sqrt(d_squared)
                    new_x, new_y = nodes[node_nr, 0] + x_diff/distance*d, nodes[node_nr, 1] + y_diff/distance*d
                    node_nr += 1
                    nodes = np.append(nodes, np.array([[new_x, new_y]]), axis=0)
                    plt.plot(nodes[:, 0], nodes[:, 1],
                             color='black', linestyle='solid', marker='o', markersize='3')

                x_diff = mouse_x - nodes[0, 0]
                y_diff = mouse_y - nodes[0, 1]
                d_squared = x_diff ** 2 + y_diff ** 2
                if d_squared <= 2 * d ** 2 and np.sum((nodes[-1] - nodes[0])**2) <= 2 * d ** 2 and nodes.size > 5:
                    nodes = np.append(nodes, np.array([[mouse_x, mouse_y]]), axis=0)
                    nodes = np.append(nodes, np.array([[nodes[0, 0], nodes[0, 1]]]), axis=0)
                    break

    fig.canvas.draw()
    fig.canvas.flush_events()

outer_line = np.zeros_like(nodes)
inner_line = np.zeros_like(nodes)
width = 5

new_nodes = np.append(nodes, np.array([nodes[1]]), axis=0)
for i, node in enumerate(zip(new_nodes[:-2], new_nodes[1:-1], new_nodes[2:])):
    add_borders(*node, i)
outer_line[-1] = outer_line[0]
inner_line[-1] = inner_line[0]


plt.plot(nodes[:, 0], nodes[:, 1],
         color='black', linestyle='solid', marker='o', markersize='3')

plt.plot(outer_line[:, 0], outer_line[:, 1],
         color='black', linestyle='solid')
plt.plot(inner_line[:, 0], inner_line[:, 1],
         color='black', linestyle='solid')

plt.show()
save_track = input("Save track? ")

if len(save_track) > 0:
    if save_track.lower() != "no" and save_track.lower() != "n":
        i = 0
        track = 'build/racetrack'
        track_name = f"{track}{i}.npy"
        while os.path.isfile(track_name):
            i += 1
            print("Name", track_name, "already taken.")
            track_name = f"{track}{i}.npy"
        np.save(track_name, nodes)
        np.save(f"{track}{i}_inner_bound.npy", inner_line)
        np.save(f"{track}{i}_outer_bound.npy", outer_line)
        print("Track saved as", track_name + ".")
    else:
        print("Track not saved.")
else:
    print("Track not saved.")
