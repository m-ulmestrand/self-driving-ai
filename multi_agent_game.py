from racing_agent import RacingAgent
from racing_network import DenseNetwork, RecurrentNetwork
import numpy as np
from matplotlib import pyplot as plt
from init_cuda import init_cuda
from collision_handling import get_lidar_lines
import sys
from copy import deepcopy


box_size = 100
runs = 1500
epsilon_scale = 1000
race_car = RacingAgent(box_size=box_size, epsilon_scale=epsilon_scale, buffer_behaviour="discard_old",
                       epsilon_start=1.0, epsilon_final=0.1, r_min=5., buffer_size=5000, seq_length=1, network_type=DenseNetwork,
                       hidden_neurons=(32,32,32), target_sync=0.1, generation_length=1000)
track = "racetrack12"
race_car.store_track(track)
race_car.save_name = 'agent_dense3'
race_car.load_network(name=race_car.save_name)

race_cars = [race_car]
n_cars = 5

for i in range(1, n_cars):
    new_car = deepcopy(race_car)
    node_id = i*2
    new_car.position = np.copy(new_car.track_nodes[node_id])
    diff = new_car.track_nodes[node_id] - new_car.track_nodes[node_id-1]
    new_car.angle = np.arctan2(diff[1], diff[0])
    race_cars.append(new_car)

device = init_cuda()
fig, ax = plt.subplots()
inner_line = np.load(f'tracks/{track}_inner_bound.npy')
outer_line = np.load(f'tracks/{track}_outer_bound.npy')
nodes = np.load(f'tracks/{track}.npy')
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect('equal', adjustable='box')
line_plot, = ax.plot([None], [None])
fig.canvas.draw()
plt.show(block=False)
race_car.load_network(race_car.save_name)

while plt.fignum_exists(fig.number) \
        and race_car.current_step < race_car.generation_length:
    race_car.multi_agent_forward_pass(race_cars)
    ax.cla()
    ax.plot(inner_line[:, 0], inner_line[:, 1], 'k')
    ax.plot(outer_line[:, 0], outer_line[:, 1], 'k')
    ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=2)

    for car in race_cars:
        xs, ys, car_bounds, car_collides = get_lidar_lines(car)
        ax.plot(car_bounds[:, 0], car_bounds[:, 1], 'k')
        # ax.plot(xs, ys, color='limegreen')

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal', adjustable='box')
    fig.canvas.draw()
    fig.canvas.flush_events()