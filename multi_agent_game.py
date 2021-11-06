'''
This experimental script allows you to test several trained agents on a track of your choice.
Just change the racetrack name to the one you want to try out, and change the agent name to the one you want to use.

Author: Mattias Ulmestrand
'''

from racing_agent import RacingAgent
from racing_network import DenseNetwork, RecurrentNetwork
import numpy as np
from matplotlib import pyplot as plt
from collision_handling import get_lidar_lines
from copy import deepcopy


box_size = 100
race_car = RacingAgent(box_size=box_size, buffer_size=1)

# Change track to the one you want to try out.
# Tracks are saved at ./tracks
track = "racetrack12"
race_car.store_track(track)

# Change race_car.save_name use an agent of your choice.
# Trained agents are saved at ./build, load just the name without the .pt extension.
# Both the final agent and the best performing one are saved.
race_car.save_name = 'agent_dense'
race_car.load_network(name=race_car.save_name)

race_cars = [race_car]
n_cars = 5

# Puts cars two nodes apart from each other.
for i in range(1, n_cars):
    new_car = deepcopy(race_car)
    node_id = i*2
    new_car.position = np.copy(new_car.track_nodes[node_id])
    diff = new_car.track_nodes[node_id] - new_car.track_nodes[node_id-1]
    new_car.angle = np.arctan2(diff[1], diff[0])
    race_cars.append(new_car)


def main():
    fig, ax = plt.subplots()
    inner_line = np.load(f'tracks/{track}_inner_bound.npy')
    outer_line = np.load(f'tracks/{track}_outer_bound.npy')
    nodes = np.load(f'tracks/{track}.npy')
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal', adjustable='box')
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

        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal', adjustable='box')
        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == '__main__':
    main()
