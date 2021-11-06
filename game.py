'''
This script allows you to test a trained agent out on a track of your choice.
Just change the racetrack name to the one you want to try out.

Author: Mattias Ulmestrand
'''


from racing_agent import RacingAgent
from racing_network import DenseNetwork, RecurrentNetwork
import numpy as np
from matplotlib import pyplot as plt
from collision_handling import get_lidar_lines
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run a simulation with a trained agent."
    )
    parser.add_argument(
        "--agent-name",
        required=False,
        help="Name of the pretrained agent."
    )
    parser.add_argument(
        "--track-name",
        required=False,
        help="Name of the track."
    )

    args = parser.parse_args()
    box_size = 100
    race_car = RacingAgent(box_size=box_size, buffer_size=1)

    # Change race_car.save_name use an agent of your choice.
    # Trained agents are saved at ./build, load just the name without the .pt extension.
    # Both the final agent and the best performing one are saved.
    race_car.save_name = 'agent_dense'
    race_car.load_network(name=race_car.save_name)

    # Change track to the one you want to try out.
    # Tracks are saved at ./tracks
    track = "racetrack12"

    # If you have specified racetrack or agent in command line, these will be used instead.
    if args.track_name is not None:
        track = args.track_name
    
    if args.agent_name is not None:
        race_car.save_name = args.agent_name

    race_car.store_track(track)
    original_pos = np.copy(race_car.position)

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
        race_car.choose_action(epsilon=0.)
        xs, ys, car_bounds, car_collides = get_lidar_lines(race_car)
        ax.cla()
        ax.plot(inner_line[:, 0], inner_line[:, 1], 'k')
        ax.plot(outer_line[:, 0], outer_line[:, 1], 'k')
        ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=2)
        ax.plot(car_bounds[:, 0], car_bounds[:, 1], 'k')
        ax.plot(xs, ys, color='limegreen')
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal', adjustable='box')
        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == "__main__":
    main()
