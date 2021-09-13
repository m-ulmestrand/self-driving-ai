from racing_agent_improved import RacingAgent
import numpy as np
from matplotlib import pyplot as plt
from init_cuda import init_cuda
from collision_handling import get_lidar_lines
import sys


box_size = 100
runs = 5000
race_car = RacingAgent(box_size=box_size, epsilon_scale=runs, buffer_behaviour="discard_old",
                       epsilon_final=0.5, r_min=5., buffer_size=5000)
race_car.save_name = 'racing_agent_improved3'
race_car.load_network(name=race_car.save_name)

track = "racetrack0"
race_car.store_track(track)
original_pos = np.copy(race_car.position)
train_network = False

if train_network:
    message = ""
    for i in range(runs):
        while not race_car.has_collided and race_car.current_step < race_car.generation_length:
            race_car.choose_action()
            race_car.reward_per_node()

        race_car.reinforce(epochs=10)
        sys.stdout.write("\b" * len(message))
        message = f"Generation: {race_car.generation}, " \
                  f"Time before crash: {race_car.current_step}, " \
                  f"Number of passed nodes: {len(race_car.node_passing_times)}, " \
                  f"Distance: {round(np.sqrt(np.sum((race_car.position-original_pos)**2)), 2)}, " \
                  f"Epsilon: {round(race_car.get_epsilon(), 2)}, " \
                  f"Loss: {round(race_car.total_loss, 4)}"
        sys.stdout.write(message)
        race_car.reinitialize_random_track()
    race_car.save_network('final_' + race_car.save_name)

else:
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
    race_car.load_network('final_'+race_car.save_name)

    while plt.fignum_exists(fig.number) \
            and race_car.current_step < race_car.generation_length:
        race_car.choose_action(epsilon=0.)
        xs, ys, car_bounds, car_collides = get_lidar_lines(race_car)
        features = race_car.get_features()
        ax.cla()
        ax.plot(inner_line[:, 0], inner_line[:, 1], 'k')
        ax.plot(outer_line[:, 0], outer_line[:, 1], 'k')
        ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=2)
        ax.plot(car_bounds[:, 0], car_bounds[:, 1], 'r' if race_car.has_collided else 'k')
        ax.plot(xs, ys, color='limegreen')
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal', adjustable='box')
        fig.canvas.draw()
        fig.canvas.flush_events()