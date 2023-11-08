'''
This script allows you to train an agent on a set of tracks.
Provide the tracks you want to train on by feeding the list training_track_numbers to the agent.
If you want to train on tracks 0, 3, 4, 5 and 8, construct the list [0, 3, 4, 5, 8].
Other parameters can be changed as well, although the base settings should work fine.

Author: Mattias Ulmestrand
'''


from matplotlib import pyplot as plt
from racing_agent import RacingAgent
from racing_network import *
import numpy as np
import sys


def main():
    runs = 1500
    training_track_numbers = [15]
    n_epochs = 10
    race_car = RacingAgent(
        box_size=100, 
        buffer_behaviour="discard_old", 
        turning_speed=0.125,
        r_min=5.,  
        seq_length=1, 
        network_type=DenseNetwork,
        network_params=(32,32,32), 
        generation_length=1000, 
        c_entropy=0.02,
        c_value=0.75,
        policy_lr=5e-4,
        value_lr=1e-3,
        gradient_clip=10.,
        track_numbers=training_track_numbers,
        turn_radius_decay=1., 
        append_scale=20,
        name='agent_dense_policy'
    )

    # Change this to initialize and train a new agent.
    # Trained agents are saved at ./build, load just the name without the .pt extension.
    # Both the final agent and the best performing one are saved.
    race_car.load_network(name=race_car.save_name)
    race_car.store_track(training_track_numbers[0])

    avg_interval = 50
    y_max = 1000
    fig, ax = plt.subplots()
    ax.set_xlim([0, runs])
    ax.set_ylim([0, y_max])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Distance travelled")
    x_data = np.arange(runs)
    dists = np.zeros(runs)
    avg_dists = np.zeros(runs)
    best_dists = np.zeros(runs)
    plot, = ax.plot(dists, color="black", label="Distance")
    avg_plot, = ax.plot(avg_dists, color="red", linestyle="dashed", label="Running average")
    best_plot, = ax.plot(best_dists, color="dodgerblue", label="Maximum")
    ax.legend(loc="upper left")
    fig.canvas.draw()
    plt.show(block=False)

    message = ""
    for i in range(runs):
        track_number = int(np.random.choice(training_track_numbers))
        race_car.reinitialise_with_track(track_number, keep_progress=True)
        all_collided = (race_car.collision_times < race_car.generation_length - 1).all()
        while not all_collided and race_car.current_step < race_car.generation_length:
            race_car.choose_action()
            race_car.reward_per_node()

        race_car.reinforce(n_epochs)
        max_distance = race_car.distance.max()
        sys.stdout.write("\b" * len(message))
        message = f"Generation: {i}, " \
                    f"Distance: {max_distance:.2f}, " \
                    f"Policy expectation: {race_car.policy_expectation}, " \
                    f"Value loss: {race_car.value_loss:.4f}, " \
                    f"Entropy: {race_car.entropy:.4f}"
        sys.stdout.write(message)

        current_dist = max_distance
        if current_dist > y_max:
            y_max = current_dist
            ax.set_ylim([0, y_max * 1.1])

        if current_dist > best_dists[i - 1]:
            best_dists[i] = current_dist
        else:
            best_dists[i] = best_dists[i - 1]

        dists[i] = current_dist
        avg_dists[i] = dists[max(0, i - avg_interval): i + 1].mean()
        race_car.reinitialise()

        plot.set_data(x_data[:i + 1], dists[:i + 1])
        avg_plot.set_data(x_data[:i + 1], avg_dists[:i + 1])
        best_plot.set_data(x_data[:i + 1], best_dists[:i + 1])
        ax.draw_artist(best_plot)
        ax.draw_artist(plot)
        ax.draw_artist(avg_plot)
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.savefig("./figures/history.png", format="png")

    race_car.save_network('final_' + race_car.save_name)


if __name__ == "__main__":
    main()
