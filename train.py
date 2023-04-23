'''
This script allows you to train an agent on a set of tracks.
Provide the tracks you want to train on by feeding the list training_track_numbers to the agent.
If you want to train on tracks 0, 3, 4, 5 and 8, construct the list [0, 3, 4, 5, 8].
Other parameters can be changed as well, although the base settings should work fine.

Author: Mattias Ulmestrand
'''


from racing_agent import RacingAgent
from racing_network import *
import numpy as np
import sys


def main():
    box_size = 100
    runs = 1500
    epsilon_scale = 3000
    training_track_numbers = [8]
    n_epochs = 1
    race_car = RacingAgent(box_size=box_size, epsilon_scale=epsilon_scale, buffer_behaviour="discard_old", turning_speed=0.125,
                        epsilon_start=0.5, epsilon_final=0., r_min=5., buffer_size=5000, seq_length=1, network_type=DenseNetwork,
                        hidden_neurons=(32,32,32), target_sync=100, generation_length=1000, track_numbers=training_track_numbers,
                        turn_radius_decay=1.5, append_scale=20)

    # Change this to initialize and train a new agent.
    # Trained agents are saved at ./build, load just the name without the .pt extension.
    # Both the final agent and the best performing one are saved.
    race_car.save_name = 'agent_attn'
    race_car.load_network(name=race_car.save_name)
    race_car.reinitialise_random_track()

    message = ""
    for i in range(runs):
        while not race_car.has_collided and race_car.current_step < race_car.generation_length:
            race_car.choose_action()
            race_car.reward_per_node()

        race_car.reinforce(n_epochs)
        sys.stdout.write("\b" * len(message))
        message = f"Generation: {i}, " \
                    f"Time before crash: {race_car.current_step}, " \
                    f"Number of passed nodes: {len(race_car.node_passing_times)}, " \
                    f"Distance: {round(race_car.distance, 2)}, " \
                    f"Epsilon: {round(race_car.get_epsilon(), 2)}, " \
                    f"Loss: {round(race_car.total_loss, 4)}"
        sys.stdout.write(message)
        race_car.reinitialise_random_track()
    race_car.save_network('final_' + race_car.save_name)


if __name__ == "__main__":
    main()
