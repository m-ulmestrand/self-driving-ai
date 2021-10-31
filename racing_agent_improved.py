import torch
from torch import nn
import numpy as np
from typing import Literal
from racing_network import DenseNetwork
from collision_handling import *
from init_cuda import init_cuda
import os.path


device = init_cuda()


class RacingAgent:
    def __init__(self, box_size: int = 100, car_width: float = 1., car_length: float = 4., lane_width: float = 5.,
                 r_min: float = 4., turning_speed: float = 0.25, speed: float = 1., epsilon_start: float = 1.,
                 epsilon_scale: int = 2000, epsilon_final: float = 0.01, buffer_size: int = 5000, n_tracks: int = 7,
                 learning_rate: float = 0.001, batch_size: int = 100, network_type: nn.Module = DenseNetwork,
                 buffer_behaviour: Literal["until_full", "discard_old"] = "discard_old",
                 hidden_neurons: tuple = (32, 32), seq_length: int = 5, generation_length: int = 2000):

        # Various car model parameters
        self.box_size = box_size
        self.diag = box_size * np.sqrt(2)
        self.car_width, self.car_length = car_width, car_length
        self.lane_width = lane_width
        self.r_min = r_min
        self.max_speed = speed
        self.velocity = np.array([0., 0.])
        self.n_actions = 4
        self.n_inputs = 7
        self.angle = 0
        self.position = np.array([0., 0.])
        self.car_bounds = np.zeros((8, 2))
        self.angles = np.array([-0.4, -0.2, 0.2, 0.4]) * np.pi
        self.current_node = 0
        self.max_node = 0
        self.turning_angle = 0
        self.turning_speed = turning_speed
        self.save_name = "racing_agent_improved"
        self.n_tracks = n_tracks
        self.max_angle = np.pi / 4

        # Controlling exploration during Q-learning
        self.epsilon_start = epsilon_start
        self.epsilon_scale = epsilon_scale
        self.epsilon_final = epsilon_final

        # Placeholder for the track
        self.track_nodes = np.array([])
        self.track_outer = np.array([])
        self.track_inner = np.array([])

        # A few variables for checking progress on track
        self.has_collided = False
        self.node_passing_times = np.zeros(0, dtype='intc')
        self.passed_node = False

        # Neural network parameters and training buffers
        self.buffer_size = buffer_size
        self.buffer_behaviour = buffer_behaviour
        self.learning_rate = learning_rate
        self.generation = 0
        self.generation_length = generation_length
        self.current_step = 0
        self.network_params = [self.n_inputs, *hidden_neurons, self.n_actions]
        self.network_type = network_type
        self.network = network_type(self.network_params)
        self.seq_length = seq_length

        # Learning parameters and various Q-learning parameters
        self.rewards = torch.zeros(generation_length, dtype=torch.double)
        self.rewards_buffer = torch.zeros(0, dtype=torch.double)
        self.states = torch.zeros((generation_length, seq_length, self.n_inputs), dtype=torch.double)
        self.states_buffer = torch.zeros((0, seq_length, self.n_inputs), dtype=torch.double)
        self.old_states = torch.zeros((generation_length, seq_length, self.n_inputs), dtype=torch.double)
        self.old_states_buffer = torch.zeros((0, seq_length, self.n_inputs), dtype=torch.double)
        self.actions = torch.zeros(generation_length, dtype=torch.long)
        self.actions_buffer = torch.zeros(0, dtype=torch.long)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss(reduction="mean").to(device)
        self.batch_size = batch_size
        self.total_loss = 0
        self.longest_survival = 0

    def reinitialize(self):
        self.generation += 1
        self.current_step = 0
        self.position = np.copy(self.track_nodes[0])
        self.turning_angle = 0
        diff = self.track_nodes[1] - self.track_nodes[0]
        self.angle = np.arctan2(diff[1], diff[0])
        self.velocity = diff / np.sqrt(np.sum(diff**2)) * self.max_speed * 0.5
        self.rewards = torch.zeros(self.generation_length, dtype=torch.double)
        self.states = torch.zeros((self.generation_length, self.seq_length, self.n_inputs), dtype=torch.double)
        self.old_states = torch.zeros((self.generation_length, self.seq_length, self.n_inputs), dtype=torch.double)
        self.actions = torch.zeros(self.generation_length, dtype=torch.long)
        self.has_collided = False
        self.node_passing_times = np.zeros(0, dtype='intc')
        self.passed_node = False
        self.current_node = 0
        self.total_loss = 0

    def reinitialize_random_track(self):
        self.reinitialize()
        self.store_random_track()

    def load_network(self, name=None):
        if name is None:
            name = self.save_name
        name = 'build/' + name + '.pt'
        if os.path.isfile(name):
            self.network = self.network_type(self.network_params).to(device)
            self.network.load_state_dict(torch.load(name))
        else:
            print("PyTorch checkpoint does not exist. Skipped loading.")

    def save_network(self, name=None):
        if name is None:
            name = self.save_name
        name = 'build/' + name + '.pt'
        torch.save(self.network.state_dict(), name)

    def store_track(self, track_name):
        npy = '.npy'
        track_name = 'tracks/' + track_name
        self.track_nodes = np.load(track_name + npy)
        self.track_inner = np.load(track_name + '_inner_bound' + npy)
        self.track_outer = np.load(track_name + '_outer_bound' + npy)
        self.position = np.copy(self.track_nodes[0])
        diff = self.track_nodes[1] - self.track_nodes[0]
        self.velocity = diff / np.sqrt(np.sum(diff ** 2)) * self.max_speed * 0.5
        self.angle = np.arctan2(diff[1], diff[0])
        self.car_bounds = car_lines(self.position, self.angle, self.car_width, self.car_length)

    def store_random_track(self, choices=None):
        if choices is None:
            name = 'racetrack' + str(np.random.randint(0, self.n_tracks))
        else:
            name = 'racetrack' + str(np.random.choice(choices))
        self.store_track(track_name=name)

    def get_epsilon(self):
        return max(self.epsilon_start - self.generation / self.epsilon_scale, self.epsilon_final)

    def move(self):
        speed = np.sqrt(np.sum(self.velocity ** 2))
        angular_vel = speed / (self.r_min * np.tan(np.pi/2 - self.turning_angle))
        self.angle += angular_vel
        self.velocity[0] = np.cos(self.angle) * speed
        self.velocity[1] = np.sin(self.angle) * speed
        self.position += self.velocity

    def get_features(self):
        position = self.position
        nodes = self.track_nodes
        track_outer = self.track_outer
        track_inner = self.track_inner

        node_index = get_node(nodes[:-1], position)

        if node_index > self.current_node or (len(self.node_passing_times) > 0 and
                                              (node_index == 0 and self.current_node == self.track_nodes.shape[0] - 1)):
            self.current_node = node_index
            self.node_passing_times = np.append(self.node_passing_times, self.current_step)
            self.passed_node = True

        dist_to_node = np.sqrt(np.sum((position - nodes[node_index]) ** 2))
        vector = np.array([np.cos(self.angle), np.sin(self.angle)])
        vector /= np.sqrt(np.sum(vector ** 2))
        car_front = position + vector * self.car_length
        vector *= self.diag
        new_vectors = rotate(self.angles, vector.reshape((2, 1)))
        new_vectors = np.append(new_vectors, vector.reshape((1, 2)), axis=0)
        car_bounds = car_lines(position, self.angle, self.car_width, self.car_length)
        features = torch.ones((1, self.seq_length, self.n_inputs), dtype=torch.double)

        for i, vec in enumerate(new_vectors):

            for node_id in np.append(np.arange(node_index - 1, nodes.shape[0] - 1), np.arange(0, node_index - 1)):
                distance_frac = line_intersect_distance(car_front, car_front + vec,
                                                        track_outer[node_id], track_outer[node_id + 1])
                if distance_frac != 0:
                    features[0, -1, i] = distance_frac
                    break
                else:
                    distance_frac = line_intersect_distance(car_front, car_front + vec,
                                                            track_inner[node_id], track_inner[node_id + 1])
                    if distance_frac != 0:
                        features[0, -1, i] = distance_frac
                        break

        car_collides = True if dist_to_node > self.lane_width else False
        if not car_collides:
            for node_id in (node_index, node_index+1):
                for i in np.arange(3):
                    if line_intersect_distance(car_bounds[2 * i], car_bounds[2 * i + 1], track_outer[node_id],
                                               track_outer[node_id + 1]):
                        car_collides = True
                        break
                    elif line_intersect_distance(car_bounds[2 * i], car_bounds[2 * i + 1], track_inner[node_id],
                                                 track_inner[node_id + 1]):
                        car_collides = True
                        break

        features[0, -1, -2] = self.turning_angle / self.max_angle
        features[0, -1, -1] = np.sqrt(np.sum(self.velocity ** 2)) / self.max_speed
        
        if self.seq_length > 1:
            if self.current_step != 0:
                features[0, :-1] = self.old_states[self.current_step, 1:]
            else:
                for i in range(self.seq_length-1):
                    features[0, i] = features[0, -1]

        self.has_collided = car_collides
        return features

    def turn_left(self):
        max_angle = self.max_angle
        self.turning_angle = max_angle if self.turning_angle > max_angle else self.turning_angle + self.turning_speed

    def turn_right(self):
        min_angle = -self.max_angle
        self.turning_angle = min_angle if self.turning_angle < min_angle else self.turning_angle - self.turning_speed

    def take_action(self, action):
        current_speed = np.sqrt(np.sum(self.velocity ** 2))
        if action == 0:
            self.turn_left()
        elif action == 1:
            self.turn_right()
        elif action == 2 and current_speed < self.max_speed:
            # Accelerate
            self.velocity[0] += np.cos(self.angle) * 0.1
            self.velocity[1] += np.sin(self.angle) * 0.1
            new_speed = np.sqrt(np.sum(self.velocity ** 2))
            if new_speed > self.max_speed:
                self.velocity *= (self.max_speed / new_speed)
        else:
            # Decelerate
            if current_speed > 0.5 * self.max_speed:
                self.velocity = self.velocity * 0.9

        self.move()
        self.states[self.current_step] = self.get_features()[0]

    def choose_action(self, epsilon=None):
        # Exploration
        eps = self.get_epsilon() if epsilon is None else epsilon
        if np.random.rand() < eps:
            action = np.random.randint(0, self.n_actions)
            if self.current_step == 0:
                self.old_states[self.current_step] = self.get_features()[0]
            else:
                self.old_states[self.current_step] = self.states[self.current_step-1].reshape((self.seq_length, self.n_inputs))

        # Exploitation
        else:
            self.network.eval()

            if self.current_step == 0:
                features = self.get_features()
            else:
                features = self.states[self.current_step-1].reshape((1, self.seq_length, self.n_inputs))
            self.old_states[self.current_step] = features[0]
            output = self.network(features.to(device).double())
            action = torch.argmax(output).detach().cpu().item()

        self.actions[self.current_step] = action
        self.take_action(action)

    def append_tensors(self, rewards, actions, old_states, states):
        buffer_current_size = self.rewards_buffer.shape[0]
        new_size = rewards.shape[0]
        surplus = buffer_current_size + new_size - self.buffer_size

        if self.buffer_behaviour == "discard_old":
            if surplus > 0:
                self.rewards_buffer[0: buffer_current_size - surplus] = self.rewards_buffer[surplus: buffer_current_size].clone()
                self.rewards_buffer[-surplus:] = rewards[:surplus]
                self.rewards_buffer = torch.cat((self.rewards_buffer, rewards[surplus:]), dim=0)

                self.actions_buffer[0: buffer_current_size - surplus] = self.actions_buffer[surplus: buffer_current_size].clone()
                self.actions_buffer[-surplus:] = actions[:surplus]
                self.actions_buffer = torch.cat((self.actions_buffer, actions[surplus:]), dim=0)

                self.old_states_buffer[0: buffer_current_size - surplus] = \
                    self.old_states_buffer[surplus: buffer_current_size].clone()
                self.old_states_buffer[-surplus:] = old_states[:surplus]
                self.old_states_buffer = torch.cat((self.old_states_buffer, old_states[surplus:]), dim=0)

                self.states_buffer[0: buffer_current_size - surplus] = self.states_buffer[surplus: buffer_current_size].clone()
                self.states_buffer[-surplus:] = states[:surplus]
                self.states_buffer = torch.cat((self.states_buffer, states[surplus:]), dim=0)

            else:
                self.rewards_buffer = torch.cat((self.rewards_buffer, rewards), dim=0)
                self.actions_buffer = torch.cat((self.actions_buffer, actions), dim=0)
                self.old_states_buffer = torch.cat((self.old_states_buffer, old_states), dim=0)
                self.states_buffer = torch.cat((self.states_buffer, states), dim=0)

        elif self.buffer_behaviour == "until_full":
            if buffer_current_size < self.buffer_size:
                n_to_append = self.buffer_size - buffer_current_size
                self.rewards_buffer = torch.cat((self.rewards_buffer, rewards[:n_to_append]), dim=0)
                self.actions_buffer = torch.cat((self.actions_buffer, actions[:n_to_append]), dim=0)
                self.old_states_buffer = torch.cat((self.old_states_buffer, old_states[:n_to_append]), dim=0)
                self.states_buffer = torch.cat((self.states_buffer, states[:n_to_append]), dim=0)

    def reward_per_node(self):
        self.current_step += 1
        if self.has_collided or self.current_step == self.generation_length:
            if len(self.node_passing_times) == 1:
                n_steps = self.node_passing_times[0] + 1
                reward = 1 / n_steps
                for i in range(n_steps):
                    self.rewards[i] = reward * (i + 1) / self.current_step
                # Save the times after passing nodes, they will be penalised
                t_after_nodes = self.node_passing_times[-1] + 1

            elif len(self.node_passing_times) > 1:
                previous_times = np.append(0, self.node_passing_times[:-1])
                for time1, time in zip(previous_times, self.node_passing_times):
                    time2 = time + 1
                    diff = time2 - time1
                    reward = 1 / diff
                    for step, i in enumerate(range(time1, time2)):
                        self.rewards[i] = reward * (step + 1) / diff
                t_after_nodes = self.node_passing_times[-1] + 1
            else:
                # If the car didn't pass any nodes, all times are penalised
                t_after_nodes = 0

            diff = self.current_step - t_after_nodes
            reward = -1
            for i in range(t_after_nodes, self.current_step):
                self.rewards[i] = reward * (i + 1) / diff
            self.passed_node = False

        do_append = np.random.rand() < len(self.node_passing_times) / 20
        if do_append and (self.has_collided or self.current_step == self.generation_length - 1) and self.current_step > 0:
            self.append_tensors(self.rewards[:self.current_step], self.actions[:self.current_step],
                                self.old_states[:self.current_step], self.states[:self.current_step])

        if len(self.node_passing_times) > self.max_node:
            self.max_node = len(self.node_passing_times)
            self.save_network()

    def reinforce(self, epochs=1):
        if self.rewards_buffer.shape[0] > self.batch_size:
            self.network.train()

            indices = torch.randperm(self.rewards_buffer.shape[0]).long()
            batch_size = self.batch_size

            end_index = 0
            start_index = 0
            for epoch in range(epochs):
                while end_index < self.rewards_buffer.shape[0]:
                    self.optimizer.zero_grad()
                    end_index += batch_size
                    batch_inds = indices[start_index:end_index]
                    q_old = self.network(self.old_states_buffer[batch_inds].to(device))
                    q_new = self.network(self.states_buffer[batch_inds].to(device))

                    q_old_a = q_old[torch.arange(batch_inds.shape[0]), self.actions_buffer[batch_inds]]
                    q_new_max = torch.max(q_new, dim=1)[0]
                    loss = self.loss_function(self.rewards_buffer[batch_inds].to(device) + q_new_max, q_old_a)
                    loss.backward()
                    self.optimizer.step()
                    self.total_loss += loss.detach().cpu().item()
                    start_index += batch_size
            self.total_loss /= epochs
