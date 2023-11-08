'''
This script is the class for the racing agent. 
A lot of different parameters are available to change for different behaviours.

Author: Mattias Ulmestrand
'''


import torch
from torch import nn, Tensor
from torch.distributions import Categorical
import numpy as np
from typing import Literal, Union, Tuple
from racing_network import DenseNetwork, get_network_classes
from collision_handling import *
from init_cuda import init_cuda
import math
import os.path
import json
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


class RacingAgent:
    def __init__(
        self, 
        box_size: int = 100, 
        car_width: float = 1., 
        car_length: float = 4., 
        lane_width: float = 5.,
        r_min: float = 4., 
        turning_speed: float = 0.25, 
        speed: float = 1., 
        acceleration: float = 0.1,
        deceleration: float = 0.9, 
        speed_lower: float = 0.5, 
        drift: float = 0.,
        turn_radius_decay: float = 1., 
        value_lr: float = 1e-3,
        policy_lr: float = 2e-4, 
        gamma: float = 0.95,
        c_value: float = 0.75,
        c_entropy: float = 0.001,
        ppo_eps: float = 0.2,
        reward_method: Literal["continuous", "rising"] = "continuous",
        sample_size: int = 50, 
        batch_size: int = 50,
        network_type: nn.Module = DenseNetwork,
        buffer_behaviour: Literal["until_full", "discard_old"] = "discard_old",
        gradient_clip: float = 10.0,
        network_params: tuple = (32, 32), 
        seq_length: int = 1, 
        generation_length: int = 1000, 
        track_numbers=np.arange(8), 
        append_scale: int = 20, 
        device: str = "cuda:0",
        name: str = "agent"
    ):

        """
        box_size: Size of the box which the track is contained in. 
        This does not ever need to be changed unless you change it in draw_track.

        car_width, car_length: Width and length of the car
        lane_width: width of the track
        r_min: Minimal turning radius
        turning_speed: How fast the steering wheel can be turned
        speed: Maximal speed of the car
        acceleration: How fast the agent can accelerate 
        deceleration: How fast the agent can decelerate 
        speed_lower: Fraction of maximum speed that the car can minimally decelerate to
        drift: How fast the car can catch up with steering angle. 
            0.0: instant. 1.0: can't catch up at all.
        turn_radius_decay: Controls how much the turning radius decays for higher speeds. 
            Range: 1.0 - inf.
        
        value_lr: Rate of learning for the value optimizer
        policy_lr: Rate of learning for the policy optimizer
        gamma: Discount for future Q-values
        c_value: Emphasis on learning for the value network
        c_entropy: Emphasis on entropy during learning
        ppo_eps: Epsilon in Proximal Policy Optimization
        reward_method: Which reward method to use
        sample_size: Number of samples in training
        batch_size: Batch size for training
        network_type: Which kind of network will be used
        buffer_behaviour: Determines whether to continuously discard old items, 
            or to just fill the buffer until full.
        gradient_clip: Gradient clipping during optimization
        network_params: Network parameters as defined in racing_network.py
        seq_length: Sequence length for using RNN. For DenseNetwork, this should be 1
        generation_length: How long one generation can maximally be
        track_numbers: Which racetrack numbers will be used for training
        target_sync: How long the target network is kept constant
        append_scale: Determines how likely it is to append to replay buffer for a certain number of passed nodes.
            If the number of passed nodes is greater than append_scale, it will always append
        name: Name of the agent, used during saving and loading
        """

        # Various car model parameters
        self.box_size = box_size
        self.diag = box_size * np.sqrt(2)
        self.car_width, self.car_length = car_width, car_length
        self.lane_width = lane_width
        self.r_min = r_min
        self.max_speed = speed
        self.velocity = np.zeros((sample_size, 2))
        self.drift_velocity = self.velocity.copy()
        self.n_actions = 4
        self.n_inputs = 7
        self.angle = np.zeros(sample_size)
        self.drift_angle = self.angle.copy()
        self.position = np.zeros((sample_size, 2))
        self.prev_pos = self.position.copy()
        self.distance = np.zeros(sample_size)
        self.angles = np.array([-0.4, -0.2, 0.2, 0.4]) * np.pi
        self.angles = np.repeat(self.angles[None, ...], sample_size, axis=0)
        self.current_node = np.zeros(sample_size, dtype=int)
        self.max_distance = 0
        self.turning_angle = np.zeros(sample_size)
        self.turning_speed = turning_speed
        self.track_numbers = track_numbers
        self.max_angle = np.pi / 4
        self.acc = acceleration
        self.dec = deceleration
        self.drift = drift
        self.speed_lower = speed_lower
        self.max_angular_vel = speed / (self.r_min * math.tan(math.pi/2 - self.max_angle))
        self.angle_buffer = np.zeros(sample_size)
        self.turn_radius_decay = turn_radius_decay
        self.save_name = name

        # Placeholder for the track
        self.track_nodes = np.array([])
        self.track_outer = np.array([])
        self.track_inner = np.array([])

        # A few variables for checking progress on track
        self.collision_times = np.full(sample_size, np.iinfo(int).max, dtype=int)
        self.node_passing_times = [np.zeros(0, dtype='intc') for _ in range(sample_size)]

        # Neural network parameters and training buffers
        self.buffer_behaviour = buffer_behaviour
        self.gamma = gamma
        self.generation = 0
        self.generation_length = generation_length
        self.current_step = 0
        self.seq_length = seq_length
        self.c_value = c_value
        self.c_entropy = c_entropy
        self.ppo_eps = ppo_eps
        self.network_param_dict = {
            "n_inputs": self.n_inputs,
            "params": network_params,
            "n_outputs": self.n_actions,
            "seq_length": self.seq_length,
            "value_lr": value_lr,
            "policy_lr": policy_lr
        }
        self.network_type = network_type
        self.network = network_type(self.network_param_dict, device)
        self.gradient_clip = gradient_clip
        self.device = init_cuda(device)

        # Learning parameters and various Q-learning parameters
        self.rewards = torch.zeros((sample_size, generation_length), dtype=torch.double) 
        self.actions = torch.full((sample_size, generation_length), -1, dtype=torch.long) 
        self.old_states = torch.zeros(
            (sample_size, generation_length, seq_length, self.n_inputs), 
            dtype=torch.double
        )
        
        self.reward_method = {
            "at_end": self.reward_at_end,
            "discrete": self.reward_discrete,
            "continuous": self.reward_continuous,
            "rising": self.reward_rising
        }[reward_method]

        self.value_loss_function = torch.nn.SmoothL1Loss(reduction="mean").to(device)
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.total_loss = 0
        self.longest_survival = 0
        self.append_scale = append_scale

    def reinitialise(self, keep_progress: bool = False):
        '''Reinitialises the agent'''
        self.current_step = 0
        self.position = np.repeat(self.track_nodes[0][None, ...], self.sample_size, axis=0)
        self.prev_pos = self.position.copy()
        self.turning_angle = np.zeros(self.sample_size)
        diff = self.track_nodes[1] - self.track_nodes[0]
        self.angle = np.full(self.sample_size, np.arctan2(diff[1], diff[0]))
        self.drift_angle = self.angle.copy()
        self.velocity = diff / np.sqrt(np.sum(diff**2)) * self.max_speed * 0.5
        self.velocity = np.repeat(self.velocity[None, ...], self.sample_size, axis=0)
        self.drift_velocity = self.velocity.copy()
        self.node_passing_times = [np.zeros(0, dtype='intc') for _ in range(self.sample_size)]
        
        self.rewards = torch.zeros((self.sample_size, self.generation_length), dtype=torch.double) 
        self.actions = torch.full((self.sample_size, self.generation_length), -1, dtype=torch.long) 
        self.old_states = torch.zeros(
            (self.sample_size, self.generation_length, self.seq_length, self.n_inputs), 
            dtype=torch.double
        )

        self.collision_times = np.full(self.sample_size, np.iinfo(int).max, dtype=int)
        self.current_node = np.zeros(self.sample_size, dtype=int)
        self.total_loss = 0

        if not keep_progress:
            self.generation += 1
            self.distance = np.zeros(self.sample_size)

    def reinitialise_with_track(
            self, track_name: Union[str, int] = None, keep_progress: bool = False
    ):
        '''Reinitialises the agent and stores a random track'''
        self.store_track(track_name)
        self.reinitialise(keep_progress)

    @staticmethod
    def parse_json_config(name: str):
        '''Parses a JSON model config'''
        final_identifier = "final_"
        build_folder = "./build/"
        json_extension = ".json"

        if name.startswith(final_identifier):
            network_param_name = build_folder + name[len(final_identifier):] + json_extension
        else:
            network_param_name = build_folder + name + json_extension
        with open(network_param_name) as parameter_file:
            model_config = json.load(parameter_file)
        
        return model_config
    
    def set_network_params(self, model_config: dict):
        '''Sets network parameters by a model config'''
        cls_list = get_network_classes()
        network_dict = {
            name: cls for (name, cls) in cls_list
        }
        self.network_type = network_dict[model_config["network_type"]]
        self.network_param_dict = {
            "n_inputs": self.n_inputs,
            "params": model_config["params"],
            "n_outputs": self.n_actions,
            "seq_length": model_config["seq_length"],
            "policy_lr": model_config["policy_lr"],
            "value_lr": model_config["value_lr"]
        }
        self.seq_length = model_config["seq_length"]

    def set_agent_params(self, model_config: dict):
        '''Sets agent car model parameters by a model config'''
        self.r_min = model_config["r_min"]
        self.turn_radius_decay = model_config["turn_radius_decay"]
        self.turning_speed = model_config["turning_speed"]
        self.max_speed = model_config["max_speed"]
        self.acc = model_config["acceleration"]
        self.dec = model_config["deceleration"]
        self.speed_lower = model_config["speed_lower"]
        self.drift = model_config["drift"]
        self.max_angle = model_config["max_angle"]

    def load_state_dict(self, network_file_name: str):
        '''Loads network state dicts'''
        checkpoint = torch.load(network_file_name)
        self.network = self.network_type(self.network_param_dict).to(self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.value_optimizer.load_state_dict(
            checkpoint["value_optimizer_state_dict"]
        )
        self.network.policy_optimizer.load_state_dict(
            checkpoint["policy_optimizer_state_dict"]
        )
        
    def load_network(self, name: str = None, model_config: dict = None):
        '''Loads a saved network'''
        if name is None:
            name = self.save_name
        network_file_name = "build/" + name + ".pt"

        if model_config is not None:
            self.set_network_params(model_config)
            self.load_state_dict(network_file_name)
            return None

        if os.path.isfile(network_file_name):
            try:
                model_config = self.parse_json_config(name)
                self.set_network_params(model_config)
            except:
                print("Network " + name + " not found or incorrectly formatted. Using stored settings instead.")
            self.load_state_dict(network_file_name)
        else:
            print("PyTorch checkpoint does not exist. Skipped loading.")

    def save_network(self, name: str = None):
        '''Saves the current network'''
        if name is None:
            name = self.save_name
        name = "build/" + name + ".pt"
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "value_optimizer_state_dict": self.network.value_optimizer.state_dict(),
                "policy_optimizer_state_dict": self.network.policy_optimizer.state_dict(),
                "max_distance": self.max_distance
            }, 
            name
        )
        
        with open("build/" + self.save_name + ".json", 'w') as save_file:
            model_config = {
                "network_type": self.network_type.__name__,
                "params": self.network_param_dict["params"],
                "seq_length": self.seq_length,
                "r_min": self.r_min,
                "turn_radius_decay": self.turn_radius_decay,
                "turning_speed": self.turning_speed,
                "max_speed": self.max_speed,
                "acceleration": self.acc,
                "deceleration": self.dec,
                "speed_lower": self.speed_lower,
                "drift": self.drift,
                "max_angle": self.max_angle,
                "value_lr": self.network_param_dict["value_lr"],
                "policy_lr": self.network_param_dict["policy_lr"]
            }
            json.dump(model_config, save_file, indent=4)

    def store_track(self, track_name: Union[str, int] = None):
        '''Stores a track in the agent class instance'''
        if track_name is None:
            track_name = 'racetrack' + str(np.random.choice(self.track_numbers))
        elif isinstance(track_name, int):
            track_name = 'racetrack' + str(track_name)
        npy = '.npy'
        track_name = 'tracks/' + track_name
        self.track_nodes = np.load(track_name + npy)
        self.track_inner = np.load(track_name + '_inner_bound' + npy)
        self.track_outer = np.load(track_name + '_outer_bound' + npy)
        self.position = np.repeat(self.track_nodes[0][None, ...], self.sample_size, axis=0)
        diff = self.track_nodes[1] - self.track_nodes[0]
        self.velocity = diff / np.sqrt(np.sum(diff ** 2)) * self.max_speed * 0.5
        self.velocity = np.repeat(self.velocity[None, ...], self.sample_size, axis=0)
        self.drift_velocity = self.velocity.copy()
        self.angle = np.full(self.sample_size, np.arctan2(diff[1], diff[0]))
        self.drift_angle = self.angle.copy()

    def move(self):
        '''Moves the agent with current settings'''
        speed = np.sqrt(np.sum(self.velocity ** 2, axis=1))
        angular_vel = speed * self.turn_radius_decay ** (-speed) / (
            self.r_min * np.tan(math.pi/2 - self.turning_angle)
        )
        self.angle = (self.angle + angular_vel) % (2 * math.pi)
        self.angle_buffer += angular_vel
        max_angular_vel = self.max_angular_vel * (1 - self.drift)
        drift_angular_vel = np.maximum(np.minimum(max_angular_vel, self.angle_buffer), -max_angular_vel)
        self.drift_angle = (self.drift_angle + drift_angular_vel) % (2 * math.pi)
        self.angle_buffer -= drift_angular_vel
        self.velocity[:, 0] = np.cos(self.angle) * speed
        self.velocity[:, 1] = np.sin(self.angle) * speed
        self.drift_velocity[:, 0] = np.cos(self.drift_angle) * speed
        self.drift_velocity[:, 1] = np.sin(self.drift_angle) * speed
        self.position += self.drift_velocity

    def get_features(
            self, 
            sample_index: int = 0, 
            other_agents: list = None,
            training: bool = True
        ):
        '''Returns the features: LiDAR line measurements, speed and angle'''
        position = self.position[sample_index]
        nodes = self.track_nodes
        track_outer = self.track_outer
        track_inner = self.track_inner

        node_index = get_node(nodes[:-1], position)

        if node_index > self.current_node[sample_index] or (
            len(self.node_passing_times[sample_index]) > 0 and
            (node_index == 0 and self.current_node[sample_index] == nodes.shape[0] - 3)
            ):
            self.current_node[sample_index] = node_index
            self.node_passing_times[sample_index] = np.append(
                self.node_passing_times[sample_index], self.current_step
            )
            self.distance[sample_index] += np.sqrt(np.sum((position - self.prev_pos[sample_index]) ** 2))
            self.prev_pos[sample_index] = position.copy()

        dist_to_node = np.sqrt(np.sum((position - nodes[node_index]) ** 2))
        angle = self.angle[sample_index]
        vector = np.array([math.cos(angle), math.sin(angle)])
        vector /= np.sqrt(np.sum(vector ** 2))
        car_front = position + vector * self.car_length
        vector *= self.diag
        new_vectors = rotate(self.angles[sample_index], vector.reshape((2, 1)))
        new_vectors = np.append(new_vectors, vector.reshape((1, 2)), axis=0)
        car_bounds = car_lines(position, angle, self.car_width, self.car_length)
        features = torch.ones((1, self.seq_length, self.n_inputs), dtype=torch.double)

        for i, vec in enumerate(new_vectors):
            lidar_line = car_front + vec
            for node_id in np.append(np.arange(node_index - 1, nodes.shape[0] - 1), np.arange(0, node_index - 1)):
                distance_frac = line_intersect_distance(
                    car_front, lidar_line, track_outer[node_id], track_outer[node_id + 1]
                )
                if distance_frac != 0:
                    features[0, -1, i] = distance_frac
                    break
                else:
                    distance_frac = line_intersect_distance(
                        car_front, lidar_line, track_inner[node_id], track_inner[node_id + 1]
                    )
                    if distance_frac != 0:
                        features[0, -1, i] = distance_frac
                        break

            if other_agents is None:
                continue
            
            # If there are other cars present, check if the distance to them is smaller
            for other_agent in other_agents:
                other_lines = car_lines(
                    other_agent.position, other_agent.angle, other_agent.car_width, other_agent.car_length
                )
                for j in (np.arange(3) * 2):
                    distance_frac = line_intersect_distance(car_front, lidar_line,
                                                            other_lines[j], other_lines[j + 1])
                    if distance_frac < features[0, -1, i] and distance_frac != 0:
                        features[0, -1, i] = distance_frac

        car_collides = True if dist_to_node > self.lane_width else False
        if not car_collides:
            for node_id in (node_index, node_index + 1):
                for i in (np.arange(3) * 2):
                    if line_intersect_distance(car_bounds[i], car_bounds[i + 1], track_outer[node_id],
                                               track_outer[node_id + 1]):
                        car_collides = True
                        break
                    elif line_intersect_distance(car_bounds[i], car_bounds[i + 1], track_inner[node_id],
                                                 track_inner[node_id + 1]):
                        car_collides = True
                        break
        
        features[0, -1, -2] = self.turning_angle[sample_index] / self.max_angle
        features[0, -1, -1] = np.sqrt(np.sum(self.velocity[sample_index] ** 2)) / self.max_speed
        
        if self.seq_length > 1:
            if self.current_step != 0:
                features[0, :-1] = self.old_states[self.current_step, 1:]
            else:
                for i in range(self.seq_length - 1):
                    features[0, i] = features[0, -1]

        if car_collides:
            self.collision_times[sample_index] = self.current_step
            if training:
                self.reward_method(sample_index)

        return features

    def turn_left(self, sample_index: int):
        self.turning_angle[sample_index] = min(
            self.turning_angle[sample_index] + self.turning_speed, 
            self.max_angle
        )

    def turn_right(self, sample_index: int):
        self.turning_angle[sample_index] = max(
            self.turning_angle[sample_index] - self.turning_speed, 
            -self.max_angle
        )

    def take_action(self, actions: np.ndarray, indices: np.ndarray):
        '''Performs a selected action'''
        current_speed = np.sqrt(np.sum(self.velocity ** 2, axis=1))
        for i, sample_index in enumerate(indices):
            action = actions[i]
            if action == 0:
                self.turn_left(sample_index)
            elif action == 1:
                self.turn_right(sample_index)
            elif action == 2 and current_speed[sample_index] < self.max_speed:
                # Accelerate
                self.velocity[sample_index, 0] += math.cos(self.angle[sample_index]) * self.acc
                self.velocity[sample_index, 1] += math.sin(self.angle[sample_index]) * self.acc
                new_speed = math.sqrt(np.sum(self.velocity[sample_index] ** 2))
                if new_speed > self.max_speed:
                    self.velocity[sample_index] *= (self.max_speed / new_speed)
            else:
                # Decelerate
                if current_speed[sample_index] > self.speed_lower * self.max_speed:
                    self.velocity[sample_index] = self.velocity[sample_index] * self.dec

        self.move()

    def forward_pass(self, features: Tensor):
        return self.network(features.to(self.device))

    def choose_action(self):
        '''Chooses an action with the policy network'''

        self.network.eval()
        surviving_inds = np.where(self.collision_times > self.generation_length)[0]

        if surviving_inds.size > 0:
            features = torch.ones(
                (0, self.seq_length, self.n_inputs), 
                dtype=torch.double
            )
            for sample_index in surviving_inds:
                sample_features = self.get_features(sample_index)
                if self.collision_times[sample_index] > self.generation_length:
                    features = torch.cat((features, sample_features), dim=0)

            surviving_inds = np.where(self.collision_times > self.generation_length)[0]

        if surviving_inds.size > 0:
            action_logits = self.network(features.to(self.device))
            sampler = Categorical(logits=action_logits)
            sample_actions = sampler.sample().cpu()

            self.take_action(sample_actions.numpy(), surviving_inds)
            self.actions[surviving_inds, self.current_step] = sample_actions
            self.old_states[surviving_inds, self.current_step] = features

    def multi_agent_choose_action(self, agents: list):
        '''Used for simultaneously deciding what actions to take for an ensemble of cars'''
        n_agents = len(agents)
        features_list = [0] * n_agents

        for i in np.arange(n_agents):
            other_agents = [agent for j, agent in enumerate(agents) if j != i]
            features_list[i] = agents[i].get_features(other_agents)[0]

        features = torch.stack(features_list)
        outputs = self.network(features.to(self.device))
        actions = torch.argmax(outputs, dim=1).detach().cpu()

        for agent, action in zip(agents, actions):
            agent.take_action(action)

    def reward_at_end(self, sample_index: int):
        self.rewards[sample_index, self.current_step] = self.distance[sample_index]

    def reward_discrete(self, sample_index: int):
        passing_times = self.node_passing_times[sample_index]
        previous_times = np.append(0, passing_times)
        diffs = passing_times - previous_times[:-1]
        rewards = 1 / diffs

        if not self.current_step == self.generation_length:
            final_reward = -1
        else:
            final_reward = 0
        
        self.rewards[sample_index, passing_times, 0] = torch.from_numpy(rewards)
        self.rewards[sample_index, self.current_step] = final_reward

    def reward_rising(self):
        '''Gives rewards that rises between node passing points, then drops'''
        if len(self.node_passing_times) >= 1:
            previous_times = np.append(0, self.node_passing_times[:-1])

            for time1, time in zip(previous_times, self.node_passing_times):
                time2 = time
                diff = time2 - time1
                reward = 1 / diff

                for step, i in enumerate(range(time1, time2)):
                    self.rewards[i] = reward * (step + 1) / diff
            t_after_nodes = self.node_passing_times[-1] + 1

        else:
            # If the car didn't pass any nodes, all times are penalised
            t_after_nodes = 0

        if not self.current_step == self.generation_length:
            diff = self.current_step - t_after_nodes
            reward = -1

            for step, i in enumerate(range(t_after_nodes, self.current_step)):
                self.rewards[i] = reward * (step + 1) / diff
        

    def reward_continuous(self, sample_index: int):
        '''Gives a reward such that a continuous line is drawn between all reward points'''
        passing_times = self.node_passing_times[sample_index]
        previous_times = np.append(0, passing_times)
        diffs = passing_times - previous_times[:-1]
        rewards = 1 / diffs

        if not self.current_step == self.generation_length:
            rewards = np.append(rewards, -1)
        else:
            rewards = np.append(rewards, 0)

        passing_times = np.append(passing_times, self.current_step)
        reward_before = 0

        for pass_time, pass_time_before, reward in zip(passing_times, previous_times, rewards):
            for step, i in enumerate(range(pass_time_before, pass_time)):
                t = (step + 1) / (pass_time - pass_time_before)
                self.rewards[sample_index, i] = (1 - t) * reward_before + t * reward
            
            reward_before = reward

    def reward_per_node(self):
        '''Gives rewards depending on how well the agent performs'''
        self.current_step += 1

        if self.current_step == self.generation_length or (self.collision_times < self.generation_length - 1).all():
            reward_indices = np.where(self.collision_times > self.generation_length)[0]
            
            for sample_index in reward_indices:
                self.reward_method(sample_index)

            max_distance = self.distance.max()
            if max_distance >= self.max_distance:
                self.max_distance = max_distance
                self.save_network()

    def get_returns(self) -> Tensor:
        """Returns the weighted cumulative expected future rewards"""
        expected_returns = torch.zeros_like(self.rewards)
        for i in range(self.sample_size):
            for t in range(self.collision_times[i]):         
                weights = self.gamma ** torch.arange(self.collision_times[i] - t)[:, None]
                weighted_rewards = weights * self.rewards[i, t: self.collision_times[i]]
                current_returns = weighted_rewards.sum()
                expected_returns[i, t] = current_returns
        return expected_returns

    def get_advantages(self, returns: Tensor, values: Tensor = None) -> Tensor:
        if values is not None:
            advantages = returns - values
        else:
            advantages = returns
        return ((advantages - advantages.mean()) / advantages.std())

    def optimize(
        self, 
        states: Tensor, 
        actions: Tensor,
        returns: Tensor, 
        old_log_probs: Tensor
    ):
        '''Proximal Policy Optimization step'''
        new_policy, values = self.network(states, get_values=True)
        values = values.flatten()
        new_dist = Categorical(logits=new_policy)
        new_log_probs = new_dist.log_prob(actions)

        # Clamp ratio to not blow up exponential
        policy_ratio = (new_log_probs - old_log_probs).clamp(None, 10).exp()

        advantages = self.get_advantages(returns, values)
        weighted_ratio = advantages * policy_ratio
        clipped_ratio = advantages * policy_ratio.clamp(1 - self.ppo_eps, 1 + self.ppo_eps)

        # Entropy is subtracted to promote exploration
        policy_expectation =  torch.minimum(weighted_ratio, clipped_ratio).mean()
        value_loss = self.c_value * self.value_loss_function(values, returns)
        entropy = self.c_entropy * new_dist.entropy().mean()
        total_loss = -policy_expectation + value_loss - entropy

        self.network.policy_optimizer.zero_grad()
        self.network.value_optimizer.zero_grad()
        total_loss.backward()
        
        gradient_clip = self.gradient_clip
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), gradient_clip)
        self.network.policy_optimizer.step()
        self.network.value_optimizer.step()

        self.policy_expectation += policy_expectation.detach().cpu().item()
        self.value_loss += value_loss.detach().cpu().item()
        self.entropy += entropy.detach().cpu().item()

    def policy_gradient_optimize(self, states: Tensor, actions: Tensor, returns: Tensor):
        logits = self.network(states)
        log_probs = -F.cross_entropy(logits, actions, reduction="none")
        loss = -log_probs * returns

        self.network.policy_optimizer.zero_grad()
        loss.mean().backward()
        self.network.policy_optimizer.step()
        self.policy_expectation += loss.mean().item()


    def reinforce(self, n_epochs: int = 1):
        '''Reinforcement learning training'''
        self.network.train()
        batch_size = self.batch_size
        self.policy_expectation = 0.
        self.value_loss = 0.
        self.entropy = 0.

        actions = self.actions.flatten(0, 1)
        indices = torch.nonzero(actions >= 0)[:, 0]
        n_data = indices.size(dim=0)

        # If one batch only contains one data point, BatchNorm will complain
        if n_data % batch_size == 1:
            indices = indices[:-1]
            actions = actions[:-1]
            n_data = indices.size(dim=0)

        actions = actions[indices]
        states = self.old_states.flatten(0, 1)[indices]
        returns = self.get_returns().flatten(0, 1)[indices]
        old_log_probs = torch.zeros(n_data).to(self.device)

        b = 0
        while b < n_data:
            old_policy = self.network(states[b: b + batch_size].to(self.device))
            old_dist = Categorical(logits=old_policy)
            old_log_probs[b: b + batch_size] = old_dist.log_prob(
                actions[b: b + batch_size].to(self.device)
            )
            b += batch_size
        
        old_log_probs = old_log_probs.detach()

        for _ in range(n_epochs):
            permutation = torch.randperm(n_data)
            actions = actions[permutation].to(self.device)
            states = states[permutation].to(self.device)
            returns = returns[permutation].to(self.device)
            old_log_probs = old_log_probs[permutation].to(self.device)

            b = 0
            while b < n_data:
                self.optimize(
                    states[b: b + batch_size],
                    actions[b: b + batch_size],
                    returns[b: b + batch_size],
                    old_log_probs[b: b + batch_size]
                )
                b += batch_size