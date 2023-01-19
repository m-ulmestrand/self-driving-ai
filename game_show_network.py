'''
This script allows you to test a trained agent out on a track of your choice.
Just change the racetrack name to the one you want to try out.

Author: Mattias Ulmestrand
'''


from racing_agent import RacingAgent
from collision_handling import get_lidar_lines, line_intersect_distance
import numpy as np
import pygame
from pygame import Surface
from pygame.math import Vector2
from pygame import gfxdraw
from pygame_widgets.slider import Slider
import pygame_widgets
import argparse
import math
from time import perf_counter
from typing import List
from matplotlib.cm import get_cmap


class Car(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, downscale: int = 20) -> None:
        super().__init__()
        
        car_image = pygame.image.load('sprites/car_blue.png')
        width, height = car_image.get_width() // downscale, car_image.get_height() // downscale
        self.width = height
        self.length = width
        self.base_image = pygame.transform.scale(car_image, (width, height)).convert_alpha()
        self.image = self.base_image
        self.rect = self.base_image.get_rect()
        self.offset = Vector2(self.length / 2, 0)
        self.x = x
        self.y = y
    
    def set_pos(self, position: np.ndarray):
        self.rect.centerx = position[0]
        self.rect.centery = position[1]

    def update(self, position: np.ndarray, angle: float):
        self.rotate(angle)
        self.set_pos(position)

    def rotate(self, angle: float):
        pi = np.pi
        angle_degrees = angle / pi * 180
        self.image = pygame.transform.rotate(self.base_image, (-angle_degrees))
        offset_rotated = self.offset.rotate(angle_degrees)
        self.rect = self.image.get_rect(center=offset_rotated)


def draw_track(inner_track: np.ndarray,
               outer_track: np.ndarray,
               screen: Surface,
               scale: float,
               fill_color: tuple = (190, 190, 190)):
    
    inner_track_scaled = (inner_track * scale).astype('intc')
    outer_track_scaled = (outer_track * scale).astype('intc')

    # Filling the track 
    for i in np.arange(inner_track.shape[0] - 1):
        gfxdraw.filled_trigon(screen, 
                              *inner_track_scaled[i], 
                              *inner_track_scaled[i + 1], 
                              *outer_track_scaled[i], 
                              fill_color)

        gfxdraw.filled_trigon(screen, 
                              *outer_track_scaled[i], 
                              *outer_track_scaled[i + 1], 
                              *inner_track_scaled[i + 1], 
                              fill_color)

    # Drawing track outline
    pygame.draw.aalines(screen, 'black', False, inner_track_scaled)
    pygame.draw.aalines(screen, 'black', False, outer_track_scaled)


def draw_network(x_vals: np.ndarray, 
                 y_vals: list,
                 node_vals: list,
                 edge_vals: list, 
                 screen: Surface):

    '''cmap = get_cmap('seismic')
    for i, (x1, x2) in enumerate(zip(x_vals[:-1], x_vals[1:])):
        edges = (1 - 1 / (1 + np.exp(-5 * edge_vals[i].T)))

        for j, y1 in enumerate(y_vals[i]):
            for k, y2 in enumerate(y_vals[i + 1]):
                c = cmap(edges[j, k])
                color = np.array(c)
                color[:3] *= 255
                color[-1] = 0.75
                pygame.draw.aaline(screen, color, (x1, y1), (x2, y2))'''

    for i, ys in enumerate(y_vals[:-1]):
        nodes = node_vals[i] / node_vals[i].max()
        node_cols = (1 - 1 / (1 + np.exp(-5 * (nodes - 0.75)))).flatten()
        for j, y in enumerate(ys):
            c = node_cols[j] * 255
            # pygame.draw.circle(screen, (c, c, c), (x_vals[i], y), 10)
            gfxdraw.filled_circle(screen, x_vals[i], y, 10, (c, c, c))
            gfxdraw.aacircle(screen, x_vals[i], y, 10, (0, 0, 0))
    
    nodes = node_vals[-1] / node_vals[-1].max()
    node_cols = (1 - 1 / (1 + np.exp(-5 * (nodes - 0.75)))).flatten()
    max_ind = np.argmax(node_vals[-1])
    
    for j, y in enumerate(y_vals[-1]):
        c = node_cols[j] * 255
        outer_color = (255, 0, 0) if j == max_ind else (0, 0, 0)
        inner_color = (c, c, c)
        gfxdraw.filled_circle(screen, x_vals[-1], y, 14, outer_color)
        gfxdraw.filled_circle(screen, x_vals[-1], y, 10, inner_color)
        gfxdraw.aacircle(screen, x_vals[-1], y, 10, inner_color)
        gfxdraw.aacircle(screen, x_vals[-1], y, 14, outer_color)



def main():
    parser = argparse.ArgumentParser(
        description='Run a simulation with a trained agent.'
    )
    parser.add_argument(
        "--agent-name",
        required=False,
        help='Name of the pretrained agent.'
    )
    parser.add_argument(
        '--track-name',
        required=False,
        help='Name of the track.'
    )

    args = parser.parse_args()
    box_size = 100
    screen_scale = 10

    max_turn_angle = math.pi / 4
    turning_speed = 0.125
    drift = 0.0
    acc = 0.005
    agent = RacingAgent(
        box_size=box_size, 
        buffer_size=1, 
        turning_speed=turning_speed, 
        drift=drift, 
        acceleration=acc, 
        device='cpu'
    )

    # Change race_car.save_name use an agent of your choice.
    # Trained agents are saved at ./build, load just the name without the .pt extension.
    # Both the final agent and the best performing one are saved.
    agent.save_name = 'agent_dense'
    agent.load_network(name=agent.save_name)

    # Change track to the one you want to try out.
    # Tracks are saved at ./tracks
    track = 'racetrack14'

    # If you have specified racetrack or agent in command line, these will be used instead.
    if args.track_name is not None:
        track = args.track_name
    
    if args.agent_name is not None:
        agent.save_name = args.agent_name

    agent.store_track(track)

    pygame.init()
    screen_x1 = box_size * screen_scale
    screen_x2 = 1.9 * screen_x1
    screen_y = box_size * screen_scale
    screen = pygame.display.set_mode((screen_x2, screen_y))

    inner_line = np.load(f'tracks/{track}_inner_bound.npy')
    outer_line = np.load(f'tracks/{track}_outer_bound.npy')

    car = Car(agent.position[0], agent.position[1], downscale=(200 // screen_scale))
    sprites = pygame.sprite.Group(car)

    network_params = agent.network_params
    max_neurons = max(network_params)

    neurons_x = np.linspace(screen_x1, screen_x2 * 0.95, len(network_params))
    neurons_x = neurons_x.astype('intc')
    neurons_y = [None for _ in network_params]

    for i, n_neurons in enumerate(network_params):
        neurons_y[i] = np.arange(n_neurons) / max_neurons * screen_y * 0.9
        neurons_y[i] -= neurons_y[i].mean()
        neurons_y[i] += screen_y / 2
        neurons_y[i] = neurons_y[i].astype('intc')
    
    clock = pygame.time.Clock()
    running = True 

    screen.fill('white')
    draw_track(inner_line, outer_line, screen, screen_scale)
    pygame.image.save(screen, "background/track_background.jpg")
    track_background = pygame.image.load("background/track_background.jpg")
    track_rect = track_background.get_rect()
    node_statistics = [None for _ in range(len(agent.network.layers) + 1)]
    layer = agent.network.layers[0].weight.detach().cpu().numpy()
    n_statistics = 500
    node_statistics[0] = np.zeros((n_statistics, 1, layer.shape[1]))

    for i, layer in enumerate(agent.network.layers, 1):
        n_nodes = layer.weight.detach().cpu().numpy().shape[0]
        node_statistics[i] = np.zeros((n_statistics, 1, n_nodes))
    
    count = 0
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        _, nodes, edges = agent.choose_action(epsilon=0., return_hidden_states=True)
        nodes = [node.cpu().detach().numpy() for node in nodes]
        edges = [edge.cpu().detach().numpy() for edge in edges]
        xs, ys, car_bounds, _ = get_lidar_lines(agent)
        lidar_lines = np.vstack((xs, ys)).T
        car_center = (car_bounds[0, :] + car_bounds[-1, :]) / 2

        screen.fill('white')
        screen.blit(track_background, track_rect)
        pygame.draw.lines(screen, 'red', False, lidar_lines * screen_scale, width=2)
        sprites.update(car_center * screen_scale, agent.angle)
        sprites.draw(screen)
        draw_network(neurons_x, neurons_y, nodes, edges, screen)
        pygame.display.update()
        clock.tick(60)

        if count < n_statistics:
            for i, node in enumerate(nodes):
                node_statistics[i][count] = node
        
        elif count == n_statistics:
            '''for statistics in node_statistics:
                print("Mean:")
                print(statistics.mean(axis=0))
                print("Standard dev:")
                print(statistics.std(axis=0))
                print()'''
            running = False
        count += 1

    all_means = [node_statistics[i].mean(axis=0).flatten() for i in range(len(node_statistics))]
    all_stds = [node_statistics[i].std(axis=0).flatten() for i in range(len(node_statistics))]
    # agent.network.prune_verified_dead_neurons()
    agent.network.prune_dead_neurons(all_means[1:-1], all_stds[1:-1])
    agent.network_params = agent.network.n_neurons

    network_params = agent.network_params
    max_neurons = max(network_params)

    neurons_x = np.linspace(screen_x1, screen_x2 * 0.95, len(network_params))
    neurons_x = neurons_x.astype('intc')
    neurons_y = [None for _ in network_params]

    for i, n_neurons in enumerate(network_params):
        neurons_y[i] = np.arange(n_neurons) / max_neurons * screen_y * 0.9
        neurons_y[i] -= neurons_y[i].mean()
        neurons_y[i] += screen_y / 2
        neurons_y[i] = neurons_y[i].astype('intc')

    running = True

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        _, nodes, edges = agent.choose_action(epsilon=0., return_hidden_states=True)
        nodes = [node.cpu().detach().numpy() for node in nodes]
        edges = [edge.cpu().detach().numpy() for edge in edges]
        xs, ys, car_bounds, _ = get_lidar_lines(agent)
        lidar_lines = np.vstack((xs, ys)).T
        car_center = (car_bounds[0, :] + car_bounds[-1, :]) / 2

        screen.fill('white')
        screen.blit(track_background, track_rect)
        pygame.draw.lines(screen, 'red', False, lidar_lines * screen_scale, width=2)
        sprites.update(car_center * screen_scale, agent.angle)
        sprites.draw(screen)
        draw_network(neurons_x, neurons_y, nodes, edges, screen)
        pygame.display.update()
        clock.tick(60)
        count += 1

    pygame.quit()

if __name__ == "__main__":
    main()
