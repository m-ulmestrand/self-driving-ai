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


class Sliders:
    def __init__(self, screen: Surface, 
                 x: int, 
                 width: int, 
                 height: int,
                 drift: float = 0.0,
                 acc: float = 0.01,
                 turn: float = (math.pi / 4)) -> None:

        self.x = x
        self.width = width
        self.height = height
        self.font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 32)
        self.drift_text = self.font.render('Drift tendency', False, (0, 0, 0))
        self.acc_text   = self.font.render('Acceleration',   False, (0, 0, 0))
        self.turn_text  = self.font.render('Max turn angle', False, (0, 0, 0))
        self.screen = screen

        pi = math.pi
        self.drift_slider = Slider(screen, x, 50,  width, height, min=0.0,   max=1.0,  step=0.01,  initial=drift)
        self.acc_slider   = Slider(screen, x, 150, width, height, min=0.001, max=0.1,  step=0.001, initial=acc)
        self.turn_slider  = Slider(screen, x, 250, width, height, min=pi/16, max=pi/4, step=0.05,  initial=turn)

    def update(self, events: List[pygame.event.Event], agent: RacingAgent):
        pygame_widgets.update(events)
        agent.drift = self.drift_slider.getValue()
        agent.acc = self.acc_slider.getValue()
        agent.max_angle = self.turn_slider.getValue()
        self.screen.blit(self.drift_text, (self.x, 10))
        self.screen.blit(self.acc_text, (self.x, 110))
        self.screen.blit(self.turn_text, (self.x, 210))


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


def main():
    parser = argparse.ArgumentParser(
        description='Run a simulation with a trained agent.'
    )
    parser.add_argument(
        "--agent-name",
        required=False,
        default="agent_dense",
        help="Name of the pretrained agent."
    )
    parser.add_argument(
        "--track-name",
        required=False,
        default="racetrack15",
        help="Name of the track."
    )

    args = parser.parse_args()
    box_size = 100
    screen_scale = 10

    max_turn_angle = math.pi / 4
    turning_speed = 0.125
    drift = 0.0
    acc = 0.005
    model_config = RacingAgent.parse_json_config(args.agent_name)

    agent = RacingAgent(
        box_size=box_size, 
        buffer_size=1, 
        turning_speed=turning_speed, 
        drift=drift, 
        acceleration=acc, 
        device='cpu',
        seq_length=model_config["seq_length"],
    )

    track = args.track_name
    agent.save_name = args.agent_name
    agent.store_track(track)
    agent.load_network(model_config=model_config)

    pygame.init()
    screen_x1 = box_size * screen_scale
    screen_x2 = 1.3 * screen_x1
    screen_y = box_size * screen_scale
    slider_width = (screen_x2 - screen_x1) * 0.9
    screen = pygame.display.set_mode((screen_x2, screen_y))

    inner_line = np.load(f'tracks/{track}_inner_bound.npy')
    outer_line = np.load(f'tracks/{track}_outer_bound.npy')
    # nodes = np.load(f'tracks/{track}.npy')

    car = Car(agent.position[0], agent.position[1], downscale=(200 // screen_scale))
    sprites = pygame.sprite.Group(car)
    sliders = Sliders(screen, screen_x1, slider_width, 20, drift=drift, acc=acc, turn=max_turn_angle)

    clock = pygame.time.Clock()
    running = True 

    screen.fill('white')
    draw_track(inner_line, outer_line, screen, screen_scale)
    pygame.image.save(screen, "background/track_background.jpg")
    track_background = pygame.image.load("background/track_background.jpg")
    track_rect = track_background.get_rect()
    
    while running and agent.current_step < agent.generation_length:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        agent.choose_action(epsilon=0.)
        xs, ys, car_bounds, car_collides = get_lidar_lines(agent)
        lidar_lines = np.vstack((xs, ys)).T
        car_center = (car_bounds[0, :] + car_bounds[-1, :]) / 2
        scaled_center = screen_scale * car_center

        screen.fill('white')
        screen.blit(track_background, track_rect)
        pygame.draw.lines(screen, 'black', False, car_bounds * screen_scale)
        pygame.draw.lines(screen, 'red', False, lidar_lines * screen_scale, width=2)
        sprites.update(scaled_center, agent.angle)
        sprites.draw(screen)
        sliders.update(events, agent)

        # pygame.draw.line(screen, "green", scaled_center, scaled_center + agent.velocity * 100)
        # pygame.draw.line(screen, "red", scaled_center, scaled_center + agent.drift_velocity * 100)
        pygame.display.update()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
