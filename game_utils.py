import pygame
from pygame import Surface
from pygame.math import Vector2
from pygame import gfxdraw
from pygame_widgets.slider import Slider
import pygame_widgets
import math
from typing import List
import numpy as np
from racing_agent import RacingAgent


class Car(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, downscale: int = 20, car_path: str = "sprites/car_blue.png") -> None:
        super().__init__()
        
        car_image = pygame.image.load(car_path)
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