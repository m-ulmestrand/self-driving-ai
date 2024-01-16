"""
This script allows you to test drive the car yourself, with arrow keys.

Author: Mattias Ulmestrand
"""


from racing_agent import RacingAgent
from collision_handling import get_lidar_lines, line_intersect_distance
import numpy as np
import pygame
from pygame import Surface
from pygame.math import Vector2
from pygame import gfxdraw
import argparse
import math
from pygame_recorder import ScreenRecorder


class Car(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, downscale: int = 20) -> None:
        super().__init__()
        
        car_image = pygame.image.load("sprites/car_grey.png")
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
    
    inner_track_scaled = (inner_track * scale).astype("intc")
    outer_track_scaled = (outer_track * scale).astype("intc")

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
    pygame.draw.aalines(screen, "black", False, inner_track_scaled)
    pygame.draw.aalines(screen, "black", False, outer_track_scaled)
    

def text(s: str, font: pygame.font.Font) -> pygame.Surface:
    return font.render(s, True, (0, 0, 0))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive the car yourself."
    )
    parser.add_argument(
        "--track-name",
        required=False,
        type=str,
        default="racetrack15",
        help="Name of the track."
    )

    parser.add_argument(
        "--save-name",
        default="default_name",
        type=str
    )

    return parser.parse_args()

def main(track_name: str, save_name: str):

    def get_wheel_lines():
        total_angle = agent.angle + agent.turning_angle

        # scale decides how far we are willing to look for an intersection
        scale = box_size * 100
        top_vector = scale * (np.array([-math.sin(total_angle), math.cos(total_angle)]))

        bottom_line = car_bounds[0:2]

        if agent.turning_angle < 0:
            bottom_line = bottom_line[::-1]
            top_vector *= -1

        top_line = car_bounds[6:8]
        top_line[0] = top_line.mean(axis=0)
        top_line[1] = top_line[0] + top_vector
        bottom_vector = scale * (bottom_line[0] - bottom_line[1])
        bottom_line[1] = bottom_line[0] + bottom_vector

        top_distance = line_intersect_distance(top_line[0], top_line[1], bottom_line[0], bottom_line[1])
        bottom_distance = line_intersect_distance(bottom_line[0], bottom_line[1], top_line[0], top_line[1])
        top_line[1] = top_line[0] + top_distance * top_vector
        bottom_line[1] = bottom_line[0] + bottom_distance * bottom_vector

        return bottom_distance, top_distance, bottom_line, top_line, bottom_vector


    def plot_turning_circle():
        bottom_distance, top_distance, bottom_line, top_line, bottom_vector = get_wheel_lines()

        if top_distance != 0:
            pygame.draw.aalines(screen, "blue", False, bottom_line * screen_scale)
            pygame.draw.aalines(screen, "blue", False, top_line * screen_scale)

            focal_point = (top_line[1] * screen_scale).astype(int)
            turning_radius = int(np.linalg.norm(screen_scale * bottom_distance * bottom_vector))
            gfxdraw.filled_circle(screen, *focal_point, 4, (0, 0, 255))
            gfxdraw.aacircle(screen, *focal_point, 4, (0, 0, 255))
            # gfxdraw.aacircle(screen, *focal_point, turning_radius, (0, 0, 255, 120))


    def plot_arc():
        bottom_distance, top_distance, bottom_line, top_line, bottom_vector = get_wheel_lines()

        if top_distance != 0:
            turning_sign = np.sign(agent.turning_angle)
            angle_deg = 1
            angle = np.radians(angle_deg) * turning_sign

            c, s = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[c, -s], [s, c]])
            n_segments = 145 // angle_deg

            bottom_line *= screen_scale
            x = bottom_line[0] - bottom_line[1]
            x = x[:, None]

            for i in range(n_segments):
                x_new = rotation_matrix @ x
                pygame.draw.aaline(screen, (0, 0, 255), bottom_line[1] + x[:, 0], bottom_line[1] + x_new[:, 0])
                x = x_new


    box_size = 100
    screen_scale = 8

    turning_speed = 0.02
    drift = 0.
    acc = 0.01

    agent = RacingAgent(
        box_size=box_size, 
        buffer_size=1, 
        device="cpu",
    )

    agent.store_track(track_name)
    agent.turning_speed=turning_speed
    agent.drift=drift
    agent.acc=acc 
    agent.speed_lower *= 0.1
    agent.velocity *= 0.01
    agent.dec = 0.97

    pygame.init()
    screen_x1 = box_size * screen_scale
    screen_x2 = int(1.3 * screen_x1)
    screen_y = box_size * screen_scale
    slider_width = (screen_x2 - screen_x1) * 0.9
    screen = pygame.display.set_mode((screen_x2, screen_y))
    recorder = ScreenRecorder(screen_x2, screen_y, 60, "./pygame_recordings/" + save_name + ".avi")

    inner_line = np.load(f"tracks/{track_name}_inner_bound.npy")
    outer_line = np.load(f"tracks/{track_name}_outer_bound.npy")

    car = Car(agent.position[0], agent.position[1], downscale=(100 // screen_scale))
    sprites = pygame.sprite.Group(car)

    clock = pygame.time.Clock()
    previous_key = None
    running = True 
    show_turning_circle = False
    show_turning_arc = False

    screen.fill("white")
    draw_track(inner_line, outer_line, screen, screen_scale)
    pygame.image.save(screen, "background/track_background.jpg")
    track_background = pygame.image.load("background/track_background.jpg")
    track_rect = track_background.get_rect()
    
    height = 30
    font = pygame.font.Font(r"C:\Windows\Fonts\timesi.ttf", height)
    texts = [text(s, font) for s in ['v', 'θ', "d₋₂", "d₋₁", "d₀", "d₁", "d₂"]]
    lidar_order = [0, 1, 2, 3, 8, 9, 4, 5, 6, 7]

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        keys = pygame.key.get_pressed() 
        if keys[pygame.K_UP]:
            agent.accelerate()
        if keys[pygame.K_DOWN]:
            agent.decelerate()
        if keys[pygame.K_LEFT]:
            agent.turn_right()
        if keys[pygame.K_RIGHT]:
            agent.turn_left()

        if keys[pygame.K_t]:
            if previous_key != pygame.K_t:
                show_turning_circle = not show_turning_circle
            previous_key = pygame.K_t

        elif keys[pygame.K_a]:
            if previous_key != pygame.K_a:
                show_turning_arc = not show_turning_arc
            previous_key = pygame.K_a
        else:
            previous_key = None
        
        agent.limit_speeds()
        agent.move()

        xs, ys, car_bounds, car_collides = get_lidar_lines(agent)
        agent.current_step += 1
        
        lidar_lines = np.vstack((xs, ys)).T
        car_center = (car_bounds[0, :] + car_bounds[-1, :]) / 2
        scaled_center = screen_scale * car_center

        screen.fill("white")
        screen.blit(track_background, track_rect)
        scaled_lidar_lines = screen_scale * lidar_lines
        pygame.draw.aalines(screen, "black", False, car_bounds * screen_scale)
        pygame.draw.aalines(screen, "red", False, scaled_lidar_lines)

        for i in range(1, scaled_lidar_lines.shape[0], 2):
            circle_xy = scaled_lidar_lines[i].astype(int)
            gfxdraw.filled_circle(screen, *circle_xy, 2, (255, 0, 0))
            gfxdraw.aacircle(screen, *circle_xy, 2, (255, 0, 0))

        if show_turning_circle:
            plot_turning_circle()
        
        if show_turning_arc:
            plot_arc()
        
        rect_x1 = screen_x1
        rect_width = screen_x2 - screen_x1
        rect_y1 = 10
        angle_width = rect_width * (agent.turning_angle / agent.max_angle + 1) / 2
        speed_width = rect_width * np.linalg.norm(agent.velocity) / agent.max_speed

        color = (0, 200, 100)
        border_radius = 10
        pygame.draw.rect(screen, color, pygame.Rect(rect_x1, rect_y1, speed_width, height), border_radius=border_radius)
        pygame.draw.rect(screen, color, pygame.Rect(rect_x1, rect_y1 + height, angle_width, height), border_radius=border_radius)

        ordered_lidar_lines = lidar_lines[lidar_order]
        for count, i in enumerate(range(1, lidar_lines.shape[0], 2), 2):
            line_width = np.linalg.norm((ordered_lidar_lines[i] - ordered_lidar_lines[i - 1]) / box_size * rect_width)
            pygame.draw.rect(screen, color, pygame.Rect(rect_x1, rect_y1 + count * height, line_width, height), border_radius=border_radius)

        for i, txt in enumerate(texts):
            screen.blit(txt, (rect_x1, rect_y1 + height * i))

        sprites.update(scaled_center, agent.angle)
        sprites.draw(screen)

        # pygame.draw.line(screen, "green", scaled_center, scaled_center + agent.velocity * 100)
        # pygame.draw.line(screen, "red", scaled_center, scaled_center + agent.drift_velocity * 100)
        pygame.display.update()
        recorder.capture_frame(screen)
        clock.tick(60)

    recorder.end_recording()
    # pygame.quit()

if __name__ == "__main__":
    args = parse_args()
    main(args.track_name, args.save_name)