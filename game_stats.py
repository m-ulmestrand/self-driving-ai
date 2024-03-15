"""
This script allows you to test drive the car yourself, with arrow keys.

Author: Mattias Ulmestrand
"""


from racing_agent import RacingAgent
from collision_handling import get_lidar_lines, line_intersect_distance
import numpy as np
import pygame
from pygame import gfxdraw
import argparse
import math
from pygame_recorder import ScreenRecorder
from game_utils import Car, draw_track


def text(s: str, font: pygame.font.Font) -> pygame.Surface:
    return font.render(s, True, (0, 0, 0))


def draw_rewards(
    screen: pygame.Surface, 
    x1: int, 
    x2: int, 
    y1: int, 
    y2: int, 
    rewards: np.ndarray, 
    has_collided: bool,
    reward_max: float = 1.0,
    n_rewards: int = 1000
):
    black = (0, 0, 0)
    y_mid = (y2 + y1) // 2
    y_diff = (y2 - y1) // 2
    pos1 = np.array([x1, y1])
    pos2 = np.array([x1, y2])
    pygame.draw.line(screen, black, pos1, pos2, width=3)

    pos1 = np.array([x1, y_mid])
    pos2 = np.array([x2, y_mid])
    pygame.draw.line(screen, black, pos1, pos2, width=3)
    
    if not has_collided:
        rewards = rewards[rewards > 0]

    if rewards.size > 2:
        x = np.linspace(x1, x2, n_rewards)[:rewards.size - 1]
        r = rewards[:rewards.size - 1, None]
        if has_collided:
            r[-1] = -1

        r = np.append(x[:, None], r[:rewards.size - 1] * y_diff / reward_max + y_mid, axis=1)
        pygame.draw.aalines(screen, black, closed=False, points=r)


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
        default="default_stats",
        type=str
    )

    parser.add_argument(
        "--agent-name",
        required=False,
        type=str,
        default=None,
        help="Name of the pretrained agent."
    )

    parser.add_argument(
        "--generation-length",
        required=False,
        type=int,
        default=1000,
    )

    return parser.parse_args()


def main(track_name: str, save_name: str, agent_name: str, generation_length: int):

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
    screen_scale = 5

    turning_speed = 0.02
    drift = 0.
    acc = 0.01

    model_params = {
        "box_size": box_size,
        "buffer_size": 1,
        "device": "cpu",
        "generation_length": generation_length
    }

    if agent_name is not None:
        model_config = RacingAgent.parse_json_config(agent_name)
        model_params["seq_length"] = model_config["seq_length"]
        agent = RacingAgent(**model_params)
        agent.load_network(agent_name)
        agent.set_agent_params(model_config)
        agent.store_track(track_name)
    else:
        agent = RacingAgent(**model_params)
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
    screen_x3 = 2 * screen_x2
    screen_y = box_size * screen_scale
    margin = screen_y - screen_y * 0.9
    collide_counter = 0

    screen = pygame.display.set_mode((screen_x3, screen_y))
    recorder = ScreenRecorder(screen_x3, screen_y, 60, "./pygame_recordings/" + save_name + ".avi")

    inner_line = np.load(f"tracks/{track_name}_inner_bound.npy")
    outer_line = np.load(f"tracks/{track_name}_outer_bound.npy")

    car = Car(agent.position[0], agent.position[1], downscale=(100 // screen_scale), car_path="sprites/car_grey.png")
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
    y_label = text("R(t)", font)
    x_label = text("t", font)
    lidar_order = [0, 1, 2, 3, 8, 9, 4, 5, 6, 7]

    while agent.current_step < generation_length and running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        keys = pygame.key.get_pressed() 

        if agent_name is not None:
            agent.choose_action(epsilon=0.)
        else:
            if keys[pygame.K_UP]:
                agent.accelerate()
            if keys[pygame.K_DOWN]:
                agent.decelerate()
            if keys[pygame.K_LEFT]:
                agent.turn_right()
            if keys[pygame.K_RIGHT]:
                agent.turn_left()
            agent.limit_speeds()
            agent.move()

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

        xs, ys, car_bounds, car_collides = get_lidar_lines(agent)

        if car_collides:
            collide_counter += 1
        else: 
            collide_counter = 0

        agent.get_features()
        agent.reward_continuous()
        rewards = agent.rewards.numpy().flatten()
        agent.current_step += 1
        
        lidar_lines = np.vstack((xs, ys)).T
        car_center = (car_bounds[0, :] + car_bounds[-1, :]) / 2
        scaled_center = screen_scale * car_center

        screen.fill("white")
        screen.blit(track_background, track_rect)
        draw_rewards(
            screen, 
            screen_x2, 
            screen_x3, 
            screen_y - margin, 
            margin, 
            rewards[:agent.current_step], 
            car_collides, 
            n_rewards=generation_length
        )
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
            line_width = np.linalg.norm((ordered_lidar_lines[i] - ordered_lidar_lines[i - 1]) / (box_size * np.sqrt(2)) * rect_width)
            pygame.draw.rect(screen, color, pygame.Rect(rect_x1, rect_y1 + count * height, line_width, height), border_radius=border_radius)

        for i, txt in enumerate(texts):
            screen.blit(txt, (rect_x1, rect_y1 + height * i))
        
        screen.blit(x_label, (screen_x3 - screen_x2 // 2, screen_y // 2 + 20))
        screen.blit(y_label, (screen_x2, 10))

        sprites.update(scaled_center, agent.angle)
        sprites.draw(screen)

        # pygame.draw.line(screen, "green", scaled_center, scaled_center + agent.velocity * 100)
        # pygame.draw.line(screen, "red", scaled_center, scaled_center + agent.drift_velocity * 100)
        pygame.display.update()
        recorder.capture_frame(screen)
        clock.tick(60)

        if collide_counter > 0:
            running = False
            clock.tick(0.5)

    recorder.end_recording()
    # pygame.quit()

if __name__ == "__main__":
    args = parse_args()
    main(args.track_name, args.save_name, args.agent_name, args.generation_length)