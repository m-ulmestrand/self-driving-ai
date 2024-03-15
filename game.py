'''
This script allows you to test a trained agent out on a track of your choice.
Just change the racetrack name to the one you want to try out.

Author: Mattias Ulmestrand
'''


from racing_agent import RacingAgent
from collision_handling import get_lidar_lines, line_intersect_distance
import numpy as np
import pygame
from pygame import gfxdraw
import argparse
import math
from pygame_recorder import ScreenRecorder
from game_utils import Car, Sliders, draw_track


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a simulation with a trained agent.'
    )
    parser.add_argument(
        "--agent-name",
        required=False,
        type=str,
        default="agent_dense",
        help="Name of the pretrained agent."
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

def main(agent_name: str, track_name: str, save_name: str):

    box_size = 100
    screen_scale = 10

    max_turn_angle = math.pi / 4
    turning_speed = 0.125
    drift = 0.0
    acc = 0.005
    model_config = RacingAgent.parse_json_config(agent_name)

    agent = RacingAgent(
        box_size=box_size, 
        buffer_size=1, 
        device='cpu',
        seq_length=model_config["seq_length"],
    )

    agent.save_name = agent_name
    agent.store_track(track_name)
    agent.load_network(model_config=model_config)
    agent.set_agent_params(model_config)
    agent.turning_speed=turning_speed
    agent.drift=drift
    agent.acc=acc 

    pygame.init()
    screen_x1 = box_size * screen_scale
    screen_x2 = int(1.3 * screen_x1)
    screen_y = box_size * screen_scale
    slider_width = (screen_x2 - screen_x1) * 0.9
    screen = pygame.display.set_mode((screen_x2, screen_y))
    recorder = ScreenRecorder(screen_x2, screen_y, 60, "./pygame_recordings/" + save_name + ".avi")

    inner_line = np.load(f'tracks/{track_name}_inner_bound.npy')
    outer_line = np.load(f'tracks/{track_name}_outer_bound.npy')
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
        agent.current_step += 1
        
        if car_collides:
            running = False
            np.save("passing_times.npy", agent.node_passing_times)
        lidar_lines = np.vstack((xs, ys)).T
        car_center = (car_bounds[0, :] + car_bounds[-1, :]) / 2
        scaled_center = screen_scale * car_center

        screen.fill('white')
        screen.blit(track_background, track_rect)
        scaled_lidar_lines = screen_scale * lidar_lines
        # pygame.draw.aalines(screen, 'black', False, car_bounds * screen_scale)
        pygame.draw.aalines(screen, 'red', False, scaled_lidar_lines)

        for i in range(1, scaled_lidar_lines.shape[0], 2):
            circle_xy = scaled_lidar_lines[i].astype(int)
            gfxdraw.filled_circle(screen, *circle_xy, 2, (255, 0, 0))
            gfxdraw.aacircle(screen, *circle_xy, 2, (255, 0, 0))

        sprites.update(scaled_center, agent.angle)
        sprites.draw(screen)
        sliders.update(events, agent)

        # pygame.draw.line(screen, "green", scaled_center, scaled_center + agent.velocity * 100)
        # pygame.draw.line(screen, "red", scaled_center, scaled_center + agent.drift_velocity * 100)
        pygame.display.update()
        recorder.capture_frame(screen)
        clock.tick(60)

    recorder.end_recording()
    # pygame.quit()

if __name__ == "__main__":
    args = parse_args()
    main(args.agent_name, args.track_name, args.save_name)