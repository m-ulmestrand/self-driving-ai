'''
This file is for drawing your own tracks.
When running the script, hold A and draw a curve with the mouse.
Nodes will appear on the plot, and when you have connected
the final node to the first one (this will happen when close enough),
two boundary lines will be made.
When you draw the line, be careful not to make too sharp corners.
This will produce yanks in the track - dealing with that problem is difficult.

When exiting the plot, you will be asked if you want to save the track.
Press yes/y or no/n to confirm your answer (case insensitive).

Author: Mattias Ulmestrand
'''


from collision_handling import line_intersect_distance
from typing import Callable
from pygame import gfxdraw
import numpy as np
import os.path
import pygame
import math


def angle_change(v1: np.ndarray, v2: np.ndarray):
    return math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v2[0] * v1[0] + v2[1] * v1[1])


def add_borders(nodes: np.ndarray,
                track_width) -> tuple:
    
    if not isinstance(track_width, (np.ndarray, list, tuple)):
        track_width = np.ones(nodes.shape[0]) * track_width

    outer_line = np.zeros_like(nodes)
    inner_line = np.zeros_like(nodes)

    new_nodes = nodes[:-1]
    add_border(new_nodes[-1], new_nodes[0], new_nodes[1], inner_line, outer_line, 0, track_width[0])

    for i, node in enumerate(zip(new_nodes[:-2], new_nodes[1:-1], new_nodes[2:])):
        add_border(*node, inner_line, outer_line, i + 1, track_width[i + 1])

    add_border(nodes[-3], nodes[-2], nodes[-1], inner_line, outer_line, nodes.shape[0] - 2, track_width[-2])
    # add_borders(nodes[-2], nodes[-1], nodes[0], inner_line, outer_line, nodes.shape[0] - 2, track_width)
    outer_line[-1] = outer_line[0]
    inner_line[-1] = inner_line[0]

    return inner_line, outer_line


def add_border(node1: np.ndarray, 
               node2: np.ndarray, 
               node3: np.ndarray, 
               inner_line: np.ndarray,
               outer_line: np.ndarray,
               i: int, 
               track_width: float):

    v1 = node2 - node1
    v2 = node3 - node2

    gradient = v1 + v2
    distance = np.sqrt(np.sum(gradient ** 2))

    # Direction of gradient at node i + 1
    direction = gradient / distance
    
    # Orthogonal to [x_diff, y_diff]
    width_vect = np.array([-direction[1], direction[0]]) * track_width
    
    outer_line[i] = node2 - width_vect
    inner_line[i] = node2 + width_vect


def replace_average(points: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(np.mean(points, axis=0)[:, None], n, axis=0)


def replace_bezier(points: np.ndarray, n: int) -> np.ndarray:
    v10 = points[1] - points[0]
    v21 = points[2] - points[1]

    replaced_points = np.zeros((n, 2))

    for i, t in enumerate(np.arange(n) / n):
        p3 = points[0] + t * v10
        p4 = points[1] + t * v21
        v43 = p4 - p3
        replaced_points[i] = p3 + t * v43
    
    return replaced_points


def remove_loops(line: np.ndarray, n_points: int, replace_method: Callable = replace_bezier) -> None:
    '''If a 180 degree rotation occurs, all points in the line segment
       are replaced with some interpolation method.'''
    
    replaced_pts = np.zeros((0, 2))
    n_pts_total = line.shape[0]
    i = 0

    while i < (line.shape[0]):
        d_theta = 0.0
        v1 = line[(i + 1) % n_pts_total] - line[i]

        for j in np.arange(i + 1, i + n_points - 1):
            v2 = line[(j + 1) % n_pts_total] - line[j % n_pts_total]
            d_theta += angle_change(v1, v2)
            v1 = v2.copy()
        
        if abs(d_theta) > math.pi:
            if line_intersect_distance(line[i], line[(i + 1) % n_pts_total], 
                                       line[(i + n_points - 2) % n_pts_total], 
                                       line[(i + n_points - 1) % n_pts_total]):

                surplus = i + n_points - line.shape[0]
                if surplus <= 0:
                    loop = line[i: i + n_points]
                    replaced_pts = np.append(replaced_pts, loop, axis=0)
                    points = np.vstack((loop[0:1], np.mean(loop, axis=0), loop[-1:]))
                    line[i: i + n_points] = replace_method(points, n_points)
                else:
                    loop = np.append(line[i: i + n_points], line[:surplus], axis=0)
                    replaced_pts = np.append(replaced_pts, loop, axis=0)
                    points = np.vstack((loop[0:1], np.mean(loop, axis=0), loop[-1:]))
                    new_points = replace_method(points, n_points)
                    line[i: i + n_points] = new_points[:n_points - surplus]
                    line[:surplus] = new_points[:surplus]
                i += n_points
            else:
                i += 1
        else:
            i += 1
    
    return line, replaced_pts


def remove_all_loops(inner_line: np.ndarray, outer_line: np.ndarray, n_points: int) -> None:
    inner_line, replaced_inner = remove_loops(inner_line, n_points)
    outer_line, replaced_outer = remove_loops(outer_line, n_points)

    inner_line[-1] = inner_line[0]
    outer_line[-1] = outer_line[0]
    return replaced_inner, replaced_outer

def main():
    # Maximum bounds of track
    box_size = 100

    # Placeholder for positions of track nodes
    nodes = np.zeros((0, 2))

    # Distance between nodes. This should be lower than track_width.
    d = 3

    # Width of the racetrack
    track_width = 5

    pygame.init()
    screen_scale = 10
    d *= screen_scale
    screen = pygame.display.set_mode((box_size * screen_scale, box_size * screen_scale))
    screen.fill("white")
    pygame.display.update()
    clock = pygame.time.Clock()
    node_nr = 0
    max_d_theta = 6
    mouse_x_prev = 0
    mouse_y_prev = 0
    running = True


    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed() 
        if keys[pygame.K_a]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_x is None:
                mouse_x = mouse_x_prev
            else:
                mouse_x_prev = mouse_x

            if mouse_y is None:
                mouse_y = mouse_y_prev
            else:
                mouse_y_prev = mouse_y
                
            if mouse_x != 0.0:
                if nodes.size == 0:
                    nodes = np.append(nodes, np.array([[mouse_x, mouse_y]]), axis=0)
                else:
                    x_diff = mouse_x - nodes[node_nr, 0]
                    y_diff = mouse_y - nodes[node_nr, 1]
                    distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
                    if distance >= d:
                        
                        if nodes.shape[0] > 1:
                            v2 = np.array([x_diff, y_diff])
                            v1 = nodes[node_nr] - nodes[node_nr-1]
                            d_theta = angle_change(v1, v2)
                        else:
                            d_theta = 0.0

                        d_theta_abs = abs(d_theta)
                        if d_theta_abs < max_d_theta:
                            new_x, new_y = nodes[node_nr, 0] + x_diff/distance*d, nodes[node_nr, 1] + y_diff/distance*d
                        else:
                            old_angle = np.arctan2(v1[1], v1[0])
                            signed_d_theta_max = d_theta / d_theta_abs * max_d_theta
                            new_angle = signed_d_theta_max + old_angle
                            new_x = nodes[node_nr, 0] + d * np.cos(new_angle)
                            new_y = nodes[node_nr, 1] + d * np.sin(new_angle)

                        node_nr += 1
                        nodes = np.append(nodes, np.array([[new_x, new_y]]), axis=0)
                        screen.fill("white")
                        pygame.draw.aalines(screen, "black", False, nodes)

                        for x, y in nodes:
                            gfxdraw.filled_circle(screen, int(x), int(y), 5, (0, 0, 0))
                            gfxdraw.aacircle(screen, int(x), int(y), 5, (0, 0, 0))

                        pygame.display.update()
                        clock.tick(60)

                    x_diff = mouse_x - nodes[0, 0]
                    y_diff = mouse_y - nodes[0, 1]
                    d_squared = x_diff ** 2 + y_diff ** 2
                    if d_squared <= 3 * d ** 2 and np.sum((nodes[-1] - nodes[0])**2) <= 2 * d ** 2 and nodes.size > 5:
                        nodes = np.append(nodes, np.array([[mouse_x, mouse_y]]), axis=0)
                        nodes = np.append(nodes, np.array([[nodes[0, 0], nodes[0, 1]]]), axis=0)
                        break
    
    screen.fill("white")
    nodes_display = nodes.copy()
    nodes /= screen_scale
    inner_line, outer_line = add_borders(nodes, track_width)
    old_outer, old_inner = outer_line.copy(), inner_line.copy()

    pygame.draw.aalines(screen, "black", False, nodes_display)
    pygame.draw.aalines(screen, "black", False, outer_line * screen_scale)
    pygame.draw.aalines(screen, "black", False, inner_line * screen_scale)

    for x, y in nodes_display:
        gfxdraw.filled_circle(screen, int(x), int(y), 5, (0, 0, 0))
        gfxdraw.aacircle(screen, int(x), int(y), 5, (0, 0, 0))

    pygame.display.update()
    clock.tick(1)
    screen.fill("white")

    for i in range(4, 10):
        remove_all_loops(inner_line, outer_line, i)

    pygame.draw.aalines(screen, "black", False, nodes_display)
    pygame.draw.aalines(screen, "black", False, inner_line * screen_scale)
    pygame.draw.aalines(screen, "black", False, outer_line * screen_scale)

    for x, y in nodes_display:
        gfxdraw.filled_circle(screen, int(x), int(y), 5, (0, 0, 0))
        gfxdraw.aacircle(screen, int(x), int(y), 5, (0, 0, 0))

    pygame.display.update()

    running = True
    while running: 
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()

    save_track = input("Save track? ")
    if len(save_track) > 0:
        if save_track.lower() != "no" and save_track.lower() != "n":
            track = "tracks/demo_track"
            np.save(f"{track}.npy", nodes)
            np.save(f"{track}_inner_bound.npy", inner_line)
            np.save(f"{track}_outer_bound.npy", outer_line)

            old_track = f"{track}_old"
            np.save(f"{old_track}_inner_bound.npy", old_inner)
            np.save(f"{old_track}_outer_bound.npy", old_outer)

    save_track = input("Save track? ")
    if len(save_track) > 0:
        if save_track.lower() != "no" and save_track.lower() != "n":
            i = 0
            track = 'tracks/racetrack'
            track_name = f"{track}{i}.npy"
            while os.path.isfile(track_name):
                i += 1
                print("Name", track_name, "already taken.")
                track_name = f"{track}{i}.npy"
            np.save(track_name, nodes)
            np.save(f"{track}{i}_inner_bound.npy", inner_line)
            np.save(f"{track}{i}_outer_bound.npy", outer_line)
            print("Track saved as", track_name + ".")
        else:
            print("Track not saved.")
    else:
        print("Track not saved.")


if __name__ == '__main__':
    main()