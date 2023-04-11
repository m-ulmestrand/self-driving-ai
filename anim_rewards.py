from manim import *
import numpy as np
from typing import Tuple


class DrawRewards(Scene):
    def construct(self):
        ax = Axes(x_range=(0, 20), y_range=(-2, 2))
        reward_pts = np.array([1, 3, 6, 7, 9, 14, 18])
        x, y = self.get_reward(reward_pts, self.fill_rewards1)
        line_graph = ax.plot_line_graph(x, y, add_vertex_dots=False, line_color=BLUE_D)
        where_rewards = y != 0
        vertex_graph = ax.plot_line_graph(
            x[where_rewards], 
            y[where_rewards],
            vertex_dot_style=dict(stroke_width=3,  fill_color=BLUE_E),
        )
        dot = Dot(color=RED, stroke_width=3)
        dot.move_to(ax.c2p(x[-2], y[-2]))

        x, y = self.get_reward(reward_pts, self.fill_rewards2)
        line_graph2 = ax.plot_line_graph(x, y, add_vertex_dots=False, line_color=BLUE_D)
        line_graph.set_z_index(vertex_graph.z_index - 1)
        line_graph2.set_z_index(vertex_graph.z_index - 1)
        ax.set_z_index(vertex_graph.z_index - 2)

        self.play(Create(ax))
        self.play(Create(vertex_graph["vertex_dots"]))
        self.add(dot)
        self.wait(2)
        self.play(Create(line_graph), run_time=2)
        self.wait(2)
        self.play(Transform(line_graph, line_graph2))
        self.wait(2)

    @staticmethod
    def fill_rewards2(
            x: np.ndarray,
            rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        x_curve = np.zeros(2 * len(x) + 1)
        y_curve = np.copy(x_curve)

        i = 1
        for j in range(len(rewards)):
            x_curve[i: i + 2] = x[j]
            y_curve[i] = rewards[j]
            i += 2
        
        print(x_curve, y_curve)
        return x_curve, y_curve
    
    @staticmethod
    def fill_rewards1(
            x: np.ndarray,
            rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        x_curve = np.zeros(3 * len(x) + 1)
        y_curve = np.copy(x_curve)

        i = 1
        for j in range(len(rewards)):
            x_curve[i: i + 3] = x[j]
            y_curve[i + 1] = rewards[j]
            i += 3
        
        return x_curve, y_curve
    
    def get_reward(self, x: np.ndarray, fill_method) -> Tuple[np.ndarray, np.ndarray]:
        x_shifted = x[1:]
        diff = np.copy(x)
        diff[1:] = x_shifted - x[:-1]
        rewards = 1 / diff
        rewards[-1] = -1

        return fill_method(x, rewards)