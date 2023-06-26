from manim import *
import numpy as np
from typing import Tuple


class DrawRewardsCase(Scene):
    def construct(self):
        reward_pts = np.load("passing_times.npy")
        ax = Axes(x_range=(0, max(reward_pts) + 10, 90), y_range=(-1.5, 1.5))
        ax.add_coordinates()
        x_label = ax.get_x_axis_label("t")
        y_label = ax.get_y_axis_label("r(t)")
        x, y = self.get_reward(reward_pts)
        where_rewards = np.logical_and(y != 0, y > 0)

        vertex_graph2 = ax.plot_line_graph(
            x[where_rewards], 
            y[where_rewards],
            vertex_dot_style=dict(stroke_width=3, fill_color=BLUE_E),
        )

        line_graph = ax.plot_line_graph(x, y, add_vertex_dots=False, line_color=WHITE)
        line_graph.set_z_index(vertex_graph2.z_index - 1)
        ax.set_z_index(vertex_graph2.z_index - 2)

        self.play(Create(ax), Write(x_label), Write(y_label))
        self.wait(0.5)
        self.play(Write(line_graph), run_time=23)


    @staticmethod
    def fill_rewards(
            x: np.ndarray,
            rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        x_curve = np.zeros(len(x) + 1)
        x_curve[1:] = x
        y_curve = np.copy(x_curve)
        y_curve[1:] = rewards
        
        return x_curve, y_curve
    
    def get_reward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_shifted = x[1:]
        diff = np.copy(x)
        diff[1:] = x_shifted - x[:-1]
        rewards = 1 / diff
        rewards[-1] = -1

        return self.fill_rewards(x, rewards)