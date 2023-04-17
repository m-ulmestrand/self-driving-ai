from manim import *
import numpy as np
from typing import Tuple


class DrawRewards(Scene):
    def construct(self):
        ax = Axes(x_range=(0, 20), y_range=(-2, 2))
        reward_pts = np.array([1, 3, 6, 7, 9, 14, 18])
        x, y = self.get_reward(reward_pts, self.fill_rewards1)
        where_rewards = np.logical_and(y != 0, y > 0)

        vertex_graph1 = ax.plot_line_graph(
            x[where_rewards], 
            np.zeros(len(y[where_rewards])),
            vertex_dot_style=dict(stroke_width=3, fill_color=BLUE_E),
        )
        vertex_graph2 = ax.plot_line_graph(
            x[where_rewards], 
            y[where_rewards],
            vertex_dot_style=dict(stroke_width=3, fill_color=BLUE_E),
        )

        dot1 = Dot(color=RED, stroke_width=1)
        dot2 = Dot(color=RED, stroke_width=1)
        dot_outline1 = Dot(color=WHITE, stroke_width=4)
        dot_outline2 = Dot(color=WHITE, stroke_width=4)

        dot1.move_to(ax.c2p(x[-2], 0))
        dot2.move_to(ax.c2p(x[-2], y[-2]))
        dot_outline1.move_to(ax.c2p(x[-2], 0))
        dot_outline2.move_to(ax.c2p(x[-2], y[-2]))
        circle1 = VGroup(dot_outline1, dot1)
        circle2 = VGroup(dot_outline2, dot2)

        line_graph = ax.plot_line_graph(x, y, add_vertex_dots=False, line_color=WHITE)
        x, y = self.get_reward(reward_pts, self.fill_rewards2)
        line_graph2 = ax.plot_line_graph(x, y, add_vertex_dots=False, line_color=WHITE)
        x, y = self.get_reward(reward_pts, self.fill_rewards3)
        line_graph3 = ax.plot_line_graph(x, y, add_vertex_dots=False, line_color=WHITE)
        line_graph.set_z_index(vertex_graph2.z_index - 1)
        line_graph2.set_z_index(vertex_graph2.z_index - 1)
        line_graph3.set_z_index(vertex_graph2.z_index - 1)
        ax.set_z_index(vertex_graph2.z_index - 2)

        p1 = ax.c2p(reward_pts[0], 0, 0)
        p2 = ax.c2p(reward_pts[1], 0, 0)
        p3 = ax.c2p(reward_pts[0], 2 / (reward_pts[1] - reward_pts[0]), 0)
        p4 = ax.c2p(reward_pts[1], 1.5 / (reward_pts[2] - reward_pts[1]), 0)

        brace1 = BraceBetweenPoints(ax.c2p(0, 0, 0), p1)
        brace2 = BraceBetweenPoints(p1, p2)
        brace3 = BraceBetweenPoints(ax.c2p(reward_pts[0], 0, 0), p3)
        brace4 = BraceBetweenPoints(ax.c2p(reward_pts[1], 0, 0), p4)

        d1 = int(reward_pts[0] - 0)
        d2 = int(reward_pts[1] - reward_pts[0])

        t1 = Text(f"{d1}",).scale(0.5).next_to(brace1, 0.5 * DOWN)
        t2 = Text(f"{d2}",).scale(0.5).next_to(brace2, 0.5 * DOWN)
        t3 = Text(f"1/{d1}").scale(0.5).next_to(brace3, 0.5 * RIGHT)
        t4 = Text(f"1/{d2}").scale(0.5).next_to(brace4, 0.5 * RIGHT)

        self.play(Create(ax))
        self.play(Create(vertex_graph1["vertex_dots"]))
        self.add(circle1)
        self.play(Transform(vertex_graph1["vertex_dots"], vertex_graph2["vertex_dots"]))
        self.play(Transform(circle1, circle2))
        self.wait(2)
        self.play(Create(brace1), Create(t1))
        self.play(Create(brace3), Create(t3))
        self.wait(1)
        self.play(FadeOut(brace1, brace3, t1, t3))
        self.play(Create(brace2), Create(t2))
        self.play(Create(brace4), Create(t4))
        self.wait(1)
        self.play(FadeOut(brace2, brace4, t2, t4))
        self.play(Create(line_graph), run_time=2)
        self.wait(2)
        self.play(ReplacementTransform(line_graph, line_graph2))
        self.wait(2)
        self.play(ReplacementTransform(line_graph2, line_graph3))
        self.wait(2)

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
        
        return x_curve, y_curve
    
    @staticmethod
    def fill_rewards3(
            x: np.ndarray,
            rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        x_curve = np.zeros(len(x) + 1)
        x_curve[1:] = x
        y_curve = np.copy(x_curve)
        y_curve[1:] = rewards
        
        return x_curve, y_curve
    
    def get_reward(self, x: np.ndarray, fill_method) -> Tuple[np.ndarray, np.ndarray]:
        x_shifted = x[1:]
        diff = np.copy(x)
        diff[1:] = x_shifted - x[:-1]
        rewards = 1 / diff
        rewards[-1] = -1

        return fill_method(x, rewards)