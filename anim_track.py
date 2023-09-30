from manim import *
import numpy as np


class DrawTrack(Scene):
    def construct(self):
        # Old track with loops
        track = "demo_track"
        inner = np.load(f"./tracks/{track}_old_inner_bound.npy")
        outer = np.load(f"./tracks/{track}_old_outer_bound.npy")
        nodes = np.load(f"./tracks/{track}.npy")
        
        ax = Axes(x_range=(-50, 150), y_range=(0, 100))
        track_plot = ax.plot_line_graph(nodes[:, 0], nodes[:, 1], line_color=BLUE_C, vertex_dot_radius=0.025)
        self.play(Create(track_plot["vertex_dots"]), run_time=2)
        # self.play(Create(track_plot["line_graph"]), run_time=2)
        
        outer_plot = ax.plot_line_graph(outer[:, 0], outer[:, 1], line_color=BLUE_E, add_vertex_dots=False)
        inner_plot = ax.plot_line_graph(inner[:, 0], inner[:, 1], line_color=BLUE_E, add_vertex_dots=False)

        plots = [None for _ in range(nodes.shape[0] * 2)]
        for i in range(nodes.shape[0]):
            plots[i] = ax.plot_line_graph([nodes[i, 0], inner[i, 0]], [nodes[i, 1], inner[i, 1]], line_color=BLUE_E, add_vertex_dots=False)
        
        for i, plot_num in enumerate(range(nodes.shape[0], 2 * nodes.shape[0])):
            plots[plot_num] = ax.plot_line_graph([nodes[i, 0], outer[i, 0]], [nodes[i, 1], outer[i, 1]], line_color=BLUE_E, add_vertex_dots=False)
        
        self.play(*[Create(l) for l in plots], run_time=1.5)
        self.wait(1)
        self.play(Create(outer_plot), Create(inner_plot), run_time=2)
        self.play(*[FadeOut(l) for l in plots], run_time=1)
        self.wait(1)
        inner_circles = self.draw_circles(inner[[25, 32, 40], :], ax)
        outer_pos = outer[[28, 36, 45], :]
        outer_pos[:, 0] -= 1
        outer_circles = self.draw_circles(outer_pos, ax)
        self.wait(3)
        
        # New track, removed loops
        inner = np.load(f"./tracks/{track}_inner_bound.npy")
        outer = np.load(f"./tracks/{track}_outer_bound.npy")
        outer_plot_new = ax.plot_line_graph(outer[:, 0], outer[:, 1], line_color=BLUE_E, add_vertex_dots=False)
        inner_plot_new = ax.plot_line_graph(inner[:, 0], inner[:, 1], line_color=BLUE_E, add_vertex_dots=False)

        self.play(Transform(outer_plot, outer_plot_new), Transform(inner_plot, inner_plot_new))
        self.play(FadeOut(*inner_circles, *outer_circles))
        self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects], run_time=1
        )

    def draw_circles(self, positions: np.ndarray, ax: Axes, r: float = 0.15, run_time: float = 0.75):
        circles = [None] * positions.shape[0]
        for i, pos in enumerate(positions):
            circle = Circle(r).move_to(ax.c2p(*pos))
            circles[i] = circle
            self.play(Create(circle), run_time=run_time)
        
        return circles
