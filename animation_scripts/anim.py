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
        
        """start = np.zeros(3)
        end = np.zeros(3)
        end[:2] = nodes[5, :]
        arr = Arrow(start_point=start, end_point=end)
        self.play(Create(arr))"""
        
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
        
        # New track, removed loops
        inner = np.load(f"./tracks/{track}_inner_bound.npy")
        outer = np.load(f"./tracks/{track}_outer_bound.npy")
        outer_plot_new = ax.plot_line_graph(outer[:, 0], outer[:, 1], line_color=BLUE_E, add_vertex_dots=False)
        inner_plot_new = ax.plot_line_graph(inner[:, 0], inner[:, 1], line_color=BLUE_E, add_vertex_dots=False)

        self.play(Transform(outer_plot, outer_plot_new), Transform(inner_plot, inner_plot_new))
        self.wait(2)


"""def construct(self):
    vertices1 = range(50)
    vertices2 = range(50)
    edges = [(48, 49), (3, 4)]
    g1 = Graph(vertices1, edges, layout="spiral")
    g2 = Graph(vertices2, edges, layout="circular")

    # self.add(graph)
    self.play(Create(g1))
    self.wait(5)
    self.play(*[g1[i].animate.move_to(g2[i]) for i in vertices1])
    self.wait()"""