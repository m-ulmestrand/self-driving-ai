from manim import *
import numpy as np


class DrawNet(Scene):
    def construct(self):
        n_neurons = np.array([7, 10, 8, 4], dtype='intc')

        neurons_x = np.linspace(-3, 3, len(n_neurons))
        neurons_y = [None for _ in n_neurons]
        max_neurons = max(n_neurons)
        y_scale = 6

        for i, n in enumerate(n_neurons):
            neurons_y[i] = np.arange(n) / max_neurons
            neurons_y[i] -= neurons_y[i].mean()
            neurons_y[i] *= y_scale
        
        circles = [[None for _ in range(n_neurons[i])] for i in range(len(n_neurons))]

        for i, (x, ys) in enumerate(zip(neurons_x, neurons_y)):
            for j, y in enumerate(ys):
                coord = np.array([x, y, 0])
                circles[i][j] = Circle(0.15, color=WHITE, fill_color=BLACK, fill_opacity=1, stroke_width=1.5).move_to(coord)
        
        self.play(*[FadeIn(circle) for circle in circles[0]], run_time=1)
        self.wait(2)

        in_features = [r"d_{-72}", r"d_{-36}", r"d_{0}", r"d_{36}", r"d_{72}", r"v", r"\theta_{\text{wheels}}"]
        in_features = [MathTex(feat, font_size=30) for feat in in_features]
        x = neurons_x[0] - 1.5
        x_start = x + 0.4

        for txt, y in zip(in_features, neurons_y[0]):
            txt: MathTex
            txt.move_to([x, y, 0])
            arrow = Arrow(start=[x_start, y, 0], end=[neurons_x[0], y, 0])
            self.play(Write(txt), Create(arrow))

        lines = [[None for _ in range(n_neurons[i] * n_neurons[i + 1])] for i in range(len(n_neurons) - 1)]

        for i, (x1, x2) in enumerate(zip(neurons_x[:-1], neurons_x[1:])):
            j = 0
            for y1 in neurons_y[i]:
                for y2 in neurons_y[i + 1]:
                    lines[i][j] = Line([x1, y1, 0], [x2, y2, 0], color=GREY, stroke_width=1)
                    lines[i][j].set_z_index(circles[0][0].z_index - 1)
                    j += 1

        for circle_collection, line_collection in zip(circles[1:], lines):
            self.play(
                *[FadeIn(line) for line in line_collection],
                *[FadeIn(circle) for circle in circle_collection],
                run_time=0.5
            )

        out_features = [r"+\Delta \theta", r"-\Delta \theta", r"+v", r"-v"]
        out_features = [MathTex(feat, font_size=30) for feat in out_features]
        x = neurons_x[-1] + 1.5
        x_end = x - 0.2

        for txt, y in zip(out_features, neurons_y[-1]):
            txt: MathTex
            txt.move_to([x, y, 0])
            arrow = Arrow(start=[neurons_x[-1], y, 0], end=[x_end, y, 0])
            self.play(Write(txt), Create(arrow))
        
        self.wait(2)
