from manim import *
import numpy as np


class DrawNet(Scene):
    def construct(self):
        n_neurons = np.array([7, 10, 8, 4], dtype='intc')

        start_neuron_x = -3
        end_neuron_x = 3
        neurons_x = np.linspace(start_neuron_x, end_neuron_x, len(n_neurons))
        neurons_x[0] = np.mean(neurons_x[[0, 1]])
        neurons_x[-1] = np.mean(neurons_x[[-1, -2]])
        neurons_y = [None for _ in n_neurons]
        max_neurons = max(n_neurons)
        y_scale = 6

        for i, n in enumerate(n_neurons):
            neurons_y[i] = np.arange(n) / max_neurons
            neurons_y[i] -= neurons_y[i].mean()
            neurons_y[i] *= y_scale
        
        neurons = np.array([[None for _ in range(n_neurons[i])] for i in range(len(n_neurons))], dtype=object)

        for i, (x, ys) in enumerate(zip(neurons_x, neurons_y)):
            for j, y in enumerate(ys):
                coord = np.array([x, y, 0])
                neurons[i][j] = Circle(0.15, color=WHITE, fill_color=BLACK, fill_opacity=1, stroke_width=1.5).move_to(coord)
        
        self.play(*[FadeIn(neuron) for neuron in neurons[0]], run_time=1)
        self.wait(2)

        in_features = [r"d_{-72}", r"d_{-36}", r"d_{0}", r"d_{36}", r"d_{72}", r"v", r"\theta_{\text{wheels}}"]
        in_features = [MathTex(feat, font_size=30) for feat in in_features]
        x = neurons_x[0] - 1.5
        x_start = x + 0.4
        in_arrows = [None] * len(neurons[0])

        for i, (txt, y) in enumerate(zip(in_features[:-2], neurons_y[0][:-2])):
            txt: MathTex
            txt.move_to([x, y, 0])
            in_arrows[i] = Arrow(start=[x_start, y, 0], end=[neurons_x[0], y, 0])
            self.play(Write(txt), Create(in_arrows[i]), run_time=0.75)
        self.wait(2)

        brace_x = x - 0.25
        d_brace = BraceBetweenPoints([brace_x, neurons_y[0][0], 0], [brace_x, neurons_y[0][4], 0], LEFT)
        brace_text = Paragraph(
            "Distances to", 
            "\ntrack edges", 
            font_size=15, 
            alignment="center",
            line_spacing=0
        ).next_to(d_brace, LEFT)
        # brace_text = Text("Distances\n to track edges", font_size=15).next_to(d_brace, LEFT)
        self.play(Create(d_brace), Write(brace_text))
        self.wait(1)
        self.play(FadeOut(d_brace), FadeOut(brace_text))

        for i, (txt, y) in enumerate(zip(in_features[-2:], neurons_y[0][-2:]), start=len(in_features[:-2])):
            txt: MathTex
            txt.move_to([x, y, 0])
            in_arrows[i] = Arrow(start=[x_start, y, 0], end=[neurons_x[0], y, 0])
            self.play(Write(txt), Create(in_arrows[i]), run_time=0.75)
        self.wait(2)

        self.play(*[FadeIn(neuron) for neuron in neurons[-1]], run_time=1)
        out_features = [r"+\Delta \theta_\text{wheels}", r"-\Delta \theta_\text{wheels}", r"+\Delta v", r"-\Delta v"]
        out_features = [MathTex(feat, font_size=30) for feat in out_features]
        x = neurons_x[-1] + 1.5
        x_end = x - 0.5
        out_arrows = [None] * len(neurons[-1])

        for i, (txt, y) in enumerate(zip(out_features, neurons_y[-1])):
            txt: MathTex
            txt.move_to([x, y, 0])
            out_arrows[i] = Arrow(start=[neurons_x[-1], y, 0], end=[x_end, y, 0])
            self.play(Write(txt), Create(out_arrows[i]), run_time=0.75)
        
        self.wait(2)
        lines_first = [None] * (n_neurons[0] * n_neurons[-1])

        x1, x2 = neurons_x[[0, -1]]
        j = 0
        for y1 in neurons_y[0]:
            for y2 in neurons_y[-1]:
                lines_first[j] = Line([x1, y1, 0], [x2, y2, 0], color=GREY, stroke_width=1)
                lines_first[j].set_z_index(neurons[0][0].z_index - 1)
                j += 1

        """
        # This doesn't work as expected with z_index...
        line_anims = Succession(*[Create(line) for line in lines_first])
        line_group = Group(*lines_first)
        line_group.z_index = neurons[0][0].z_index - 1
        self.play(line_anims, run_time=3)
        """
        line_group = Group(*lines_first)
        line_group.z_index = neurons[0][0].z_index - 1
        self.play(FadeIn(line_group), run_time=0.5)
        self.wait(2)

        shift_first = [start_neuron_x - neurons_x[0], 0, 0]
        shift_last = [end_neuron_x - neurons_x[-1], 0, 0]
        neurons_x = np.linspace(start_neuron_x, end_neuron_x, len(n_neurons))

        lines = [[None for _ in range(n_neurons[i] * n_neurons[i + 1])] for i in range(len(n_neurons) - 1)]

        for i, (x1, x2) in enumerate(zip(neurons_x[:-1], neurons_x[1:])):
            j = 0
            for y1 in neurons_y[i]:
                for y2 in neurons_y[i + 1]:
                    lines[i][j] = Line([x1, y1, 0], [x2, y2, 0], color=GREY, stroke_width=1)
                    lines[i][j].set_z_index(neurons[0][0].z_index - 1)
                    j += 1

        flattened_lines = Group(*[item for sublist in lines for item in sublist])
        in_neurons = neurons[0]
        out_neurons = neurons[-1]
        neurons = neurons[1:-1]
        self.play(
            *[neur.animate.shift(shift_first) for neur in in_neurons],
            *[feat.animate.shift(shift_first) for feat in in_features],
            *[arr.animate.shift(shift_first) for arr in in_arrows],
            *[neur.animate.shift(shift_last) for neur in out_neurons],
            *[feat.animate.shift(shift_last) for feat in out_features],
            *[arr.animate.shift(shift_last) for arr in out_arrows],
            Transform(line_group, flattened_lines),
            *[FadeIn(neuron) for neuron_collection in neurons for neuron in neuron_collection]
        )

        self.wait(2)
