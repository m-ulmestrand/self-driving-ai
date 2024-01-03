from manim import *
import numpy as np
import math


class DrawNet2(Scene):
    def construct(self):
        radius = 1
        neuron_pos = np.zeros(3)
        x_start = -4
        edge_start = np.array([x_start, 0, 0])
        x_end = -0.75
        x_text_start = -4.5
        weight_x = -2.5
        edge_end = np.array([x_end, 0, 0])
        font_size = DEFAULT_FONT_SIZE / 2

        neuron = Circle(
            radius, color=WHITE, fill_color=BLACK, fill_opacity=1, stroke_width=2
        ).move_to(neuron_pos)
        edge = Arrow(edge_start, edge_end, color=WHITE, stroke_width=2, tip_length=0.2)
        edge.set_z_index(neuron.z_index - 1)

        self.play(FadeIn(neuron))
        self.play(Create(edge))
        self.wait(1)

        input_text_pos = np.array([x_text_start, 0, 0])
        input_text = MathTex("x").move_to(input_text_pos)
        weight_text_pos = np.array([weight_x, 0.5, 0])
        weight_text = MathTex("w").move_to(weight_text_pos)

        self.play(Write(input_text))
        self.play(Write(weight_text))
        self.wait(2)

        bias_start = np.array([-3.5, 2, 0])
        bias_direction = bias_start.copy()
        bias_direction /= np.linalg.norm(bias_direction)
        bias_edge = Arrow(bias_start, 0.75 * bias_direction, color=WHITE, stroke_width=2, tip_length=0.2)
        bias_text = MathTex("b").move_to(np.array([-3.8, 2, 0]))
        self.play(FadeIn(bias_edge, bias_text))
        self.wait(2)
        self.play(FadeOut(bias_edge, bias_text))

        out_edge_x = 4
        out_edge_start = np.array([0.75, 0, 0])
        out_edge_end = np.array([out_edge_x, 0, 0])
        out_edge = Arrow(out_edge_start, out_edge_end, color=WHITE, stroke_width=2, tip_length=0.2)
        out_text_pos = np.array([5, 0, 0])
        out_text_list = [
            r"\sigma{\left(",
            r'w',
            r'',
            r'x',
            r'+',
            r'b',
            r'\right)}'
        ]
        out_text = MathTex(*out_text_list).move_to(out_text_pos)

        self.play(Create(out_edge))
        self.play(Write(out_text))
        self.wait(2)

        sigma_pos = np.array([0, -2, 0])
        sigma_text = MathTex(r"\sigma \text{ nonlinear function}").move_to(sigma_pos)
        self.play(Write(sigma_text))

        nonlinear_pos = np.array([0, -2.6, 0])
        nonlinear_text = MathTex(
            r"\text{e.g. } \sigma(wx + b) = \text{ReLU}(wx + b) = \max(0, wx + b)", font_size=0.75 * DEFAULT_FONT_SIZE
        ).move_to(nonlinear_pos)
        self.play(Write(nonlinear_text))
        self.wait(3)

        self.play(FadeOut(nonlinear_text, sigma_text))
        self.wait(3)

        vector_out_text_list = out_text_list.copy()
        vector_out_text_list[1] = r"\mathbf{w}"
        vector_out_text_list[2] = r"\cdot"
        vector_out_text_list[3] = r"\mathbf{x}"
        vector_out_text = MathTex(*vector_out_text_list).move_to(out_text_pos).align_to(out_text, LEFT)
        n_edges = 5
        edges = [None for _ in range(n_edges)]
        input_texts = edges.copy()
        w_texts = edges.copy()
        w_squares = edges.copy()
        w_len = 2.5
        y_inputs = np.arange(2, -3, -1)
        transformation_text_pos = np.array([4, 3, 0])
        transformation_text_list = [
            r"\mathbf{w}",
            r"\cdot",
            r"\mathbf{x}",
            '+',
            'b',
            '=',
            r"\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\ w_5 \end{bmatrix}",
            r"\cdot",
            r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \end{bmatrix}",
            '+',
            'b'
        ]
        transformation_text = MathTex(
            *transformation_text_list, font_size=font_size
        ).move_to(transformation_text_pos)

        for i, y in enumerate(y_inputs):
            square = Rectangle(color=GRAY_B, fill_color=BLACK, fill_opacity=1)
            direction = np.array([x_end, 0, 0]) - np.array([x_start, y, 0])
            direction /= np.linalg.norm(direction)
            edges[i] = Arrow(
                np.array([x_start, y, 0]), 
                x_end * direction,
                color=WHITE, 
                stroke_width=2,
                tip_length=0.2
            )
            w_x = -w_len * direction[0] - 0.4
            w_y = -w_len * direction[1]
            input_texts[i] = MathTex(f"x_{i + 1}").move_to(np.array([x_text_start, y, 0]))
            w_texts[i] = MathTex(f"w_{i + 1}", color=WHITE).move_to(np.array([w_x, w_y, 0]))
            square.surround(w_texts[i])
            w_squares[i] = square
            w_texts[i].set_z_index(edges[i].z_index + 2)
            w_squares[i].set_z_index(edges[i].z_index + 1)
        
        edges[2] = edge
        edges_group = VGroup(*[e for i, e in enumerate(edges) if i != 2])
        inputs_group = VGroup(*[x for i, x in enumerate(input_texts) if i != 2])
        w_group = VGroup(*w_texts)
        weight_text.set_z_index(edges[2].z_index + 1)
        squares_group = VGroup(*w_squares)
        self.play(
            ReplacementTransform(input_text, input_texts[2]),
            TransformMatchingTex(out_text, vector_out_text),
            Create(transformation_text),
            Create(squares_group),
            FadeOut(weight_text),
            Create(edges_group),
            Write(inputs_group),
            Write(w_group)
        )

        r = 0.5
        y_neurons = np.linspace(2, -2, 4)
        neurons = [
            Circle(
                r, 
                color=WHITE, 
                fill_color=BLACK,
                fill_opacity=1, 
                stroke_width=2
            ).move_to(np.array([0, y, 0])) 
            for y in y_neurons
        ]
        neurons_group = VGroup(*neurons)
        edges = np.zeros((5, 4), dtype=Arrow)
        x_end = -0.3
        transformation_text_list[0] = r"\mathbf{W}"
        transformation_text_list[1] = ''
        transformation_text_list[4] = r"\mathbf{b}"
        transformation_text_list[7] = ''
        transformation_text_list[6] = r"\begin{bmatrix} "
        transformation_text_list[-1] = r"\begin{bmatrix} b_1 \\ b_2 \\ b_3 \\ b_4 \end{bmatrix}"

        for i, y1 in zip(range(edges.shape[0]), y_inputs):
            for j, y2 in zip(range(edges.shape[1]), y_neurons):
                transformation_text_list[6] += r"w_{"
                transformation_text_list[6] += f"{i + 1}{j + 1}"
                transformation_text_list[6] += r"} &"
                direction = np.array([x_end, y2, 0]) - np.array([x_start, y1, 0])
                direction /= np.linalg.norm(direction)
                edges[i, j] = Arrow(
                    np.array([x_start, y1, 0]), 
                    x_end * direction + np.array([0, y2, 0]),
                    color=WHITE, 
                    stroke_width=2,
                    tip_length=0.2
                )
            transformation_text_list[6] = transformation_text_list[6][:-2] + r"\\"
        transformation_text_list[6] = transformation_text_list[6][:-2] + r"\end{bmatrix}"

        out_edges = np.zeros(edges.shape[1], dtype=Arrow)
        for i, y in zip(range(edges.shape[1]), y_neurons):
            out_edges[i] = Arrow(
                np.array([0, y, 0]) - x_end * direction,
                np.array([out_edge_x, y, 0]),
                color=WHITE, 
                stroke_width=2,
                tip_length=0.2,
            )

        input_neurons = np.zeros((edges.shape[0]), dtype=Circle)
        for i, y in zip(range(edges.shape[0]), y_inputs):
            input_neurons[i] = Circle(
                r, color=WHITE, fill_color=BLACK, fill_opacity=1, stroke_width=2
            ).move_to(np.array([x_start, y, 0]))
            input_neurons[i].set_z_index(edges[0, 0].z_index + 1)

        input_texts[2].set_z_index(edges[0, 0].z_index + 1)
        edges_group2 = VGroup(*edges.flatten())
        out_edge_group = VGroup(*out_edges)
        out_edge_group.set_z_index(neurons_group.z_index - 10)
        matrix_out_text_list = out_text_list.copy()
        matrix_out_text_list[1] = r"\mathbf{W}"
        matrix_out_text_list[2] = ''
        matrix_out_text_list[3] = r"\mathbf{x}"
        matrix_out_text_list[5] = r"\mathbf{b}"
        matrix_out_text = MathTex(*matrix_out_text_list).move_to(out_text_pos).align_to(out_text, LEFT)
        transformation_text_matrix = MathTex(
            *transformation_text_list, font_size=font_size
        ).move_to(transformation_text_pos)
        self.wait(3)

        self.play(*[ReplacementTransform(x_t, inp) for x_t, inp in zip(input_texts, input_neurons)])
        self.wait(3)
        self.play(
            ReplacementTransform(neuron, neurons_group),
            ReplacementTransform(edges_group, edges_group2),
            ReplacementTransform(out_edge, out_edge_group),
            FadeOut(edge, *squares_group, w_group),
            TransformMatchingTex(vector_out_text, matrix_out_text),
            TransformMatchingTex(transformation_text, transformation_text_matrix)
        )
        self.wait(5)
        self.play(FadeOut(transformation_text_matrix))

        start_subtract = 1
        end_subtract = 2.5
        x_start -= start_subtract
        buff = 0.
        x_end -= end_subtract

        for i, y1 in zip(range(edges.shape[0]), y_inputs):
            for j, y2 in zip(range(edges.shape[1]), y_neurons):
                direction = np.array([x_end, y2, 0]) - np.array([x_start, y1, 0])
                direction /= np.linalg.norm(direction)
                edges[i, j] = Arrow(
                    np.array([x_start, y1, 0]), 
                    -buff * direction + np.array([x_end, y2, 0]),
                    color=WHITE, 
                    stroke_width=2,
                    tip_length=0.2
                )
        
        new_neurons_group = neurons_group.copy()
        new_pos = neurons_group.get_center()
        new_pos[0] -= end_subtract
        new_neurons_group.move_to(new_pos)
    
        for i in range(len(input_neurons)):
            input_neurons[i].generate_target()
            input_neurons[i].target.shift(np.array([-start_subtract, 0, 0]))

        edges_group3 = VGroup(*edges.flatten())

        y_outputs = np.array([-1, 1])
        x_output = 0
        output_neurons = np.zeros(2, dtype=Circle)

        for i in range(len(output_neurons)):
            output_neurons[i] = Circle(
                r, color=WHITE, fill_color=BLACK, fill_opacity=1, stroke_width=2
            ).move_to(np.array([x_output, y_outputs[i], 0]))

        buff = 0.3
        output_edges = np.zeros([len(neurons), len(output_neurons)], dtype=Arrow)
        for i, y1 in enumerate(y_neurons):
            for j, y2 in enumerate(y_outputs):
                direction = np.array([x_output, y2, 0]) - np.array([x_end, y1, 0])
                direction /= np.linalg.norm(direction)
                output_edges[i, j] = Arrow(
                    np.array([x_end, y1, 0]), 
                    -buff * direction + np.array([x_output, y2, 0]),
                    color=WHITE, 
                    stroke_width=2,
                    tip_length=0.2
                )
                output_edges[i, j].set_z_index(new_neurons_group[0].z_index - 1)

        output_edges_group = VGroup(*output_edges.flatten())
        nn_out_text_list = matrix_out_text_list.copy()
        nn_out_text_list[0] = r"\mathbf{y} = \sigma{\left(\mathbf{W_2}\sigma{\left("
        nn_out_text_list[1] = r"\mathbf{W_1}"
        nn_out_text_list[5] = r"\mathbf{b_1}"
        nn_out_text_list[6] = r"\right)} + \mathbf{b_2}\right)}"
        nn_out_text_pos = out_text_pos.copy()
        nn_out_text_pos[0] -= 5
        nn_out_text_pos[1] -= 2.5
        nn_out_text = MathTex(*nn_out_text_list).move_to(nn_out_text_pos).align_to(nn_out_text_pos, LEFT)
        self.play(
            ReplacementTransform(out_edge_group, output_edges_group),
            ReplacementTransform(edges_group2, edges_group3),
            ReplacementTransform(neurons_group, new_neurons_group),
            TransformMatchingTex(matrix_out_text, nn_out_text),
            *[MoveToTarget(n) for n in input_neurons],
            FadeIn(*output_neurons)
        )
        out_values_lines = [
            Arrow(
                np.array([0.25, y_outputs[1], 0]), 
                np.array([2., y_outputs[1], 0]), 
                color=WHITE, 
                stroke_width=2, 
                tip_length=0.2
            ),
            Arrow(
                np.array([0.25, y_outputs[0], 0]), 
                np.array([2., y_outputs[0], 0]), 
                color=WHITE, 
                stroke_width=2, 
                tip_length=0.2
            )
        ]
        out_value_texts = [
            MathTex("y_1", font_size=DEFAULT_FONT_SIZE).move_to([2.3, y_outputs[1], 0]),
            MathTex("y_2", font_size=DEFAULT_FONT_SIZE).move_to([2.3, y_outputs[0], 0])
        ]
        self.play(Create(out_values_lines[0]), Write(out_value_texts[0]))
        self.play(Create(out_values_lines[1]), Write(out_value_texts[1]))
        self.wait(5)

        loss_text_pos = nn_out_text_pos.copy()
        loss_text_pos[1] -= 1
        loss_text = MathTex(
            r"L_\text{MSE}(\mathbf{Y}, \mathbf{Y}_\text{true}) = \frac{1}{n}\sum_{i=1}^n\left(\mathbf{y}^{(i)} - \mathbf{y}_\text{true}^{(i)}\right)^2",
            font_size=DEFAULT_FONT_SIZE * 0.75
        ).move_to(loss_text_pos).align_to(loss_text_pos, LEFT)
        self.play(Write(loss_text))
        self.wait(5)