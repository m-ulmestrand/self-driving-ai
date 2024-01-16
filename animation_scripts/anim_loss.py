from manim import *
import numpy as np
from typing import List, Tuple
from matplotlib import colormaps


class DrawLoss(Scene):
    def construct(self):
        x_range = (-2.5, 2.5)
        y_range = (-2.5, 2.5)
        length = x_range[1] - x_range[0]
        position1 = np.array([-3, 0, 0])
        position2 = np.array([3, 0, 0])

        n_data = 80
        slope = 0.4
        noise_scale = 0.25
        self.x_data = np.linspace(*x_range, n_data)

        seed = 42
        np.random.seed(seed)
        self.y_data = slope * self.x_data + noise_scale * np.random.randn(n_data)

        loss_text_list = [
            r"",
            r"",
            r"L_\text{MSE}\left(\mathbf{Y}_\text{true}, \mathbf{Y} \right)",
            r"=",
            r"\frac{1}{n}",
            r"\sum_{i=1}^n",
            r"",
            r"\left(",
            r"\mathbf{y}_\text{true}^{(i)} -",
            r"\mathbf{y}^{(i)}",
            r"\right)",
            r"^2"
        ]

        loss_text_position = np.zeros(3)
        header_position = loss_text_position + 1.5 * UP
        loss_header = Tex("Loss function").move_to(header_position)
        font_size = DEFAULT_FONT_SIZE
        loss_text = MathTex(*loss_text_list, font_size=font_size).move_to(loss_text_position)
        self.add(loss_text)
        self.play(Write(loss_header))
        self.wait(8)
        
        loss_text_list_nn = loss_text_list.copy()
        loss_text_list_nn[2] = r"L_\text{MSE}\left(\mathbf{Y}_\text{true}, w\right)"
        loss_text_list_nn[9] = r"wx^{(i)}"
        loss_text_list_gradient = loss_text_list_nn.copy()
        loss_text_nn = MathTex(*loss_text_list_nn, font_size=font_size).move_to(loss_text_position)
        loss_text_nn.set_color_by_tex(r"wx^{(i)}", RED)
        self.play(ReplacementTransform(loss_text, loss_text_nn))
        self.wait(3)

        gradient_header = Tex("Gradient of loss function").move_to(header_position)
        loss_text_list_gradient[1] = r"\frac{ \partial }{ \partial w } "
        loss_text_list_gradient[4]  = r"\frac{1}{n}"
        loss_text_list_gradient[6] = r"-2x^{(i)}"
        loss_text_list_gradient[-1] = r" "
        loss_text_gradient = MathTex(*loss_text_list_gradient, font_size=font_size).move_to(loss_text_position)
        self.play(TransformMatchingTex(loss_text_nn, loss_text_gradient), ReplacementTransform(loss_header, gradient_header))
        self.wait(7)

        gradient_descent_header = Tex("Gradient descent").move_to(header_position)
        gradient_descent_text_list = loss_text_list_gradient.copy()
        gradient_descent_text_list[0] = r"w \leftarrow w -\eta"
        gradient_descent_text_list[1] = ""
        gradient_descent_text_list[2] = ""
        gradient_descent_text_list[3] = ""
        loss_text_gradient_descent = MathTex(*gradient_descent_text_list, font_size=font_size).move_to(loss_text_position)
        self.play(TransformMatchingTex(loss_text_gradient, loss_text_gradient_descent), ReplacementTransform(gradient_header, gradient_descent_header))
        self.wait(7)

        font_size = DEFAULT_FONT_SIZE / 2
        loss_text_position = np.array([0, -3, 0])
        new_text = MathTex(*gradient_descent_text_list, font_size=font_size).move_to(loss_text_position)
        self.play(ReplacementTransform(loss_text_gradient_descent, new_text), FadeOut(gradient_descent_header))

        y_range2 = (-0.2, 6.0)
        self.ax1 = self.draw_axis(x_range, y_range, length, length, position1, 'x', 'y', ORANGE)
        self.ax2 = self.draw_axis(x_range, y_range2, length, length, position2, 'w', r"L_\text{MSE}", BLUE)
        self.wait(2)

        data_dots = [
            Dot(position1 + np.array([x, y, 0]), radius=DEFAULT_DOT_RADIUS / 2) 
            for x, y in zip(self.x_data, self.y_data)
        ]
        self.data_group = VGroup(*data_dots)
        self.data_group.set_z_index(1)
        self.play(Write(self.data_group))

        self.gradient_descent(2.0, [(4, 0.92), (5, 0.2), (5, 0.1)], position1)
        self.wait(5)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
    def gradient_descent(self, weight: float, steps: List[Tuple[int, float]], center: Tuple[float, float]):
        error_max = 3.3
        line = self.ax1.plot(lambda x: weight * x, color=WHITE)
        line.set_z_index(2)
        error_bars, mse, mean_error = self.get_error_bars(
            center, weight, error_max
        )
        error_group = VGroup(*error_bars)
        error_point = self.ax2.coords_to_point(weight, mse)
        mse_point = Dot(error_point)
        self.play(Create(line), Create(error_group))
        self.play(Write(mse_point))
        eta = steps[0][1]
        eta_label = ValueTracker(eta)

        eta_text = always_redraw(lambda: MathTex(f"\eta={eta_label.get_value():.2f}").move_to(np.array([0.5, 3.0, 0])))
        self.play(Write(eta_text))

        for n_steps, eta in steps:
            self.play(eta_label.animate.set_value(eta))
            for _ in range(n_steps):
                weight -= eta * mean_error
                new_line = self.ax1.plot(lambda x: weight * x, color=WHITE)
                new_line.set_z_index(2)
                error_bars, mse, mean_error = self.get_error_bars(center, weight, error_max)
                new_error_group = VGroup(*error_bars)
                error_point = self.ax2.coords_to_point(weight, mse)
                path = Line(mse_point.get_center(), error_point)
                mse_point = Dot(error_point)
                self.play(
                    ReplacementTransform(line, new_line), 
                    ReplacementTransform(error_group, new_error_group), 
                    Write(mse_point),
                    Create(path)
                )
                line = new_line
                error_group = new_error_group
            self.wait(1)

    def draw_axis(
        self,
        x_range: tuple, 
        y_range: tuple,
        x_length: int,
        y_length: int, 
        position: np.ndarray, 
        x_label: str, 
        y_label: str,
        color: str
    ) -> Axes:
        
        ax = Axes(
            x_range=x_range, 
            y_range=y_range, 
            x_length=x_length, 
            y_length=y_length, 
            tips=False,
            axis_config={"color": color}
        )
        ax.move_to(position)
        x_label = ax.get_x_axis_label(x_label)
        y_label = ax.get_y_axis_label(y_label)
        self.play(Create(ax), Write(x_label), Write(y_label), run_time=2)

        return ax
    
    def get_error_bars(
        self, 
        center: np.ndarray, 
        weight: float, 
        error_max: float
    ) -> List[Line]:
        
        cmap = colormaps["coolwarm"]
        lines = [None] * self.x_data.shape[0]
        mse = np.zeros(len(self.x_data))
        derivative = np.zeros(len(self.x_data))

        for i, (x, y) in enumerate(zip(self.x_data, self.y_data)):
            error = y - weight * x
            derivative[i] =  -x * (y - weight * x)
            abs_error = abs(error)
            mse[i] = abs_error ** 2
            color_ = cmap(np.sqrt(min(abs_error, error_max) / error_max))[:3]
            lines[i] = Line(
                center + np.array([x, y, 0]), 
                center + np.array([x, weight * x, 0]), 
                color=rgb_to_color(color_)
            )
            lines[i].set_z_index(self.data_group.z_index - 1)
        
        return lines, mse.mean(), derivative.mean()
