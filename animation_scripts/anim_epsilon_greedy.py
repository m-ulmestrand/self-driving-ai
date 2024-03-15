from manim import *


class DrawScheme(Scene):
    def construct(self):
        title_text = MathTex(r"\text{Q-learning: }", r"\text{$\epsilon$-greedy policy}", font_size=40).move_to([0, 3, 0])
        brace_text = MathTex(
            r"a_t = \begin{cases} \text{any action}  & \text{with probability } \epsilon\\ \max_a Q(s_{t}, a) & \text{with probability } 1 - \epsilon \end{cases}",
            font_size=30
        ).move_to([0, 2, 0])

        self.add(title_text)
        self.wait(4)
        self.play(Write(brace_text), run_time=3)
        self.wait(5)

        ax = Axes(
            x_range=(0, 100, 10), 
            y_range=(0.0, 1.0, 0.2), 
            x_length=6, 
            y_length=3, 
            tips=False,
            axis_config={"include_numbers": True, "font_size": 20}
        ).move_to([0, -1, 0])

        x_label = ax.get_x_axis_label(Tex("Generation", font_size=20))
        y_label = ax.get_y_axis_label(MathTex(r"\epsilon"))
        graph = ax.plot(lambda x: max(0.8 - x / 40, 0.1), x_range=[0, 100], use_smoothing=False)

        self.play(Write(ax), Write(x_label), Write(y_label))
        self.wait(2)
        self.play(Create(graph), run_time=2)
        self.wait(5)
        self.play(FadeOut(*[m for m in self.mobjects]))