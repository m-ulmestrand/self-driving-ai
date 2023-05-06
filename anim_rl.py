from manim import *
import numpy as np


class DrawDiagram(Scene):
    def construct(self):
        agent_opt_pos = np.array([0, 2.5, 0])
        ea_pos = np.array([-2, 1.5, 0])
        swarm_pos = np.array([-2, 0, 0])
        ga_pos = np.array([-3, 0.5, 0])
        more_pos = np.array([-1, 0.5, 0])
        rl_pos = np.array([2, 1.5, 0])
        ql_pos = np.array([1, 0.5, 0])
        pg_pos = np.array([3, 0.5, 0])
        ac_pos = np.array([2, -0.5, 0])
        angle_diff = 35
        a2c_pos = ac_pos + self.vector_down(-3 / 2 * angle_diff)
        a3c_pos = ac_pos + self.vector_down(-angle_diff / 2)
        trpo_pos = ac_pos + self.vector_down(angle_diff / 2)
        ppo_pos = ac_pos + self.vector_down(3 / 2 * angle_diff)

        agent_opt = Text("Agent optimization", font_size=20).move_to(agent_opt_pos)

        # Evolutionary algorithm branch
        self.play(Create(agent_opt), run_time=1.5)
        self.wait(2)
        self.new_node(agent_opt_pos)
        self.draw_branch(agent_opt_pos, ea_pos, "Evolutionary algorithms")
        self.wait(2)
        self.new_node(ea_pos)
        _, _, ga_text = self.draw_branch(ea_pos, ga_pos, "Genetic algorithms")
        self.draw_branch(ea_pos, swarm_pos, "Swarm intelligence")
        self.draw_branch(ea_pos, more_pos, "...")
        mobjects1 = self.draw_branch(agent_opt_pos, ea_pos, color=BLUE_C, size_mult=1.4, run_time_mult=0.75)
        node = self.new_node(ea_pos, color=BLUE_C, size_mult=1.4)
        mobjects2 = self.draw_branch(ea_pos, ga_pos, color=BLUE_C, size_mult=1.4, run_time_mult=0.75)
        self.play(ga_text.animate.scale(1.15), run_time=0.5)
        self.play(ga_text.animate.set_color(BLUE_D))
        self.wait(2)
        mobject_collection = (*mobjects1, *mobjects2, node, ga_text)
        self.play(*(mobject.animate.set_color(RED) for mobject in mobject_collection))
        self.wait(1)
        self.play(ga_text.animate.set_color(WHITE))
        self.play(ga_text.animate.scale(1/1.15), run_time=0.5)
        self.play(FadeOut(*mobject_collection[:-1]))

        # Reinforcement learning branch
        self.draw_branch(agent_opt_pos, rl_pos, "Reinforcement learning")
        self.wait(2)
        self.new_node(rl_pos)
        _, _, ql_text = self.draw_branch(rl_pos, ql_pos, "Deep Q-learning")
        self.draw_branch(rl_pos, pg_pos, "Policy gradient")
        self.wait(2)
        self.new_node(ql_pos)
        self.draw_branch(ql_pos, ac_pos, run_time_mult=0.5)
        self.new_node(pg_pos)
        self.draw_branch(pg_pos, ac_pos, "Actor-critic RL", run_time_mult=0.5)
        self.new_node(ac_pos)
        self.draw_branch(ac_pos, a2c_pos, "A2C", run_time_mult=0.5)
        self.draw_branch(ac_pos, a3c_pos, "A3C", run_time_mult=0.5)
        self.draw_branch(ac_pos, trpo_pos, "TRPO", run_time_mult=0.5)
        self.draw_branch(ac_pos, ppo_pos, "PPO", run_time_mult=0.5)
        self.draw_branch(agent_opt_pos, rl_pos, color=BLUE_C, size_mult=1.4, run_time_mult=0.75)
        self.new_node(rl_pos, color=BLUE_C, size_mult=1.4)
        self.draw_branch(rl_pos, ql_pos, color=BLUE_C, size_mult=1.4, run_time_mult=0.75)
        self.play(ql_text.animate.scale(1.5), run_time=0.5)
        self.play(ql_text.animate.set_color(BLUE_D))
        self.wait(4)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

    def draw_branch(
            self, 
            pos_before: np.ndarray, 
            pos: np.ndarray, 
            text: str = None, 
            font_size: int = 15,
            displace: np.ndarray = np.array([0, 0.22, 0]),
            color: str = WHITE,
            size_mult: float = 1.,
            run_time_mult: float = 1.
    ):
        if text is not None:
            txt = Text(text, font_size=font_size, color=color).move_to(pos)

        dot = Dot(
            pos + displace, 
            (DEFAULT_DOT_RADIUS * size_mult / 2), 
            color=color)
        line = Line(
            pos_before - displace, 
            pos + displace, 
            color=color
        )
        line.stroke_width *= size_mult
        line.set_z_index(dot.z_index - 1)
        self.play(Create(line), run_time=(run_time_mult * 1.))
        self.play(FadeIn(dot), run_time=(run_time_mult * 0.25))

        if text is not None:
            self.play(FadeIn(txt), run_time=0.5)
            txt.set_z_index(dot.z_index + 1)
            return line, dot, txt
        
        return line, dot

    def new_node(
            self,
            pos: np.ndarray,
            displace: np.ndarray = np.array([0, 0.22, 0]),
            color: str = WHITE,
            size_mult: float = 1.
    ):
        dot = Dot(pos - displace, (DEFAULT_DOT_RADIUS * size_mult / 2), color=color)
        self.play(FadeIn(dot), run_time=0.25)
        return dot
    
    def vector_down(
            self,
            angle: float,
            length: float = 1.,
            aspect_ratio: float = 3/2
    ) -> np.ndarray:
        down_angle = 3 * PI / 2 
        deg_to_rad = PI / 180
        rad_angle = angle * deg_to_rad
        return length * np.array([
            np.cos(down_angle + rad_angle), 
            np.sin(down_angle + rad_angle) * aspect_ratio, 
            0
        ])