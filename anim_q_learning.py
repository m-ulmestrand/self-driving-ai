from manim import *
import numpy as np
from typing import List


class DrawEquation(Scene):
    def construct(self):
        rl_text = Tex("Q-learning: The Bellman equation", font_size=40).move_to([0, 2, 0])
        self.eq_texts = []
        self.eq_texts.append(MathTex(
            r"Q(s_0, a_0) = ", 
            r"\sum_{i=1}^n \gamma^i r(s_i, a_i)", 
            font_size=30
        ).move_to([0, 1, 0]))
        tex_list = np.array([
            r"Q(s_0, a_0) = ", 
            r"r(s_0, a_0)", 
            r"+ \gamma^1 r(s_1, a_1) + \dots + ",
            r"\gamma^{n-2}", 
            r"r", 
            r"(s_{n-2}, a_{n-2})", 
            r"+ \gamma^{n-1}", 
            r"r(s_{n-1}, a_{n-1}) + ", 
            r"\gamma^n", 
            r"r", 
            r"(s_n, a_n",
            r")"
        ])
        self.append_text(tex_list)
        tex_list[-3] = r"Q"
        self.append_text(tex_list)
        tex_list[-1] = r")\right)"
        tex_list[-5] = r"\left(r(s_{n-1}, a_{n-1}) +"
        self.append_text(tex_list)
        tex_list[-1] = r")"
        tex_list[-2] = r"(s_{n-1}, a_{n-1}"
        tex_list[-4] = r" "
        tex_list[-5] = r" "
        self.append_text(tex_list)
        tex_list[4] = r"Q"
        tex_list[6:] = r" "
        self.append_text(tex_list)
        tex_list[2] = r"+ \gamma"
        tex_list[3] = r"Q"
        tex_list[4] = r"(s_1,"
        tex_list[5] = r"a_1"
        tex_list[6] = r")"
        tex_list[7:] = r" "
        self.append_text(tex_list)
        tex_list[2] = r"+ \gamma"
        tex_list[3] = r"\max_a Q"
        tex_list[5] = r"a"
        self.append_text(tex_list)
        self.play(Write(rl_text))
        self.wait(1)
        self.play(Write(self.eq_texts[0]))
        
        for txt1, txt2 in zip(self.eq_texts[0:-1], self.eq_texts[1:]): 
            self.wait(2)
            self.play(ReplacementTransform(txt1, txt2))
        self.wait(5)
        
    def append_text(self, tex_list: List[str], pos: np.ndarray = [0, 1, 0]):
        self.eq_texts.append(MathTex(*tex_list, font_size=30).move_to(pos))
