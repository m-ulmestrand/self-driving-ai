from manim import *
import numpy as np
from typing import List


class DrawEquation(Scene):
    def construct(self):
        rl_text = MathTex(r"\text{Q-learning: }", r"\text{The Bellman equation}", font_size=40).move_to([0, 3, 0])
        self.eq_texts = []
        eq_pos = [0, 2, 0]
        self.eq_texts.append(MathTex(
            r"Q(s_t, a_t) = ", 
            r"\sum_{i=t+1}^T \gamma^i r(s_i, a_i)", 
            font_size=30
        ).move_to(eq_pos))
        tex_list = np.array([
            r"Q(s_t, a_t) = ", 
            r"r(s_t, a_t)", 
            r"+ \gamma^1 r(s_{t+1}, a_{t+1}) + \dots + ",
            r"\gamma^{T-2}", 
            r"r", 
            r"(s_{T-2}, a_{T-2})", 
            r"+ \gamma^{T-1}", 
            r"r(s_{T-1}, a_{T-1}) + ", 
            r"\gamma^T", 
            r"r", 
            r"(s_T, a_T",
            r")"
        ])
        self.append_text(tex_list)
        tex_list[-3] = r"Q"
        self.append_text(tex_list)
        tex_list[-1] = r")\right)"
        tex_list[-4] = r"\gamma"
        tex_list[-5] = r"\left(r(s_{T-1}, a_{T-1}) +"
        self.append_text(tex_list)
        tex_list[4] = r"\left(r"
        tex_list[6] = r"+ \gamma"
        tex_list[-2] = r"(s_{T-1}, a_{T-1}"
        tex_list[-4] = r" "
        tex_list[-5] = r" "
        self.append_text(tex_list)
        tex_list[4] = r"Q"
        tex_list[6:] = r" "
        self.append_text(tex_list)
        tex_list[2] = r"+ \gamma"
        tex_list[3] = r"Q"
        tex_list[4] = r"(s_{t+1},"
        tex_list[5] = r"a_{t+1}"
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
        self.anims = [[None]] * (len(self.eq_texts) - 1)
        
        for i, (txt1, txt2) in enumerate(zip(self.eq_texts[0:-1], self.eq_texts[1:])): 
            self.anims[i] = [ReplacementTransform(txt1, txt2)]

        self.brace_anim_with_text(1, Group(self.eq_texts[2][-3:]), "Q value at $T$")
        self.brace_anim_with_text(2, Group(self.eq_texts[3][-5:]), "Q value at $T-1$")
        self.brace_anim_with_text(3, Group(self.eq_texts[4][4:]), "Q value at $T-2$")

        for anim in self.anims:
            self.play(*anim)
            self.wait(2)
        self.wait(5)
        
        self.play(Write(Tex(r"At each time step, store:", font_size=30).move_to([-2, 1, 0])))
        objective_text = VGroup(
            Tex(r"$\bullet$ Initial state $s_t$", font_size=30),
            Tex(r"$\bullet$ Action $a_t$", font_size=30),
            Tex(r"$\bullet$ New state $s_{t+1}$", font_size=30)
        ).move_to([-1.6, 0.5, 0])

        objective_text.arrange(DOWN, center=False, aligned_edge=LEFT)
        self.play(Write(objective_text), run_time=3)
        self.wait(1)
        self.play(Write(Tex(r"Minimize:", font_size=30).move_to([-3, -1.2, 0])))

        orig_eq = MathTex(
            r"Q(s_t, a_t)",
            r"=",
            r"r(s_t, a_t) + \gamma\max_a Q(s_{t+1}, a)",
            font_size=30
        ).move_to(eq_pos)
        
        loss_pos = [0, -2, 0]
        eq_sub = MathTex(
            "r(s_t, a_t) + \gamma\max_a Q(s_{t+1}, a)",
            "=",
            "Q(s_t, a_t)",
            font_size=30
        ).move_to(loss_pos)

        loss = MathTex(
            "r(s_t, a_t) + \gamma\max_a Q(s_{t+1}, a)",
            "-",
            "Q(s_t, a_t)",
            font_size=30
        ).move_to(loss_pos)

        loss2 = MathTex(
            "r", 
            "(s_t, a_t) + \gamma\max_a Q(s_{t+1}, a)",
            "-",
            "Q(s_t, a_t",
            ")",
            font_size=30
        ).move_to(loss_pos)

        self.play(TransformMatchingTex(orig_eq, eq_sub))

        self.play(ReplacementTransform(eq_sub, loss))
        new_loss = MathTex(
            "\left(r", 
            "(s_t, a_t) + \gamma\max_a Q(s_{t+1}, a)",
            "-",
            "Q(s_t, a_t",
            r")\right)^2",
            font_size=30
        ).move_to(loss_pos)

        self.play(ReplacementTransform(loss, loss2), run_time=0)
        self.play(TransformMatchingTex(loss2, new_loss))
        self.wait(5)

        new_rl_text = MathTex(r"\text{Q-learning: }", r"\text{$\epsilon$-greedy policy}", font_size=40).move_to([0, 3, 0])
        self.play(
            *[FadeOut(mob) for mob in self.mobjects], 
            TransformMatchingTex(rl_text, new_rl_text), 
            run_time=1
        )

    def append_text(self, tex_list: List[str], pos: np.ndarray = [0, 2, 0]):
        self.eq_texts.append(MathTex(*tex_list, font_size=30).move_to(pos))

    def brace_anim_with_text(self, id: int, mobject: Mobject, text: str):
        brace = Brace(mobject, DOWN)
        tex = Tex(text, font_size=20).next_to(brace, DOWN)
        self.anims[id].extend([Create(brace), Write(tex)])
        self.anims[id + 1].append(FadeOut(brace, tex))