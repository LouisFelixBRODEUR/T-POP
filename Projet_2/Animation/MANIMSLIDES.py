from manim import *
from manim_slides import Slide

# $manim-slides render -p -ql PlayGround.py MyPresentation
# $manim-slides present MyPresentation

class MyPresentation(Slide):
    def construct(self):
        self.play(Write(Text("Hello world!")))
        self.next_slide()
        self.play(Create(Square()))
        self.next_slide()