from manim import *
import numpy as np

# TO RUN THE CODE:
# $manim -pql <codename.py> MichelsonInterferometer
# TO RENDER IN 4K:
# $manim -p -qk <codename.py> MichelsonInterferometer

# TODO ajouter titre de linstrument le Michelphone
# TODO update gitignore to not take the videofiles(too large)

class MichelsonInterferometer(Scene):
    def construct(self):
        # Constants
        WAVELENGTH = 0.5
        BEAM_AMPLITUDE = 0.15
        PHASE_SHIFT_SPEED = 2.0
        
        # Beam Splitter with diagonal line
        beam_splitter = Square(side_length=0.5, color=BLUE, fill_opacity=0.7, 
                             stroke_width=2).shift(LEFT * 2)
        diagonal_line = Line(beam_splitter.get_corner(UR), beam_splitter.get_corner(DL), 
                          color=BLACK, stroke_width=3)
        beam_splitter_group = VGroup(beam_splitter, diagonal_line)
        
        # Mirrors
        mirror1 = Rectangle(width=0.1, height=1, color=WHITE, fill_opacity=0.7, 
                           stroke_width=2).shift(UP * 2 + LEFT * 2).rotate(PI/2)
        mirror2 = Rectangle(width=0.1, height=1, color=WHITE, fill_opacity=0.7, 
                           stroke_width=2).shift(RIGHT * 2)
        mirror_motion = ValueTracker(0)
        mirror2.add_updater(lambda m: m.move_to(RIGHT * (2 + 0.2 * np.sin(3 * mirror_motion.get_value()))))
        
        # Speaker icon
        speaker = SVGMobject("speaker.svg").scale(0.3).set_color(WHITE).next_to(mirror2, RIGHT*2, buff=0.2)

        # Detector
        # detector = ScreenRectangle(height=1.5, color=GREEN, fill_opacity=0.3, 
        #                          stroke_width=2).shift(DOWN * 2 + LEFT * 2)
        detector = Square(side_length=1, color=GREEN, fill_opacity=0.3, 
                                 stroke_width=2).shift(DOWN * 2 + LEFT * 2)
        
        # Laser source
        laser = Rectangle(width=0.6, height=0.3, color=RED, fill_opacity=0.7, 
                             stroke_width=2).shift(LEFT * 5)
        
        # Labels
        components = VGroup(
            Text("Beam Splitter").next_to(beam_splitter, DOWN + RIGHT, buff=0.1).scale(0.5).shift(LEFT * 1),
            Text("Miroir fixe").next_to(mirror1, UP, buff=0.1).scale(0.5),
            Text("Miroir mobile").next_to(mirror2, UP, buff=0.1).scale(0.5),
            Text("Détecteur").next_to(detector, DOWN, buff=0.1).scale(0.5),
            Text("Source Laser").next_to(laser, UP, buff=0.1).scale(0.5)
        )
        
        # Time counter for phase animation
        time = ValueTracker(0)
        
        # Wave creation function
        def create_wave(start, end, initial_phase=0, vertical=False):
            return always_redraw(lambda: self.create_sine_wave(
                start=start,
                end=end,
                phase=time.get_value() * PHASE_SHIFT_SPEED + initial_phase,
                amplitude=BEAM_AMPLITUDE,
                wavelength=WAVELENGTH,
                vertical=vertical,
            ))
        
        # Laser to Beam splitter
        beam1 = create_wave(
            start=laser.get_right(),
            end=beam_splitter.get_left(),
            initial_phase=0,
        )
        
        # Beam splitter to mirrors
        beam2 = create_wave(
            start=beam_splitter.get_top(),
            end=mirror1.get_bottom(),
            initial_phase=0,
            vertical=True
        )
        
        # Beam splitter to moving mirror (will be updated)
        beam3 = always_redraw(lambda: self.create_sine_wave(
            start=beam_splitter.get_right(),
            end=mirror2.get_left(),
            phase=time.get_value() * PHASE_SHIFT_SPEED,
            amplitude=BEAM_AMPLITUDE,
            wavelength=WAVELENGTH,
        ))

        # Mirrors back to beam splitter
        beam4 = create_wave(
            start=mirror1.get_bottom(),
            end=beam_splitter.get_top(),
            initial_phase=-PI*3.15,
            vertical=True
        )

        
        # Moving mirror back to beam splitter (will be updated)
        beam5 = always_redraw(lambda: self.create_sine_wave(
            start=mirror2.get_left(),
            end=beam_splitter.get_right(),
            phase=time.get_value() * PHASE_SHIFT_SPEED + PI,
            amplitude=BEAM_AMPLITUDE,
            wavelength=WAVELENGTH,
            vertical=False
        ))

        # Beam splitter to detector - first beam (static)
        beam6 = create_wave(
            start=beam_splitter.get_bottom(),
            end=detector.get_top(),
            initial_phase=0,
            vertical=True
        )
        
        # Beam splitter to detector - second beam (dynamic, responds to mirror)
        beam7 = always_redraw(lambda: self.create_sine_wave(
            start=beam_splitter.get_bottom(),
            end=detector.get_top(),
            phase=2 * np.sin(3 * mirror_motion.get_value()) + PI/2,  # Oscillating phase
            amplitude=BEAM_AMPLITUDE,
            wavelength=WAVELENGTH,
            vertical=True
        ))

        # Clean circular interference pattern in red
        interference_pattern = always_redraw(lambda: self.create_clean_circular_interference(
            time.get_value() * PHASE_SHIFT_SPEED + mirror_motion.get_value() * 10,
            detector
        ))
        
        # Animation sequence
        self.wait(0.5)
        self.play(
            LaggedStart(
                Create(laser),
                Create(beam_splitter_group),
                Create(mirror1),
                Create(mirror2),
                Create(detector),
                Create(speaker),
                lag_ratio=0.1
            ),
            run_time=4
        )
        
        self.play(Write(components), run_time=4)
        
        # Animate waves
        self.add(time)
        self.play(Create(beam1), run_time=2)
        self.play(
            AnimationGroup(
                Create(beam2),
                Create(beam3)
            ),
            run_time=2
        )
        self.play(
            AnimationGroup(
                Create(beam4),
                Create(beam5)
            ),
            run_time=2
        )
        self.play(
            AnimationGroup(
                Create(beam6),
                Create(beam7)
            ),
            run_time=2
        )
        self.play(Create(interference_pattern),run_time=2)
        self.play(mirror_motion.animate.increment_value(2 * PI), run_time=10, rate_func=linear)
                
    def create_sine_wave(self, start, end, phase=0, amplitude=0.2, wavelength=0.5, vertical=False):
        """Wave creation without arrows"""
        distance = np.linalg.norm(end - start)
        k = 2 * PI / wavelength

        def wave_func(t):
            if vertical:
                return np.array([
                    interpolate(start[0], end[0], t) + amplitude * np.sin(k * distance * t + phase),
                    interpolate(start[1], end[1], t),
                    0
                ])
            else:
                return np.array([
                    interpolate(start[0], end[0], t),
                    interpolate(start[1], end[1], t) + amplitude * np.sin(k * distance * t + phase),
                    0
                ])
        
        return ParametricFunction(
            wave_func,
            t_range=[0, 1, 0.02],
            color=RED,
            stroke_width=4
        )
    
    def create_clean_circular_interference(self, phase, detector):
        """Creates clean circular interference fringes in red"""
        pattern = VGroup()
        max_radius = min(detector.width, detector.height)/2 - 0.2
        
        # Create smooth circular fringes
        for r in np.linspace(0.1, 1, 10):
            radius = r * max_radius
            # Intensity varies with radius and phase
            intensity = (np.cos(phase + r*15)**2)
            
            circle = Circle(
                radius=radius,
                color=interpolate_color(DARK_GRAY, RED, intensity),
                stroke_width=2 + 3*intensity
            ).move_to(detector.get_center())
            
            pattern.add(circle)
        
        return pattern