from manim import *
import numpy as np

# TO RUN THE CODE:
# $manim -pql <codename.py> MichelsonInterferometer
# TO RENDER IN 4K:
# $manim -p -qk <codename.py> MichelsonInterferometer

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
        detector = Square(side_length=1, color=GREEN, fill_opacity=0.3, 
                                 stroke_width=2).shift(DOWN * 2 + LEFT * 2)
        
        # Laser source
        laser = Rectangle(width=0.6, height=0.3, color=RED, fill_opacity=0.7, 
                             stroke_width=2).shift(LEFT * 5)
        
        # Labels
        components = VGroup(
            Text("Beam Splitter").next_to(beam_splitter, UP + RIGHT, buff=0.1).scale(0.5).shift(LEFT * 1.1+ DOWN * 0.3),
            Text("Miroir fixe").next_to(mirror1, UP, buff=0.1).scale(0.5),
            Text("Miroir mobile").next_to(mirror2, UP, buff=0.1).scale(0.5),
            Text("Détecteur").next_to(detector, DOWN, buff=0.1).scale(0.5),
            Text("Source Laser").next_to(laser, UP, buff=0.1).scale(0.5)
        )
        
        # Time counter for phase animation
        time = ValueTracker(0)
        
        # graph on the right side of the detector
        graph_origin = detector.get_right() + RIGHT * 2.3
        graph_width = 4
        graph_height = 2
        graph_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 4, 0.5],
            x_length=graph_width,
            y_length=graph_height,
            axis_config={"color": WHITE},
            tips=False,
        ).move_to(graph_origin)

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

        # Calculate distances for phase continuity
        k = 2 * PI / WAVELENGTH
        distance_to_mirror1 = np.linalg.norm(mirror1.get_bottom() - beam_splitter.get_top())
        distance_to_mirror2 = lambda: np.linalg.norm(mirror2.get_left() - beam_splitter.get_right())
        distance_to_detector = np.linalg.norm(detector.get_top() - beam_splitter.get_bottom())

        # Mirrors back to beam splitter
        beam4 = create_wave(
            start=mirror1.get_bottom(),
            end=beam_splitter.get_top(),
            initial_phase=k * distance_to_mirror1,  # Phase accumulated going to mirror
            vertical=True
        )

        # Moving mirror back to beam splitter (will be updated)
        beam5 = always_redraw(lambda: 
            self.create_sine_wave(
                start=mirror2.get_left(),
                end=beam_splitter.get_right(),
                phase=time.get_value() * PHASE_SHIFT_SPEED + k * distance_to_mirror2(),
                amplitude=BEAM_AMPLITUDE,
                wavelength=WAVELENGTH,
                vertical=False
            )
        )

        # Beam splitter to detector - first beam (from fixed mirror path)
        beam6 = always_redraw(lambda: 
            self.create_sine_wave(
                start=beam_splitter.get_bottom(),
                end=detector.get_top(),
                phase=time.get_value() * PHASE_SHIFT_SPEED + k * (2 * distance_to_mirror1),  # Round trip to fixed mirror
                amplitude=BEAM_AMPLITUDE,
                wavelength=WAVELENGTH,
                vertical=True
            )
        )
        
        # Beam splitter to detector - second beam (from moving mirror path)
        beam7 = always_redraw(lambda: 
            self.create_sine_wave(
                start=beam_splitter.get_bottom(),
                end=detector.get_top(),
                phase=time.get_value() * PHASE_SHIFT_SPEED + k * (2 * distance_to_mirror2()) + PI/2,  # Round trip to moving mirror + π/2 phase shift from beam splitter
                amplitude=BEAM_AMPLITUDE,
                wavelength=WAVELENGTH,
                vertical=True
            )
        )

        # Realistic interference pattern based on phase difference
        interference_pattern = always_redraw(lambda: self.create_realistic_interference(
            detector,
            phase1=k * (2 * distance_to_mirror1),  # Phase from fixed mirror path
            phase2=k * (2 * distance_to_mirror2()) + PI/2,  # Phase from moving mirror path
            time_phase=time.get_value() * PHASE_SHIFT_SPEED
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
        self.play(Create(interference_pattern), run_time=2)
        
        # Add the graph
        self.play(
            Create(graph_axes)
        )
        
        self.play(
            mirror_motion.animate.increment_value(2 * PI),
            run_time=10,
            rate_func=linear
        )
                
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
    
    def create_realistic_interference(self, detector, phase1, phase2, time_phase, wavelength=0.5):
        """Creates physically accurate interference pattern based on phase difference"""
        pattern = VGroup()
        max_radius = min(detector.width, detector.height)/2 - 0.2
        center = detector.get_center()
        k = 2 * PI / wavelength
        
        # Calculate the phase difference between the two beams
        phase_diff = (phase2 - phase1) % (2*PI)
        
        # Create circular fringes with proper intensity distribution
        for r in np.linspace(0, max_radius, 100):
            # Intensity follows I = 4I₀cos²(Δφ/2)
            # Where Δφ is the phase difference plus any radial dependence
            # For circular fringes, we add a radial term to simulate the angular dependence
            intensity = (np.cos((phase_diff + r*k*0.5)/2)**2)
            
            # Create a colored dot at this radius
            dot = Dot(
                point=center + RIGHT * r,
                color=interpolate_color(DARK_GRAY, RED, intensity),
                radius=0.02
            )
            pattern.add(dot)
        
        # Create concentric circles for the full pattern
        full_pattern = VGroup()
        for dot in pattern:
            circle = Circle(
                radius=dot.get_center()[0] - center[0],
                color=dot.get_color(),
                stroke_width=2,
                fill_opacity=0
            ).move_to(center)
            full_pattern.add(circle)
        
        return full_pattern