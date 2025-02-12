import numpy as np
import tkinter as tk

def show_skills_draw():
    # Initialize the root window
    root = tk.Tk()
    root.title("Draw on 28x28 Canvas")

    # Create a 28x28 canvas
    canvas_size = 500  # Scaling to make it easier to draw
    scale_factor = canvas_size // 28
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
    canvas.pack()

    # Initialize the array
    drawn_array = np.zeros((28, 28), dtype=np.uint8)

    def draw(event):
        # Get the canvas coordinates
        x, y = event.x, event.y
        # Draw a black square on the canvas
        # Update the drawn array
        y_coord = y // scale_factor
        x_coord = x // scale_factor
        if 0 <= y_coord < 28 and 0 <= x_coord < 28:
            canvas.create_rectangle((x // scale_factor) * scale_factor,
                        (y // scale_factor) * scale_factor,
                        (x // scale_factor) * scale_factor + scale_factor,
                        (y // scale_factor) * scale_factor + scale_factor,
                        fill='black', outline='black')
            drawn_array[y_coord, x_coord] = 255

    # Bind the left mouse button click event to the draw function
    canvas.bind("<B1-Motion>", draw)

    # Run the Tkinter event loop
    root.mainloop()

    return drawn_array

print(show_skills_draw())