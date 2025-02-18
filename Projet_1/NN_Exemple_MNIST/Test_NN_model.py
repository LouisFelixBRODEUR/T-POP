import tkinter as tk
import numpy as np

import time
import math
import torch
import torch.nn as nn
import os

model_to_test = 'model_test.pth'

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function pour dessiner sur un 28x28 canvas
def show_skills_draw():
    root = tk.Tk()
    root.title("Draw on 28x28 Canvas")
    canvas_size = 500
    scale_factor = canvas_size // 28
    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
    canvas.pack()
    drawn_array = np.zeros((28, 28), dtype=np.uint8)
    def draw(event):
        x, y = event.x, event.y
        y_coord = y // scale_factor
        x_coord = x // scale_factor
        if 0 <= y_coord < 28 and 0 <= x_coord < 28:
            canvas.create_rectangle((x // scale_factor) * scale_factor,
                        (y // scale_factor) * scale_factor,
                        (x // scale_factor) * scale_factor + scale_factor,
                        (y // scale_factor) * scale_factor + scale_factor,
                        fill='black', outline='black')
            drawn_array[y_coord, x_coord] = 255
    canvas.bind("<B1-Motion>", draw)
    root.mainloop()
    return drawn_array

# processing du drawn_canvas
def preprocess_drawn_image(drawn_array):
    drawn_array = drawn_array.astype(np.float32) / 255.0 # Normalization
    drawn_tensor = torch.tensor(drawn_array).unsqueeze(0).unsqueeze(0) # To Tensor
    return drawn_tensor

def predict_number(model, drawn_array):
    model.eval()
    with torch.no_grad():
        drawn_tensor = preprocess_drawn_image(drawn_array)
        output = model(drawn_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    
# Load le model
model = SimpleNN()
model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+"\\"+model_to_test))
model.eval()

# Test le model avec le canvas
if __name__ == "__main__":
    while 1:
        drawn_image = show_skills_draw()
        predicted_number = predict_number(model, drawn_image)
        print(f"The model predicts that the number is: {predicted_number}")