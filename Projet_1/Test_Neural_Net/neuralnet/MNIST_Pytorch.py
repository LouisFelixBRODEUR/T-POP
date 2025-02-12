# https://www.youtube.com/watch?v=vBlO87ZAiiw

# For dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# For Loaders
from torch.utils.data import DataLoader

# For Model architechture
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For optimizing
import torch

import matplotlib.pyplot as plt
import numpy as np
import random
import tkinter as tk


# Defining model architechture
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)#1in 10out
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)#Fully Connected Layer
        self.fc2 = nn.Linear(50, 10)

    #Activation function called in foward (x is the data)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)

class MNIST_app():
    def __init__(self):
        # Downloading Trainin set
        self.train_data = datasets.MNIST(
            root = 'data',
            train = True,
            transform = ToTensor(),
            download = True
        )

        # Downloading Test set
        self.test_data = datasets.MNIST(
            root = 'data',
            train = False,
            transform = ToTensor(),
            download = True
        )

        # Creating dataloader to get the data into model
        self.loaders = {

            'train' : DataLoader(self.train_data,
                                batch_size=100,
                                shuffle=True,
                                num_workers=1),

            'test' : DataLoader(self.test_data,
                                batch_size=100,
                                shuffle=True,
                                num_workers=1)
        }
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CNN().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # learing rate

        self.loss_fn = nn.CrossEntropyLoss() # loss function


    # defining testing funciton
    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.loaders['test']:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.loaders['test'].dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(self.loaders["test"].dataset)} ({100. * correct / len(self.loaders["test"].dataset):.0f}%)\n')

    def train(self, epoch): 
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.loaders['train']):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data) # what the current model would say
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                print(f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(self.loaders["train"].dataset)} ({100. * batch_idx / len(self.loaders["train"]):.0f}%)]\t{loss.item():.6f}')

    def launch(self):
        # Starting training
        for epoch in range(1, 15):
            self.train(epoch)
        self.test()

    def show_skills(self):
        self.model.eval()
        random_integers = [random.randint(1, 9999) for _ in range(1)]
        for i in random_integers:
            data, target = self.test_data[i]
            print(data)
            data = data.unsqueeze(0).to(self.device)
            output = self.model(data)
            prediction = output.argmax(dim=1, keepdim=True).item()
            print(f'Prediction: {prediction}')
            image = data.squeeze(0).numpy()[0]
            plt.imshow(image, cmap='gray')
            plt.show()

    def convert_array_to_tensor(self, array):
        # Ensure the input is a NumPy array
        if not isinstance(array, np.ndarray):
            raise TypeError("Input should be a NumPy array")

        # Convert the NumPy array to a PyTorch tensor using ToTensor
        to_tensor = ToTensor()
        tensor = to_tensor(array).unsqueeze(0)  # Adding batch dimension if necessary

        return tensor

    def show_skills_draw(self):
        root = tk.Tk()
        root.title("Draw on 28x28 Canvas")
        canvas_size = 500  # Scaling to make it easier to draw
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

        # Evals for drawn array
        data = self.convert_array_to_tensor(drawn_array)[0]
        self.model.eval()
        data = data.unsqueeze(0).to(self.device)
        output = self.model(data)
        prediction = output.argmax(dim=1, keepdim=True).item()
        print(f'Prediction: {prediction}')
        # image = data.squeeze(0).numpy()[0]
        # plt.imshow(image, cmap='gray')
        # plt.show()

if __name__ == '__main__':
    app = MNIST_app()
    # app.show_skills_draw()
    app.launch()
    while 1:
        app.show_skills_draw()


