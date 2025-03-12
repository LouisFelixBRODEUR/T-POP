# pip install torch torchvision torchaudio scikit-learn pandas
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import random

model_name = 'model_test.pth'

# ----------------------------------------------Load and Preprocess the Data--------------------------------------------------
def get_wavelength_data(base_dir):
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()

                # Find the starting point of spectral data
                for i, line in enumerate(lines):
                    if ">>>>>Begin Spectral Data<<<<<" in line:
                        data_start = i + 1
                        break

                # Read spectral data and extract only wavelength values
                data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
                return np.array(data["Wavelength"].tolist())  # Return only the wavelength column as a list

def moving_average(data, window_size=5):
    kernel = np.ones(window_size) / window_size  # Create a uniform kernel
    return np.convolve(data, kernel, mode='same')  # Apply convolution

def Prepare_data(Plante_folder, Background_folder):
    Background_data = get_intensity_data_from_folder(Background_folder)
    Background_data = np.mean(Background_data, axis=0)
    Wavelength_bins = get_wavelength_data(Background_folder)

    Spectro_data = get_intensity_data_from_folder(Plante_folder)
    Spectro_data = [arr for arr in Spectro_data if np.all(np.max(arr) <= 63500)] # Remove saturated

    # range de nanometre a analyser
    # Cap_low, Cap_high = 0, 1000
    # Cap_low, Cap_high = 420, 670
    Cap_low, Cap_high = 420, 800
    low_index = np.argmin(np.abs(Wavelength_bins - Cap_low))
    high_index = np.argmin(np.abs(Wavelength_bins - Cap_high))

    Background_data = moving_average(Background_data, window_size=5) # Smoothen Background
    # Background_data = Background_data/np.max(Background_data) # Normalize Background
    Spectro_data = [arr/Background_data for arr in Spectro_data] # Divide Data by Background

    Spectro_data = [moving_average(arr, window_size=30) for arr in Spectro_data] # Smoothen Spectral Data
    
    Spectro_data = [arr[low_index:high_index] for arr in Spectro_data]
    Wavelength_bins = Wavelength_bins[low_index:high_index]
    Background_data = Background_data[low_index:high_index]

    Spectro_data = [arr / np.max(arr) for arr in Spectro_data] # Normalize Spectral Data
    # Spectro_data -= Background_data

    return Spectro_data

def get_intensity_data_from_folder(base_dir):
    intensity_data = []
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(base_dir, file_name)

            with open(file_path, "r") as file:
                lines = file.readlines()
            
            # Find the starting point of spectral data
            for i, line in enumerate(lines):
                if ">>>>>Begin Spectral Data<<<<<" in line:
                    data_start = i + 1
                    break

            # Read spectral data and extract only intensity values
            data = pd.read_csv(file_path, skiprows=data_start, delimiter="\t", names=["Wavelength", "Intensity"])
            intensity_data.append(data["Intensity"].tolist())  # Append only the intensity column as a list
    return intensity_data

def load_data(plante_folder_paths):
    data = []
    labels = []
    Background_Folder = os.path.dirname(os.path.abspath(__file__)) +"\\Background_7ms_feuille_blanche\\"
    for i, path in enumerate(plante_folder_paths):
        # data.extend(get_intensity_data_from_folder(path))
        data.extend(Prepare_data(path, Background_Folder))
        # plant_name = path.split('\\')[-2]
        plant_name = i
        for j in range(len(data)-len(labels)):
            labels.append(plant_name)
    return np.array(data), np.array(labels)

# Conversion du data en pytorch dataset (tensor)
class PlantDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load train & test data
# test_csv_path = os.path.dirname(os.path.abspath(__file__))+"\\mnist_test.csv"
# train_csv_path = os.path.dirname(os.path.abspath(__file__))+"\\mnist_train.csv" # PATH
Plante_1_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Scindapsus_aureus_20ms\\"
Plante_2_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Kalanchoe_daigremontianum_30ms\\"
Plante_Folders = [Plante_1_folder, Plante_2_folder]

train_images, train_labels = load_data(Plante_Folders) # Train image : 60 000*[784*[]] Train Label : 60 000*[]

# Split train en pour garder un test set pour eviter le overfitting (test_size=0.1=>10%) (random_state=seed for reproductibility)
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=666)

# initialisation des dataset pytorch
train_dataset = PlantDataset(X_train, y_train)
val_dataset = PlantDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ----------------------------------------------Define the Neural Network Model--------------------------------------------------
# fully connected neural network (FCN) pour plus facilement classser les images
# Rectified Linear Unit (ReLU) en Fonction dactivation
class SimpleNN(nn.Module):
    def __init__(self, first_layer_size = 3648):
        super(SimpleNN, self).__init__()
        self.first_layer_size = first_layer_size
        self.fc1 = nn.Linear(first_layer_size, 128)  # input layer
        self.fc2 = nn.Linear(128, 64)  # deuxieme layer
        self.fc3 = nn.Linear(64, 2)  # output layer (2 out pour 2 plantes)
        self.relu = nn.ReLU()  # fonction dactivation
    # implementation des layers
    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(-1, self.first_layer_size)  # linput dun FCN doit etre un vecteur 
        x = self.relu(self.fc1(x))  # input layer 
        x = self.relu(self.fc2(x))  # deuxieme layer
        x = self.fc3(x)  # output layer
        return x
    
class SimpleNN_5layers(nn.Module):
    def __init__(self, first_layer_size=3648):
        super(SimpleNN_5layers, self).__init__()
        self.first_layer_size = first_layer_size
        self.fc1 = nn.Linear(first_layer_size, 20)  #input  hidden layer 1
        self.fc2 = nn.Linear(20, 20) # hidden layer 2
        self.fc3 = nn.Linear(20, 20) # hidden layer 3
        self.fc4 = nn.Linear(20, 20) # hidden layer 4
        self.fc5 = nn.Linear(20, 20) # hidden layer 5
        self.fc6 = nn.Linear(20, 2)  # Output
        self.relu = nn.ReLU()  # function Activation

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(-1, self.first_layer_size)
        x = self.relu(self.fc1(x))  # hidden layer 1
        x = self.relu(self.fc2(x))  # hidden layer 2
        x = self.relu(self.fc3(x))  # hidden layer 3
        x = self.relu(self.fc4(x))  # hidden layer 4
        x = self.relu(self.fc5(x))  # hidden layer 5
        x = self.fc6(x)     
        return x
    
class ModulableNN(nn.Module):
    def __init__(self, first_layer_size=3648, num_hidden_layers=5, neurons_per_layer=20, output_size=2):
        super(ModulableNN, self).__init__()
        
        self.input_size = first_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.output_size = output_size
        
        # Create a list to hold all layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(first_layer_size, neurons_per_layer))
        self.layers.append(nn.ReLU())  # Activation function
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.ReLU())  # Activation function
        
        # Output layer
        self.layers.append(nn.Linear(neurons_per_layer, output_size))
    
    def forward(self, x):
        # Convert input to float32 and reshape it to a vector
        x = x.to(torch.float32)
        x = x.view(-1, self.input_size)
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        return x

# # ----------------------------------------------Find best architechture using montecristo search--------------------------------------------------
# def monte_carlo_search(train_loader, val_loader, input_size, output_size, num_trials=10):
#     # Define the search space
#     num_hidden_layers_range = range(1, 6)  # 1 to 5 hidden layers
#     neurons_per_layer_range = range(10, 101, 10)  # 10 to 100 neurons per layer
#     learning_rate_range = [0.001, 0.01, 0.1]  # Learning rates to try
    
#     best_accuracy = 0
#     best_hyperparams = None
#     best_model = None
    
#     for trial in range(num_trials):
#         # Randomly sample hyperparameters
#         num_hidden_layers = random.choice(num_hidden_layers_range)
#         neurons_per_layer = random.choice(neurons_per_layer_range)
#         learning_rate = random.choice(learning_rate_range)
        
#         print(f"Trial {trial + 1}/{num_trials}: "
#               f"Layers={num_hidden_layers}, Neurons={neurons_per_layer}, LR={learning_rate}")
        
#         # Initialize the model
#         model = ModulableNN(first_layer_size = len(train_images[0]), num_hidden_layers=num_hidden_layers, neurons_per_layer=neurons_per_layer)
        
#         # Define loss function and optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
#         # Train the model
#         num_epochs = 10  # Train for a few epochs for each trial
#         for epoch in range(num_epochs):
#             model.train()
#             for images, labels in train_loader:
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
        
#         # Evaluate the model on the validation set
#         model.eval()
#         correct_predictions = 0
#         total_samples = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs, 1)
#                 correct_predictions += (predicted == labels).sum().item()
#                 total_samples += labels.size(0)
        
#         accuracy = correct_predictions / total_samples * 100
#         print(f"Validation Accuracy: {accuracy:.2f}%")
        
#         # Track the best model
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_hyperparams = {
#                 "num_hidden_layers": num_hidden_layers,
#                 "neurons_per_layer": neurons_per_layer,
#                 "learning_rate": learning_rate,
#             }
#             best_model = model
    
#     print(f"Best Hyperparameters: {best_hyperparams}")
#     print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    
#     return best_hyperparams, best_model
# ----------------------------------------------Train the Model--------------------------------------------------
# model = SimpleNN(first_layer_size = len(train_images[0]))
# model = SimpleNN_5layers(first_layer_size = len(train_images[0]))
model = ModulableNN(first_layer_size = len(train_images[0]), num_hidden_layers=5, neurons_per_layer=20)

criterion = nn.CrossEntropyLoss()  # Fonction de perte (Cross-Entropy Loss) good pour job de classification (0-9 digit out)
optimizer = optim.Adam(model.parameters(), lr=0.001) 
'''
# model.parameters sont les weight a optimizer & lr(learning rate) est la grosseur des pas dptimizzation
# Adaptive Moment Estimation (ADAM) => lr modulable avec le momentum de lamelioration du model
'''
# Nombre d'iteration de training sur le dataset
num_epochs = 100

for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0 # total returns de la fonction de perte
    correct_predictions = 0 # total corrections faites au poids
    total_samples = 0
    
    for images, labels in train_loader:
        # Test et quantification de la fonction de perte
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backpropagation et optimization
        optimizer.zero_grad() # RESET from last iteration
        loss.backward()  # Calcul du gradient via backpropagation
        optimizer.step()  # mise a jour des poids
        
        # Calculs pour stats...
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    epoch_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples * 100
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ----------------------------------------------Save the Trained Model--------------------------------------------------
# torch.save(model.state_dict(), os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name)

# ----------------------------------------------Test the Model--------------------------------------------------
# model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name)) # Load the saved model
model.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():  # pas de grad car pas dajustement de poids
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # valeur de reference
        correct_predictions += (predicted == labels).sum().item()  # nombre de prediction ok
        total_samples += labels.size(0)
accuracy = correct_predictions / total_samples * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# ----------------------------------------------Display Model--------------------------------------------------
# from torchsummary import summary
# model = SimpleNN(first_layer_size=len(train_images[0]))
# summary(model, input_size=(len(train_images[0]),))