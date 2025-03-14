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
import matplotlib.cm as cm
from tkinter import filedialog


class PlantDataManager:
    def __init__(self, data_folders, data_labels, Background_Folder):
        self.data_folders = data_folders
        self.data_label = data_labels
        self.Background_Folder = Background_Folder
        self.Background_data = self.get_intensity_data_from_folder(self.Background_Folder)

    def get_wavelength_data(self, base_dir):
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

    def moving_average(self, data, window_size=5):
        kernel = np.ones(window_size) / window_size  # Create a uniform kernel
        return np.convolve(data, kernel, mode='same')  # Apply convolution

    def Prepare_data(self, Plante_folder):
        BG_data = self.Background_data
        BG_data = np.mean(BG_data, axis=0)
        Wavelength_bins = self.get_wavelength_data(self.Background_Folder)

        Spectro_data = self.get_intensity_data_from_folder(Plante_folder)
        Spectro_data = [arr for arr in Spectro_data if np.all(np.max(arr) <= 63500)] # Remove saturated

        # range de nanometre a analyser
        # Cap_low, Cap_high = 300, 840
        # Cap_low, Cap_high = 645, 735 
        Cap_low, Cap_high = 420, 800
        low_index = np.argmin(np.abs(Wavelength_bins - Cap_low))
        high_index = np.argmin(np.abs(Wavelength_bins - Cap_high))

        BG_data = self.moving_average(BG_data, window_size=5) # Smoothen Background
        # BG_data = BG_data/np.max(BG_data) # Normalize Background
        Spectro_data = [arr/BG_data for arr in Spectro_data] # Divide Data by Background

        Spectro_data = [self.moving_average(arr, window_size=30) for arr in Spectro_data] # Smoothen Spectral Data
        
        Spectro_data = [arr[low_index:high_index] for arr in Spectro_data]
        self.Wavelength_bins_for_plot = Wavelength_bins[low_index:high_index]
        self.Background_data_for_plot = BG_data[low_index:high_index]

        Spectro_data = [arr / np.max(arr) for arr in Spectro_data] # Normalize Spectral Data
        # Spectro_data -= BG_data

        return Spectro_data

    def get_intensity_data_from_folder(self, base_dir):
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

    def load_data(self, plante_folder_paths):
        data = []
        labels = []
        for i, path in enumerate(plante_folder_paths):
            data.extend(self.Prepare_data(path))
            plant_name = i
            for j in range(len(data)-len(labels)):
                labels.append(plant_name)
        return np.array(data), np.array(labels)

    def train_model(self, num_epochs = 100):
        train_images, train_labels = self.load_data(self.data_folders) # Train image : 60 000*[784*[]] Train Label : 60 000*[]

        # Split train en pour garder un test set pour eviter le overfitting (test_size=0.1=>10%) (random_state=seed for reproductibility)
        X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=666)

        # initialisation des dataset pytorch
        train_dataset = PlantDataset(X_train, y_train)
        val_dataset = PlantDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # model = SimpleNN(first_layer_size = len(train_images[0]))
        # model = SimpleNN_5layers(first_layer_size = len(train_images[0]))
        self.model = ModulableNN(first_layer_size = len(train_images[0]), num_hidden_layers=5, neurons_per_layer=20, output_size=6)

        criterion = nn.CrossEntropyLoss()  # Fonction de perte (Cross-Entropy Loss) good pour job de classification (0-9 digit out)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
        '''
        # model.parameters sont les weight a optimizer & lr(learning rate) est la grosseur des pas dptimizzation
        # Adaptive Moment Estimation (ADAM) => lr modulable avec le momentum de lamelioration du model
        '''
        # Nombre d'iteration de training sur le dataset
        for epoch in range(num_epochs):
            self.model.train() 
            running_loss = 0.0 # total returns de la fonction de perte
            correct_predictions = 0 # total corrections faites au poids
            total_samples = 0
            
            for images, labels in train_loader:
                # Test et quantification de la fonction de perte
                outputs = self.model(images)
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

    def save_model(self, model_name = 'model_test.pth'):
        torch.save(self.model.state_dict(), os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name)

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Model", "*.pt;*.pth")])
        
        if file_path:  # Ensure a file was selected
            self.model.load_state_dict(torch.load(file_path))  # Load the saved model
            print(f"Model loaded from: {file_path}")
        else:
            print("No file selected. Model not loaded.")

    def test_model(self):
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():  # pas de grad car pas dajustement de poids
            for images, labels in self.val_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1) # valeur de reference
                correct_predictions += (predicted == labels).sum().item()  # nombre de prediction ok
                total_samples += labels.size(0)
        accuracy = correct_predictions / total_samples * 100
        print(f"Final Test Accuracy: {accuracy:.2f}%")
    
    def show_data(self, graph_type='all', show_source=True):
        Plants_data = []
        for folder in self.data_folders:
            Plants_data.append(self.Prepare_data(folder))
        # plant_color = ['b', 'g', 'r', 'c', 'm', 'y']

        num_plants = len(Plants_data)
        cmap = plt.colormaps.get_cmap('tab10')  # Get the colormap object

        plt.figure(figsize=(12, 6))
        for j, data_plant in enumerate(Plants_data):
            color = cmap(j / max(1, num_plants - 1))  # Normalize j for proper color mapping
            if graph_type == 'all':
                for i, data in enumerate(data_plant):
                    if i == 0:
                        plt.plot(self.Wavelength_bins_for_plot, data, label=self.data_label[j], color=color, linewidth=0.8, alpha=1)
                    plt.plot(self.Wavelength_bins_for_plot, data, color=color, linewidth=0.8, alpha=0.3)
            if graph_type == 'mean':
                data = np.mean(data_plant, axis=0)
                plt.plot(self.Wavelength_bins_for_plot, data, color=color, label=self.data_label[j])
        if show_source:
            plt.plot(self.Wavelength_bins_for_plot, self.Background_data_for_plot/np.max(self.Background_data_for_plot), label='Source', color = 'red')
        plt.title("Analyse des données spectrales par type de plante")
        plt.grid()
        plt.xlabel("Longueur d'onde (nm)")
        plt.ylabel("Intensité")
        plt.legend()
        plt.show()

    def show_data_with_weights(self, show_source=True):
        if not hasattr(self, 'model'):
            raise ValueError("Model is not trained yet. Please train the model first.")

        if not hasattr(self.model, 'layers') or not isinstance(self.model.layers[0], nn.Linear):
            raise ValueError("Model does not have a recognizable first fully connected layer.")

        # Extract first layer weights (take the absolute sum over all neurons to see importance)
        first_layer_weights = self.model.layers[0].weight.detach().cpu().numpy()
        importance_per_wavelength = np.abs(first_layer_weights).sum(axis=0)  # Sum absolute weights across neurons

        # Normalize the weights for visualization
        importance_per_wavelength /= np.max(importance_per_wavelength)

        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot spectral data
        for i, folder in enumerate(self.data_folders):
            data = np.mean(self.Prepare_data(folder), axis=0)  # Averaged spectral data per plant
            plt.plot(self.Wavelength_bins_for_plot, data, label=self.data_label[i], alpha=0.8)

        # Fill under the weight importance curve
        plt.fill_between(self.Wavelength_bins_for_plot, importance_per_wavelength, color='red', alpha=0.1, label="Importance des poids")

        if show_source:
            plt.plot(self.Wavelength_bins_for_plot, self.Background_data_for_plot/np.max(self.Background_data_for_plot), label='Source', color = 'black')

        plt.xlabel("Longueur d'onde (nm)")
        plt.ylabel("Intensité normalisée / importance des poids")
        plt.title("Analyse des données spectrales par type de plante avec l'importance des poids du modèle entrainé")
        plt.legend()
        plt.grid()

        plt.show()

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

class PlantDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
def main():
    # Load train & test data
    Plante_1_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Scindapsus_aureus_20ms\\"
    Plante_2_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Kalanchoe_daigremontianum_30ms\\"
    Plante_3_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Dieffenbachia_seguine_20ms\\"
    Plante_4_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Dracaena_fragrans_10ms\\"
    Plante_5_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Tradescantia_spathacea_top_20ms\\"
    Plante_6_folder = os.path.dirname(os.path.abspath(__file__)) +"\\Tradescantia_spathacea_bot_25ms\\"
    Plant_Folders = [Plante_1_folder, Plante_2_folder, Plante_3_folder, Plante_4_folder, Plante_5_folder, Plante_6_folder]

    plant_names = [
        'Scindapsus_aureus_20ms (Basic Plante)',
        'Kalanchoe_daigremontianum_30ms (Bébé démembrés)',
        'Dieffenbachia_seguine_20ms (Léopart)',
        'Dracaena_fragrans_10ms (Verte-Jaune)',
        'Tradescantia_spathacea_top_20ms (Côte Vert)',
        'Tradescantia_spathacea_bot_25ms (Côte Rouge)'
    ]

    Background_Folder = os.path.dirname(os.path.abspath(__file__)) +"\\Background_7ms_feuille_blanche\\"

    MyDataManager = PlantDataManager(Plant_Folders, plant_names, Background_Folder)

    MyDataManager.train_model()
    MyDataManager.test_model()

    # MyDataManager.show_data(graph_type='mean', show_source=True)
    MyDataManager.show_data_with_weights()

if __name__ == "__main__":
    main()