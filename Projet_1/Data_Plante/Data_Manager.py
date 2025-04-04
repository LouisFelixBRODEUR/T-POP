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
import itertools
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap


class PlantDataManager:
    def __init__(self, data_folders, data_labels, Background_Folder, Loaded_data_dict='No'):
        self.data_folders = data_folders
        self.data_label = data_labels
        self.Background_Folder = Background_Folder
        if Loaded_data_dict == 'No':
            self.data_dict = {}
        else:
            self.data_dict = Loaded_data_dict
        self.Background_data = self.get_intensity_data_from_folder(self.Background_Folder)
        self.plant_number = len(data_folders)

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
        self.Background_data_for_plot = self.Background_data
        self.Background_data_for_plot = np.mean(self.Background_data_for_plot, axis=0)
        self.Wavelength_bins_for_plot = self.get_wavelength_data(self.Background_Folder)

        Spectro_data = self.get_intensity_data_from_folder(Plante_folder)
        Spectro_data = [arr for arr in Spectro_data if np.all(np.max(arr) <= 63500)] # Remove saturated

        # range de nanometre a analyser
        # Cap_low, Cap_high = 300, 840
        # Cap_low, Cap_high = 645, 735 
        # Cap_low, Cap_high = 420, 700
        Cap_low, Cap_high = 420, 800
        low_index = np.argmin(np.abs(self.Wavelength_bins_for_plot - Cap_low))
        high_index = np.argmin(np.abs(self.Wavelength_bins_for_plot - Cap_high))

        self.Background_data_for_plot = self.moving_average(self.Background_data_for_plot, window_size=5) # Smoothen Background
        # self.Background_data_for_plot = self.Background_data_for_plot/np.max(self.Background_data_for_plot) # Normalize Background
        Spectro_data = [arr/self.Background_data_for_plot for arr in Spectro_data] # Divide Data by Background

        Spectro_data = [self.moving_average(arr, window_size=30) for arr in Spectro_data] # Smoothen Spectral Data
        
        Spectro_data = [arr[low_index:high_index] for arr in Spectro_data]
        self.Wavelength_bins_for_plot = self.Wavelength_bins_for_plot[low_index:high_index]
        self.Background_data_for_plot = self.Background_data_for_plot[low_index:high_index]

        Spectro_data = [arr / np.max(arr) for arr in Spectro_data] # Normalize Spectral Data
        # Spectro_data -= self.Background_data_for_plot
    
        return Spectro_data

    def get_intensity_data_from_folder(self, base_dir):
        if base_dir in self.data_dict:
            return self.data_dict[base_dir]
        else:
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
            self.data_dict[base_dir] = intensity_data
            return intensity_data

    def load_data(self, plante_folder_paths):
        data = []
        labels = []
        for i, path in enumerate(plante_folder_paths):
            data.extend(self.Prepare_data(path))
            plant_name = i
            for j in range(len(data)-len(labels)):
                labels.append(plant_name)
        data, labels = np.array(data), np.array(labels)
        # --------------------------------------------
        # if len(data) > 100: #Coupe a 100 si plus que 100
        #     indices = np.random.choice(len(data), size=100, replace=False)
        #     data = data[indices]
        #     labels = labels[indices]
        # --------------------------------------------
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)
        new_data = []
        new_labels = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(label_indices, size=min_count, replace=False)
            new_data.append(data[sampled_indices])
            new_labels.append(labels[sampled_indices])
        new_data = np.concatenate(new_data)
        new_labels = np.concatenate(new_labels)
        # --------------------------------------------

        return new_data, new_labels

    def train_plant_detector(self, num_epochs = 100, show_progress=False):
        train_images, train_labels = self.load_data(self.data_folders) # Train image : 60 000*[784*[]] Train Label : 60 000*[]

        # Split train en pour garder un test set pour eviter le overfitting (test_size=0.1=>10%) (random_state=seed for reproductibility)
        X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=666)

        # initialisation des dataset pytorch
        train_dataset = PlantDataset(X_train, y_train)
        val_dataset = PlantDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        self.model = ModulableNN(first_layer_size = len(train_images[0]), num_hidden_layers=5, neurons_per_layer=20, output_size=self.plant_number)

        criterion = nn.CrossEntropyLoss()  # Fonction de perte (Cross-Entropy Loss) good pour job de classification (0-9 digit out)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001) 

        loss_history = []
        accuracy_history = []

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
                outputs = self.model(images) # y_pred
                loss = criterion(outputs, labels)
                optimizer.zero_grad() # RESET from last iteration

                # Backpropagation et optimization
                loss.backward()  # Calcul du gradient via backpropagation
                optimizer.step()  # mise a jour des poids
            
                # Calculs pour stats...
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            
            epoch_loss = running_loss / total_samples
            loss_history.append(epoch_loss)
            accuracy = correct_predictions / total_samples * 100
            accuracy_history.append(accuracy)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if show_progress:
            loss_history = np.array(loss_history) / np.max(loss_history)
            accuracy_history = np.array(accuracy_history) / np.max(accuracy_history)
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-', label = 'loss')
            plt.plot(range(1, num_epochs + 1), accuracy_history, marker='o', linestyle='-', label = 'accuracy')
            plt.xlabel("Epoch")
            plt.ylabel("Loss and Accuracy (Normalized)")
            plt.title("Loss Progression During Training")
            plt.grid()
            plt.legend(loc = 2)
            plt.show()

    def save_model(self, model_name = 'model_test.pth'):
        torch.save(self.model.state_dict(), os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name)

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Model", "*.pt;*.pth")])
        
        if file_path:  # Ensure a file was selected
            self.model.load_state_dict(torch.load(file_path))  # Load the saved model
            print(f"Model loaded from: {file_path}")
        else:
            print("No file selected. Model not loaded.")

    def test_plant_detector(self, all_accuracy=False, return_accuracy=False):
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        plant_correct = np.zeros(self.plant_number)
        plant_total = np.zeros(self.plant_number)
        with torch.no_grad():  # pas de grad car pas dajustement de poids
            for images, labels in self.val_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1) # valeur de reference
                correct_predictions += (predicted == labels).sum().item()  # nombre de prediction ok
                total_samples += labels.size(0)

                for i in range(len(labels)):
                    plant_total[labels[i].item()] += 1
                    if predicted[i] == labels[i]:
                        plant_correct[labels[i].item()] += 1
        accuracy = correct_predictions / total_samples * 100
        print(f"Final Test Accuracy: {accuracy:.2f}%")

        if return_accuracy:
            return accuracy

        if all_accuracy:
            for i in range(self.plant_number):
                if plant_total[i] > 0:
                    plant_acc = (plant_correct[i] / plant_total[i]) * 100
                    print(f"Accuracy for {self.data_label[i]}: {plant_acc:.2f}%")
    
    def show_data(self, graph_type='all', show_source=True):
        Plants_data = []
        for folder in self.data_folders:
            Plants_data.append(self.Prepare_data(folder))

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
        plt.legend(loc = 2)
        plt.show()

    def show_source(self):
        Wavelength_bins = self.get_wavelength_data(self.Background_Folder)
        Cap_low, Cap_high = 420, 800
        low_index = np.argmin(np.abs(Wavelength_bins - Cap_low))
        high_index = np.argmin(np.abs(Wavelength_bins - Cap_high))
        Wavelength_bins = Wavelength_bins[low_index:high_index]

        Background_Folders = [
            os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Background_7ms_feuille_blanche\\",
            os.path.dirname(os.path.abspath(__file__)) +"\\Session2\\Background_30ms_feuille_blanche\\"
        ]
        BG_data_list = []
        for BG_fold in Background_Folders:
            BG_data = self.get_intensity_data_from_folder(BG_fold)
            BG_data = np.mean(BG_data, axis=0)
            BG_data = self.moving_average(BG_data, window_size=5) # Smoothen Background
            BG_data = BG_data[low_index:high_index]
            BG_data = BG_data/np.max(BG_data)
            BG_data_list.append(BG_data)

        plt.figure(figsize=(12, 6))
        plt.plot(Wavelength_bins, BG_data_list[0], label='LED Blanche')
        plt.plot(Wavelength_bins, BG_data_list[1], label='LED RGB')
        plt.grid(True)
        plt.xlabel("Longueur d'onde (nm)", fontsize=25)
        plt.ylabel("Intensité (normalisée)", fontsize=25)
        plt.gca().axes.tick_params(axis='both', which='major', labelsize=20)
        # plt.xticks(range(min(nb_de_plante_dans_le_dataset), max(nb_de_plante_dans_le_dataset) + 1, 1))
        plt.legend(loc='upper right', fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.995,
            bottom=0.08,
            left=0.055,
            right=0.995)
        plt.show()

        # plt.xlabel("Nombre de plantes dans l'ensemble de données", fontsize=25)
        # plt.ylabel("Précision (%)", fontsize=25)
        # plt.gca().axes.tick_params(axis='both', which='major', labelsize=20)
        # plt.xticks(range(min(nb_de_plante_dans_le_dataset), max(nb_de_plante_dans_le_dataset) + 1, 1))
        # plt.legend(loc='lower left', fontsize=15)
        # plt.tight_layout()
        # plt.subplots_adjust(
        #     top=0.995,
        #     bottom=0.085,
        #     left=0.06,
        #     right=0.995)
        # plt.grid(True)
        # plt.show()

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
        num_plants = len(self.data_folders)
        norm = Normalize(vmin=0, vmax=num_plants - 1)

        colormap = plt.get_cmap('tab20', num_plants) # hsv tab20 cubehelix gist_ncar
        # cmap = plt.get_cmap('tab20')
        # dark_cmap = LinearSegmentedColormap.from_list('dark_tab20', [cmap(i) for i in range(0, 20, 2)], N=20)
        # colormap = dark_cmap

        for i, folder in enumerate(self.data_folders):
            color = colormap(norm(i))  # Ensure unique colors
            color = tuple(c * 0.8 for c in color[:3]) + (color[3],)
            data = np.mean(self.Prepare_data(folder), axis=0)  # Averaged spectral data per plant
            plt.plot(self.Wavelength_bins_for_plot, data, label=self.data_label[i], color=color)

        # Fill under the weight importance curve
        plt.fill_between(self.Wavelength_bins_for_plot, importance_per_wavelength, color='red', alpha=0.1, label="Importance des poids")

        if show_source:
            plt.plot(self.Wavelength_bins_for_plot, self.Background_data_for_plot/np.max(self.Background_data_for_plot), label='Source LED', color = 'black', linestyle="--")

        plt.xlabel("Longueur d'onde (nm)", fontsize=25)
        plt.ylabel("Intensité et Importance des poids", fontsize=25)
        # plt.title("Analyse des données spectrales par type de plante avec l'importance des poids du modèle entrainé")
        plt.gca().axes.tick_params(axis='both', which='major', labelsize=20)
        # plt.legend(loc = 'lower right', fontsize=10)

        plt.legend(loc='upper left', bbox_to_anchor=(0.99, 1.01), fontsize=19)
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.985,
            bottom=0.08,
            left=0.055,
            right=0.71)

        plt.autoscale(enable=True, axis='both', tight=True)

        plt.grid()

        plt.show()
    
class ModulableNN(nn.Module):
    def __init__(self, first_layer_size=3648, num_hidden_layers=5, neurons_per_layer=20, output_size=2):
        super(ModulableNN, self).__init__()
        
        self.input_size = first_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.output_size = output_size
        
        self.layers = nn.ModuleList()
        
        # Input
        self.layers.append(nn.Linear(first_layer_size, neurons_per_layer))
        self.layers.append(nn.ReLU())  # fonction activation
        
        # Hidden
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.ReLU())  # fonction activation
        
        # Output
        self.layers.append(nn.Linear(neurons_per_layer, output_size))
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        # x = self.softmax(x)
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
    Plant_Folders = [
        os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Scindapsus_aureus_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Kalanchoe_daigremontianum_30ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Dieffenbachia_seguine_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Dracaena_fragrans_10ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Tradescantia_spathacea_top_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Tradescantia_spathacea_bot_25ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Euphorbia_milii_50ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Pachypodium_rosulatum_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Monstera_deliciosa_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Ficus_lyrata_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Begonia_gryphon_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Iresine_herbstii_50ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Spathiphyllum_cochlearispathum_35ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Philodendron_atabapoense_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Oldenlandia_affinis_20ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Dracaena_fragrans_30ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Dracaena_trifasciata_10ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Philodendron_melanochrysum_10ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Ficus_alii_40ms\\",
        os.path.dirname(os.path.abspath(__file__)) +"\\Session4\\Specialty_aglaonema_20ms\\"]
    
    plant_names = [
        'Scindapsus aureus',
        'Kalanchoe daigremontianum',
        'Dieffenbachia seguine',
        'Dracaena fragrans',
        'Tradescantia spathacea (dessus)',
        'Tradescantia spathacea (dessous)',
        'Euphorbia milii',
        'Pachypodium rosulatum',
        'Monstera deliciosa',
        'Ficus lyrata',
        'Begonia gryphon',
        'Iresine herbstii',
        'Spathiphyllum cochlearispathum',
        'Philodendron atabapoense',
        'Oldenlandia affinis',
        'Dracaena fragrans',
        'Dracaena trifasciata',
        'Philodendron melanochrysum',
        'Ficus alii',
        'Specialty aglaonema']
    
    # mode = 'train_20'
    # mode = 'train_20_saved'
    mode = 'train_show_weight'
    # mode = 'show_data_graph'
    # mode = 'show_source_graph'
    # mode = '2_plantes_blanc'
    # mode = '2_plantes_RGB'

    Background_Folder = os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Background_7ms_feuille_blanche\\"
    nb_de_plante_dans_le_dataset = list(range(1, len(plant_names) + 1))

    if mode == 'train_20':
        accuracy_NN = []
        Loaded_data_dict = 'No'
        # for nb_de_plante in [3]:
        for nb_de_plante in nb_de_plante_dans_le_dataset:
            print(f'Testing for {nb_de_plante} plants')
            sum_accuracy = []
            for test_nb in range(100):
                print(f'Test {test_nb}/10 ({nb_de_plante} plants in dataset)')
                random_values = random.sample(list(range(0,len(plant_names))), nb_de_plante) #Choisi des plante au hasard dans le set
                MyDataManager = PlantDataManager([Plant_Folders[i] for i in random_values], [plant_names[i] for i in random_values], Background_Folder, Loaded_data_dict=Loaded_data_dict)
                MyDataManager.train_plant_detector(num_epochs=50*nb_de_plante)
                # MyDataManager.train_plant_detector(num_epochs=3)
                sum_accuracy.append(MyDataManager.test_plant_detector(return_accuracy=True))
                Loaded_data_dict = MyDataManager.data_dict
                del MyDataManager
            accuracy_NN.append(np.mean(sum_accuracy))
        print([round(float(N), 3) for N in accuracy_NN])

    if mode == 'train_20_saved':
        # accuracy_NN = [100.0, 99.256, 89.267, 92.61, 92.709, 86.857, 81.993, 85.991, 89.384, 89.439, 84.809, 88.787, 90.452, 89.988, 88.929, 89.948, 89.155, 90.79, 92.111, 89.53]#10 fois 100 300
        # accuracy_NN = [100.0, 96.667, 94.644, 87.842, 74.352, 77.93, 78.581, 83.816, 78.852, 88.812, 89.708, 78.706, 87.17, 83.84, 87.857, 83.116, 89.59, 83.824, 89.218, 86.117]#10 fois 100
        accuracy_NN = [100.0, 98.154, 91.325, 87.268, 82.861, 81.841, 85.268, 86.253, 82.322, 82.646, 84.962, 85.561, 84.814, 85.022, 85.184, 85.767, 87.808, 86.782, 86.381, 87.553]#100 fois 100
    if mode == 'train_20_saved' or mode == 'train_20':
        accuracy_fit_mean = [100.0, 91.849, 88.022, 84.169, 82.075, 78.37, 77.876, 75.527, 72.425, 71.977, 72.443, 70.163, 70.703, 69.358, 68.729, 67.306, 66.798, 66.064, 65.329, 64.865]
        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(nb_de_plante_dans_le_dataset[0:len(accuracy_NN)], accuracy_NN, marker='o', linestyle='-', label='Précision FCN')
        plt.plot(nb_de_plante_dans_le_dataset[0:len(accuracy_fit_mean)], accuracy_fit_mean, marker='o', linestyle='-', label='Précision NCC')
        plt.xlabel("Nombre de plantes dans l'ensemble de données", fontsize=25)
        plt.ylabel("Précision (%)", fontsize=25)
        plt.gca().axes.tick_params(axis='both', which='major', labelsize=20)
        plt.xticks(range(min(nb_de_plante_dans_le_dataset), max(nb_de_plante_dans_le_dataset) + 1, 1))
        plt.legend(loc='lower left', fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.995,
            bottom=0.085,
            left=0.06,
            right=0.995)
        plt.grid(True)
        plt.show()

    MyDataManager = PlantDataManager(Plant_Folders, plant_names, Background_Folder)
    if mode == 'train_show_weight':
        MyDataManager.train_plant_detector(num_epochs=1000, show_progress=True)
        MyDataManager.test_plant_detector(all_accuracy=False)
        MyDataManager.show_data_with_weights()

    if mode == 'show_data_graph':
        MyDataManager.show_data(graph_type='all', show_source=True)
    if mode == 'show_source_graph':
        MyDataManager.show_source()
    if mode == '2_plantes_RGB':
        Plant_Folders = [
            os.path.dirname(os.path.abspath(__file__)) +"\\Session2\\Scindapsus_aureus_100ms\\",
            os.path.dirname(os.path.abspath(__file__)) +"\\Session2\\Kalanchoe_daigremontianum_100ms\\"]
        plant_names = [
            'Scindapsus aureus',
            'Kalanchoe daigremontianum']
        Background_Folder = os.path.dirname(os.path.abspath(__file__)) +"\\Session2\\Background_30ms_feuille_blanche\\"
        MyDataManager = PlantDataManager(Plant_Folders, plant_names, Background_Folder)
        MyDataManager.train_plant_detector(num_epochs=70, show_progress=True)
        MyDataManager.test_plant_detector(all_accuracy=False)
        MyDataManager.show_data_with_weights()
    if mode == '2_plantes_blanc':
        Plant_Folders = [
            os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Scindapsus_aureus_20ms\\",
            os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Kalanchoe_daigremontianum_30ms\\"]
        plant_names = [
            'Scindapsus aureus',
            'Kalanchoe daigremontianum']
        MyDataManager = PlantDataManager(Plant_Folders, plant_names, Background_Folder)
        MyDataManager.train_plant_detector(num_epochs=70, show_progress=True)
        MyDataManager.test_plant_detector(all_accuracy=False)
        MyDataManager.show_data_with_weights()

if __name__ == "__main__":
    main()