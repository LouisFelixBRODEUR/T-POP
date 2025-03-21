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
        Cap_low, Cap_high = 420, 800
        # Cap_low, Cap_high = 435, 800
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
        return np.array(data), np.array(labels)

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
        for i, folder in enumerate(self.data_folders):
            color = colormap(norm(i))  # Ensure unique colors
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

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.985,
            bottom=0.08,
            left=0.055,
            right=0.77)

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
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(-1, self.input_size)
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
        'Tradescantia_spathacea (dessous)',
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
    
    Background_Folder = os.path.dirname(os.path.abspath(__file__)) +"\\Session3\\Background_7ms_feuille_blanche\\"

    # accuracy_NN = []
    # nb_de_plante_dans_le_dataset = list(range(1, len(plant_names) + 1))
    # Loaded_data_dict = 'No'
    # for nb_de_plante in nb_de_plante_dans_le_dataset:
    #     print(f'Testing for {nb_de_plante} plants')
    #     sum_accuracy = []
    #     for test_nb in range(10):
    #         print(f'Test {test_nb}/10 ({nb_de_plante} plants in dataset)')
    #         random_values = random.sample(list(range(0,len(plant_names))), nb_de_plante) #Choisi des plante au hasard dans le set
    #         MyDataManager = PlantDataManager([Plant_Folders[i] for i in random_values], [plant_names[i] for i in random_values], Background_Folder, Loaded_data_dict=Loaded_data_dict)
    #         MyDataManager.train_plant_detector(num_epochs=50*nb_de_plante)
    #         # MyDataManager.train_plant_detector(num_epochs=3)
    #         sum_accuracy.append(MyDataManager.test_plant_detector(return_accuracy=True))
    #         Loaded_data_dict = MyDataManager.data_dict
    #         del MyDataManager
    #     accuracy_NN.append(np.mean(sum_accuracy))
    # print(accuracy_NN)

    accuracy_NN = [np.float64(100.0), np.float64(99.2560975609756), np.float64(89.26666666666667), np.float64(92.60956790123456), np.float64(92.708885479672), np.float64(86.85681818181817), np.float64(81.99332858975608), np.float64(85.99078540590638), np.float64(89.38433243544735), np.float64(89.43931396171963), np.float64(84.80928255340828), np.float64(88.78736053901918), np.float64(90.45224508955985), np.float64(89.98757380758937), np.float64(88.9289277426837), np.float64(89.94766112574332), np.float64(89.1550713712729), np.float64(90.79045922762613), np.float64(92.1112851028291), np.float64(89.52978056426333)]
    accuracy_fit_mean = [np.float64(100.0), np.float64(91.848968331081), np.float64(88.02206978879114), np.float64(84.1692104260871), np.float64(82.07475183456529), np.float64(78.36975368052478), np.float64(77.87573437371694), np.float64(75.52681093884098), np.float64(72.42503440544938), np.float64(71.97732384827768), np.float64(72.44295844301737), np.float64(70.16321629294151), np.float64(70.70338892546624), np.float64(69.35790727507292), np.float64(68.72889897196383), np.float64(67.30563177739639), np.float64(66.79756944256683), np.float64(66.06421356273192), np.float64(65.3292203935755), np.float64(64.86466165413535)]
    nb_de_plante_dans_le_dataset = list(range(1, len(accuracy_NN) + 1))


    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(nb_de_plante_dans_le_dataset, accuracy_NN, marker='o', linestyle='-', label='accuracy_NN')
    plt.plot(nb_de_plante_dans_le_dataset, accuracy_fit_mean, marker='o', linestyle='-', label='accuracy_fit_mean')
    plt.xlabel("Nombre de plantes dans le dataset", fontsize=25)
    plt.ylabel("Précision (%)", fontsize=25)
    plt.gca().axes.tick_params(axis='both', which='major', labelsize=20)
    plt.xticks(range(min(nb_de_plante_dans_le_dataset), max(nb_de_plante_dans_le_dataset) + 1, 1))
    plt.legend(loc='lower left', fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.995,
        bottom=0.085,
        left=0.06,
        right=0.995)
    plt.grid(True)
    plt.show()

    # MyDataManager = PlantDataManager(Plant_Folders, plant_names, Background_Folder)

    # MyDataManager.train_plant_detector(num_epochs=1000, show_progress=True)
    # MyDataManager.test_plant_detector(all_accuracy=True)
    # MyDataManager.show_data_with_weights()

    # MyDataManager.show_data(graph_type='all', show_source=True)

if __name__ == "__main__":
    main()