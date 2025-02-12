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

model_name = 'model_test.pth'

# ----------------------------------------------Load and Preprocess the Data--------------------------------------------------
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    images = data.iloc[:, 1:].values.astype(np.float32)  # after 1rst row = data
    labels = data.iloc[:, 0].values.astype(np.int64)  # 1er row = identification
    images = images / 255.0 # Normalization des pix
    
    return images, labels

# Conversion du data en pytorch dataset (tensor)
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load train & test data
test_csv_path = os.path.dirname(os.path.abspath(__file__))+"\\mnist_test.csv"
train_csv_path = os.path.dirname(os.path.abspath(__file__))+"\\mnist_train.csv"
train_images, train_labels = load_data(train_csv_path)
test_images, test_labels = load_data(test_csv_path)

# Split train en pour garder un test set pour eviter le overfitting (test_size=0.1=>10%) (random_state=seed for reproductibility)
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=666)

# initialisation des dataset pytorch
train_dataset = MNISTDataset(X_train, y_train)
val_dataset = MNISTDataset(X_val, y_val)
test_dataset = MNISTDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----------------------------------------------Define the Neural Network Model--------------------------------------------------
# fully connected neural network (FCN) pour plus facilement classser les images
# Rectified Linear Unit (ReLU) en Fonction dactivation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # input layer 
        self.fc2 = nn.Linear(128, 64)  # deuxieme layer
        self.fc3 = nn.Linear(64, 10)  # output layer (10 out pour digit 0-9)
        self.relu = nn.ReLU()  # fonction dactivation
    # implementation des layers
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # linput dun FCN doit etre un vecteur 
        x = self.relu(self.fc1(x))  # input layer 
        x = self.relu(self.fc2(x))  # deuxieme layer
        x = self.fc3(x)  # output layer
        return x

# ----------------------------------------------Train the Model--------------------------------------------------
model = SimpleNN() 
criterion = nn.CrossEntropyLoss()  # Fonction de perte (Cross-Entropy Loss) good pour job de classification (0-9 digit out)
optimizer = optim.Adam(model.parameters(), lr=0.001) 
'''
# model.parameters sont les weight a optimizer & lr(learning rate) est la grosseur des pas dptimizzation
# Adaptive Moment Estimation (ADAM) => lr modulable avec le momentum de lamelioration du model
'''
# Nombre d'iteration de training sur le dataset
num_epochs = 5

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
torch.save(model.state_dict(), os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name)

# ----------------------------------------------Test the Model--------------------------------------------------
model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name)) # Load the saved model
model.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():  # pas de grad car pas dajustement de poids
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # valeur de reference
        correct_predictions += (predicted == labels).sum().item()  # nombre de prediction ok
        total_samples += labels.size(0)
accuracy = correct_predictions / total_samples * 100
print(f"Test Accuracy: {accuracy:.2f}%")