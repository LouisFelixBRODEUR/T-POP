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

model_name = 'model_test_20.pth'

# ----------------------------------------------Load and Preprocess the Data--------------------------------------------------
# Load the CSV files
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    images = data.iloc[:, 1:].values.astype(np.float32)  # Extract all pixels
    labels = data.iloc[:, 0].values.astype(np.int64)  # Extract the labels (first column)
    
    # Normalize the images (scaling pixel values to range [0, 1])
    images = images / 255.0
    
    return images, labels

# Convert the data into PyTorch Dataset format
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load the training and test data
test_csv_path = os.path.dirname(os.path.abspath(__file__))+"\\mnist_test.csv"
train_csv_path = os.path.dirname(os.path.abspath(__file__))+"\\mnist_train.csv"
train_images, train_labels = load_data(train_csv_path)
test_images, test_labels = load_data(test_csv_path)

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Create PyTorch datasets and dataloaders
train_dataset = MNISTDataset(X_train, y_train)
val_dataset = MNISTDataset(X_val, y_val)
test_dataset = MNISTDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----------------------------------------------Define the Neural Network Model--------------------------------------------------

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 10)  # Output layer (10 classes for digits 0-9)
        self.relu = nn.ReLU()  # Activation function
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.relu(self.fc1(x))  # First layer
        x = self.relu(self.fc2(x))  # Second layer
        x = self.fc3(x)  # Output layer
        return x

# ----------------------------------------------Train the Model--------------------------------------------------
# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Nombre d<iteration de training sur le dataset
num_epochs = 20

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the error
        optimizer.step()  # Update the weights
        
        running_loss += loss.item() * images.size(0)  # Track the loss
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)
    
    epoch_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples * 100
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ----------------------------------------------Save the Trained Model--------------------------------------------------
# Save the trained model
torch.save(model.state_dict(), os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name)

# ----------------------------------------------Test the Model--------------------------------------------------
# Load the saved model (for testing)
model.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+ '\\' + model_name))
model.eval()  # Set the model to evaluation mode

# Evaluate the model on the test set
correct_predictions = 0
total_samples = 0

with torch.no_grad():  # No need to track gradients during testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)

accuracy = correct_predictions / total_samples * 100
print(f"Test Accuracy: {accuracy:.2f}%")