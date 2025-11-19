"""
File: dl-algorithm.py
Author: Sam Moss, Josh White, Jack Wainwright
Created: 2025-09-28
Last Updated: 2025-09-28

Description:
This is a placeholder file
    
Usage:
    python dl-lstm.py

Notes:
    - LSTM algorim is a subset of the RNN Alogrithm
    - May need more stat types

"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Below are the RNN deep learning libraries 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Constants 
group_dir = "C:/Users/josh/OneDrive - Bath Spa University/3rd Year/CreatingIOT/iot-activity-recognition/data/group"
stations_path = "C:/Users/josh/OneDrive - Bath Spa University/3rd Year/CreatingIOT/iot-activity-recognition/data/public/train_stations_GB.csv"


all_files = os.listdir(group_dir)

# constant model tweaking variables
STATIONARY_SPEED = 1
WALKING_SPEED = 5
RUNNING_SPEED = 20
IN_VEHICLE_SPEED = 52

# Define the field names to be used from the CSV files
fieldNames = ['time', 'Latitude', 'Longitude', 'Altitude (m)', 'Speed (km/h)', 'Total Distance (km)' ] 

# Load station data
stationsData = pd.read_csv(stations_path)
stationsData.columns = ['id', 'name', 'norm', 'uic', 'latitude', 'longitude', 'station_id', 'country', 'time_zone', 'is_city', 'is_main_station', 'is_airport', 'entur_id', 'entur_is_enabled']

# Create DataFrame for stations
stations = pd.DataFrame(stationsData)




def loadFiles(group_dir, fieldNames):
    
    csv_files = []
    dataFrames = []

    # Get list of all csv files in the directory
    index = 0
    while index < len(all_files):
        file = all_files[index]
        if file.endswith('.csv'):
            # print("Found file: ", file)
            csv_files.append(file)
        index += 1

    index = 0
    # A while loop to go through each file, load it to the dataframe list 
    while index < len(csv_files):
        file = csv_files[index]
        file_path = os.path.join(group_dir, file)
        pima = pd.read_csv(file_path, sep='\t', encoding='utf-16')
        dataFrames.append(pima)
        print("Loaded the file ", file)
        index += 1

    # Concatenate all dataframes into a single dataframe
    pima = pd.concat(dataFrames, ignore_index=True)

    # Convert speed to a numerical value
    pima['Speed (km/h)'] = pd.to_numeric(pima['Speed (km/h)'], errors='coerce')
   
    #TODO - Check the time format in the csv files - currently mixed values. Need to standardised and then converted to numeric feature so the SVM model can use it
    #pima['time'] = pd.to_numeric(pima['time'], errors='coerce')

    return pima


# Classifies the activity based on speed
def classify_activity(speed_kmh):
    if pd.isna(speed_kmh):
        return 'unknown'
    if speed_kmh < STATIONARY_SPEED:
        return 'stationary'
    elif speed_kmh < WALKING_SPEED:
        return 'walking'
    elif speed_kmh < RUNNING_SPEED:
        return 'running'
    elif speed_kmh < IN_VEHICLE_SPEED:
        return 'in_vehicle' 
    else:
        return 'on_train'  
    
    
#Classifies the activities again - However backwards on the dataframe to refine those classifications that may be incorrect
def refine_classification(pima):
    # Create a new column for refined labels
    pima['label_refined'] = pima['label']
    
    # Refine classification by iterating backward on the dataframe
    index = len(pima) - 2
    while index >= 0:
        # Check for transition from in_vehicle to walking
        if pima.iloc[index]['label_refined'] == 'walking' and pima.iloc[index+1]['label_refined'] == 'in_vehicle':
        # consider if the current activity walking was actually in_vehicle
            if pima.iloc[index]['Speed (km/h)'] < 5:
                pima.iloc[index, pima.columns.get_loc('label_refined')] = 'in_vehicle'
        elif pima.iloc[index]['label_refined'] == 'running' and pima.iloc[index+1]['label_refined'] == 'in_vehicle':
            # consider if the current activity running was actually in_vehicle
            if pima.iloc[index]['Speed (km/h)'] > 10: 
                 pima.iloc[index, pima.columns.get_loc('label_refined')] = 'in_vehicle'
        index -= 1
    return pima


##========================================================
#   NOTE: EVERYTHING BELOW THIS LINE HAS/WILL BE CHANGED 
#   
#   Machine Learning algorithm needs to be adjusted to
#   Deep Learning
#
##========================================================

def prepare_data(df):
    # Create brand new column - averageSpeed
    df['averageSpeed'] = df['Speed (km/h)'].rolling(window=5, min_periods=1).mean()
    df['label'] = df['averageSpeed'].apply(classify_activity)
    df = refine_classification(df)

    features = ['Latitude', 'Longitude', 'Altitude (m)', 'Speed (km/h)', 'Total Distance (km)', 'averageSpeed']
    
    # x is the input (feature labels e.g. Latitude)
    x = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # y is the target output (running, walking, in_vehicle, etc)
    y = df['label_refined']
    return x, y 

def is_nearest(lat, lon, radius_km=1.0):
    EARTHS_RADIUS = 6371
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(stations['latitude'].values)
    lon2 = np.radians(stations['longitude'].values)

    deltalat = lat2 - lat1 
    deltalon = lon2 - lon1
    squared_sine = np.sin(deltalat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(deltalon/2)**2
    angular_distance = 2 * np.arcsin(np.sqrt(squared_sine))
    km = EARTHS_RADIUS * angular_distance

    if np.any(km <= radius_km):
        nearest_station_id = stations['id'].values[km <= radius_km][0]
        return True, nearest_station_id
    return False, None


# Classifies whether or not they are on train based on previous station
def clasifyOnTrain(pima, onTrainIndex, previousStationId):
    numOfEntries = onTrainIndex
    onTrain = False
    stationId = 0
    # print("Current Index = ", onTrainIndex)
    # Check back to determine whether coords were at a station
    while (onTrainIndex > 0 and onTrain == False):
        isAtTrainStn, ClosestStationId = is_nearest(pima.iloc[onTrainIndex]['Latitude'], pima.iloc[onTrainIndex]['Longitude'], 1.0) # sets 1 km radius - how close to station to be considered at station 
        if (isAtTrainStn == True):
            stationId = ClosestStationId
            onTrain = True
        onTrainIndex -= 1
    
    # When location was at a station then work to most recent entry to set on train
    # but only when not the destination station i.e. previousStationId = 0 or station is the same
    if (onTrain == True and previousStationId != 0 and stationId != previousStationId):
        index = onTrainIndex
        while(index < numOfEntries):
            pima.iloc[index, pima.columns.get_loc('label_refined')] = 'on_train'
            index += 1
    return onTrainIndex, stationId

# This class is for RNN model 'windows' (pieces of data)
# The Alogrithm will analyse windows for patterns for predictions
# OpenAI (2025) ChatGPT 4o: “how do I implement window analysis for LSTM models using pytorch DL models sequence dataset and output a prediction”
# errezeta (2021) ‘Custom dataset for time-series data for an LSTM model’, PyTorch Forums, 14 October. Available at: https://discuss.pytorch.org/t/custom-dataset-for-time-series-data-for-an-lstm-model/134275 (Accessed: 1 November 2025).
class SequenceDataset(Dataset):
    
    # NOTE: Change window size to tweak the model
    def __init__(self, x, y, window_size=20): # NOTE: also change if your hardware is weak when testing model
        self.X = x
        self.y = y
        self.window_size = window_size
        self.samples = self._create_sequences()

    # Build time windows
    def _create_sequences(self):
        seqs = []
        for i in range(len(self.X) - self.window_size):
            x_seq = self.X[i:i + self.window_size]
            y_seq = self.y[i + self.window_size - 1]
            seqs.append((x_seq, y_seq))
        return seqs

    # Tell pyTorch how many windows have been created
    def __len__(self):
        return len(self.samples)

    # Return specific window along with label when the model asks for it
    def __getitem__(self, idx):
        x_seq, y = self.samples[idx]
        return torch.tensor(x_seq.values, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Human Activity Recognition prediction class
# OpenAI (2025) ChatGPT 4o: “how do I implement window analysis for RNN models using pytorch DL models LSTM sequence dataset and output a prediction”
class HAR_LSTM(nn.Module):
    
    # Run once function to read time series data
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.3):
        super(HAR_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    # Make prediction (out) and return
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep
        out = self.fc(out)
        return out

# Train model using training data
def train_model(model, dataloader, criterion, optimizer, device):
    # Set model to training mode
    model.train()
    total_loss = 0
    
    # Training loop for PyTorch
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        
        # prediction (output) and loss computation
        output = model(xb)
        loss = criterion(output, yb)
        
        # Backward propogation (similar to rule based backward chaining)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Check how well the model is performing
def eval_model(model, dataloader, device):
    
    # Switch model to eval mode
    model.eval()
    y_true, y_pred = [], []
    
    # Pytorch evaluation loop
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
    return np.array(y_true), np.array(y_pred)

# Visulise results with a confusion matrix
def visualize_results(y_true, y_pred, label_classes):
    plt.style.use('seaborn-v0_8')
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_classes, yticklabels=label_classes)
    plt.title("Confusion Matrix - LSTM Human Activity Recognition")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    print(classification_report(y_true, y_pred, target_names=label_classes))


# Entrypoint fucnction
def main():
    
    # USE CUDA (Nvidia) else use cpu iGPU
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # RTX 50 series GPUs currently do not work with pyTorch CUDA
    # Temporary set device as cpu
    device = 'cpu'
    print(f"Using device: {device}")

    # Load data
    df = loadFiles(group_dir, fieldNames)
    X, y = prepare_data(df)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_classes = le.classes_

    # Normalize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train Test Split Data (80 / 20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


    # Dataset loaders w/ PyTorch
    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Model Setup using LSTM functions 
    input_dim = X.shape[1]
    num_classes = len(label_classes)
    model = HAR_LSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train model
    for epoch in range(1, 16):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/15] - Loss: {loss:.4f}")

    # Evaluate Model
    y_true, y_pred = eval_model(model, test_loader, device)
    visualize_results(y_true, y_pred, label_classes)
    
    report = classification_report(y_true, y_pred, target_names=label_classes, digits=4)
    print(report)


if __name__ == "__main__":
    main()