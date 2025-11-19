"""
File: ml-svm.py
Created: 2025-10-08
Last Updated: 2025-10-11

Description:
    Support Vector Machines (SVM) Algorithm for Human Activity Recognition Task

Usage:
    python ml-svm.py

Notes:
    - Completion TBC
    - Peer Review TBC
"""



import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
## Imports for Support Vector Machines ##
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import numpy as np
from math import radians, cos, sin, asin, sqrt



# Change this to your local path where the group data is stored
group_dir = "C:/Users/josh/OneDrive - Bath Spa University/3rd Year/CreatingIOT/iot-activity-recognition/data/group"
stations_path = "C:/Users/josh/OneDrive - Bath Spa University/3rd Year/CreatingIOT/iot-activity-recognition/data/public/train_stations_GB.csv"
all_files = os.listdir(group_dir)

# Constants 
STATIONARY_SPEED = 1
WALKING_SPEED = 7
RUNNING_SPEED = 18
IN_VEHICLE_SPEED = 72

# Define the field names to be used from the CSV files
fieldNames = ['time', 'Latitude', 'Longitude', 'Altitude (m)', 'Speed (km/h)', 'Total Distance (km)' ] 

# Load station data
stationsData = pd.read_csv(stations_path)
stationsData.columns = ['id', 'name', 'norm', 'uic', 'latitude', 'longitude', 'station_id', 'country', 'time_zone', 'is_city', 'is_main_station', 'is_airport', 'entur_id', 'entur_is_enabled']

# Create DataFrame for stations
stations = pd.DataFrame(stationsData)


# To determine if a location is near a station within a given radius
def is_nearest(lat, lon, radius_km):
    EARTHS_RADIUS = 6371
    found = (False, None)
    
    # Convert to radians
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(stations['latitude'].values)
    lon2 = np.radians(stations['longitude'].values)

    # Haversine formula vectorized (Ahmed, 2024)
    deltalat = lat2 - lat1 
    deltalon = lon2 - lon1
    squared_sine = np.sin(deltalat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(deltalon/2)**2
    angular_distance = 2 * np.arcsin(np.sqrt(squared_sine))
    km = EARTHS_RADIUS * angular_distance

    # Check if any station is within radius
    if np.any(km <= radius_km):
        nearest_station_id = stations['id'].values[km <= radius_km][0]  # first matching station
        found = (True, nearest_station_id) # station found
    return found


# Load all CSV files from the specified directory
def loadFiles(group_dir, fieldNames):
    # Lists to hold file names and dataframes
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

    # Convert speed to a numerical value (Python pandas.to_numeric method, 2018)
    pima['Speed (km/h)'] = pd.to_numeric(pima['Speed (km/h)'], errors='coerce')
   
    return pima 


# Classifies the activity based on speed
def classify_activity(speed_kmh):
    if pd.isna(speed_kmh):
        return 'unknown' # incase of NaN speed values
    if speed_kmh < STATIONARY_SPEED: # Currently set to 1 km/h
        return 'stationary'
    elif speed_kmh < WALKING_SPEED: # Currently set to 7 km/h
        return 'walking'
    elif speed_kmh < RUNNING_SPEED: # Currently set to 18 km/h
        return 'running'
    elif speed_kmh < IN_VEHICLE_SPEED: # Currently set to 72 km/h
        return 'in_vehicle' 
    else:
        return 'on_train'  # above 72 km/h is considered on train

# Classifies whether or not they are on train based on previous station
def clasifyOnTrain(onTrainIndex, previousStationId):
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


# Classifies the activities again - However backwards on the dataframe to refine those classifications that may be incorrect

def refine_classification(pima):
    # Create a new column for refined labels
    pima['label_refined'] = pima['label']
    

    # Check if they were on train or not - iterate backwards
    onTrainIndex = len(pima) - 1
    stationId = 0 # start with Destination station (the most recent station)
    while (onTrainIndex > 0):
        onTrainIndex, stationId = clasifyOnTrain(onTrainIndex, stationId)
    
    # Refine in_vehicle and walking classifications
    # Refine classification by iterating backward on the dataframe
    index = len(pima) - 1
    while index > 0:
        # Check for transition from in_vehicle to walking
        if ((pima.iloc[index-1]['label_refined'] == 'walking' or
             pima.iloc[index-1]['label_refined'] == 'running') and
             pima.iloc[index]['label_refined'] == 'in_vehicle'):
            # consider if the current activity walking or running was actually in_vehicle
            if pima.iloc[index-1]['Speed (km/h)'] > STATIONARY_SPEED: 
                pima.iloc[index-1, pima.columns.get_loc('label_refined')] = 'in_vehicle'
        index -= 1
    return pima

# Prepare data for training and testing
def prepare_data(pima, fieldNames):

    # Calculate the average speed over a 5 sampled rolling window. (amit, 2025)
    # In future this could be modified to use time stamps
    pima['averageSpeed'] = (
        pima['Speed (km/h)']
        .rolling(window=5, min_periods=1)
        .mean()
    )
    
    # Classify activities based on average speed
    pima['label'] = pima['averageSpeed'].apply(classify_activity)
    pima = refine_classification(pima)
    
    # Set x and y for train test split using the newly refined labels
    x = pima[fieldNames]

    # Currently drop the time column from the features as it contains mixed values. In future the SVM classifier could be modified to handle time series data.
    x = pima[fieldNames].drop('time', axis=1)
    # Fills missing records with NaN then replaces NaN fields with 0 (Pandas DataFrame fillna Method) and (Python pandas.to_numeric method, 2018)
    x = x.apply(pd.to_numeric, errors='coerce').fillna(0) 
  

    # set the y for train test split using the refined labels
    y = pima['label_refined']

    # print(pima.head(50)) # Print the first 50 rows of the dataframe to check the labels
    return x, y 

pima = loadFiles(group_dir, fieldNames) # Load Files
x, y = prepare_data(pima, fieldNames) # Prepare data

# Split the training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=1)

# Create SVM Pipeline Classifier Object
clf = make_pipeline(StandardScaler(), SVC()) # <-- could use SVC (kernel='poly') for different results. Currently is set to the default: SVC(kernel='rbf') due to it performing better for this dataset
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

# To visualize confusion matrix and classification report
def visualize_metrics(y_test, y_pred):
    # Setting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(set(y_test) | set(y_pred))
   
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Classification report heatmap
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1]  # remove averages and support
    
    sns.heatmap(report_df, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': 'Score'}, ax=ax2)
    ax2.set_title('Classification Report\n(Precision, Recall, F1-Score)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return report_df 

# Metrics
print(f"\nOverall Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score: {metrics.f1_score(y_test, y_pred, average='weighted'):.4f}")

# Generate visualizations
report_df = visualize_metrics(y_test, y_pred)

# References
# Amit, w. 'Understanding Pandas Rolling', Medium. Available at: https://medium.com/@whyamit101/understanding-pandas-rolling-f8f6d6796c07 (Accessed: Oct 18, 2025).
# Ahmed, R. 'Finding Nearest pair of Latitude and Longitude match using Python', Analytics Vidhya. Available at: https://medium.com/analytics-vidhya/finding-nearest-pair-of-latitude-and-longitude-match-using-python-ce50d62af546 (Accessed: Oct 28, 2025).
# Pandas DataFrame fillna Method Available at: https://www.w3schools.com/python/pandas/ref_df_fillna.asp (Accessed: Oct 23, 2025).
# Python pandas.to_numeric method(2018) Available at: https://www.geeksforgeeks.org/python/python-pandas-to_numeric-method/ (Accessed: Oct 21, 2025).
