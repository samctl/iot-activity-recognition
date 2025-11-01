"""
File: dl-algorithm.py
Author: Sam Moss, Josh White, Jack Wainwright
Created: 2025-09-28
Last Updated: 2025-09-28

Description:
This is a placeholder file
    
Usage:
    python dl-algorithm.py

Notes:
    - RNN Alogrithm
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
group_dir = "../data/group"
all_files = os.listdir(group_dir)

# constant model tweaking variables
STATIONARY_SPEED = 1
WALKING_SPEED = 5
RUNNING_SPEED = 20
IN_VEHICLE_SPEED = 52

fieldNames = ['time', 'Latitude', 'Longitude', 'Altitude (m)', 'Speed (km/h)', 'Total Distance (km)' ] 

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


def prepare_data(pima, fieldNames):
    
    # Calculate the average speed over a 5 sampled rolling window.
    pima['averageSpeed'] = (
        pima['Speed (km/h)']
        .rolling(window=int(5), min_periods=1)
        .mean()
    )

    pima['label'] = pima['averageSpeed'].apply(classify_activity)
    #pima['label'] = pima['Speed (km/h)'].apply(classify_activity)
    pima = refine_classification(pima)
    
    # Set x and y for train test split using the newly refined labels
    x = pima[fieldNames]
    
    # currently drop the time column from the features as it contains mixed values. TODO - the drop needs to be removed and used as a feature in the SVM classifier once the time format is standardised
    x = pima[fieldNames].drop('time', axis=1)
    x = x.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # set the y for train test split using the refined labels
    y = pima['label_refined']

    print(pima.head(50))
    return x, y 
