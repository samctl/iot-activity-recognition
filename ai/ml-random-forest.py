"""
File: ml-random-forest.py
Created: 2025-10-16
Last Updated: 2025-10-16

Description:
    Random Forest Algorithm for Human Activity Recognition Task

Usage:
    python ml-random-forest.py

Notes:
    - Completion TBC
    - Peer Review TBC
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np


# Change this to your local path where the group data is stored
group_dir = "../data/group"
all_files = os.listdir(group_dir)

# Constants 
STATIONARY_SPEED = 1
WALKING_SPEED = 3
RUNNING_SPEED = 10
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

    # Calculate the average speed over a 5 sampled rolling window. - TODO - Currently cant use the time format but use pima.set_index('time')['Speed (km/h)'] once the time format is fixed. Currently its using sample window of 5 rows instead of time stamps
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

    print(pima.head(50)) # Print the first 50 rows of the dataframe to check the labels
    return x, y 

pima = loadFiles(group_dir, fieldNames) # Load Files
x, y = prepare_data(pima, fieldNames) # Prepare data

# Split the training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=1)

# Create Random Forest Classifier Object
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


def visualize_metrics(y_test, y_pred):
    # setting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(set(y_test) | set(y_pred))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # classification report heatmap
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1]  # remove averages and support
    
    sns.heatmap(report_df, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': 'Score'}, ax=ax2)
    ax2.set_title('Classification Report\n(Precision, Recall, F1-Score)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return report_df

# metrics
print(f"\nOverall Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score: {metrics.f1_score(y_test, y_pred, average='weighted'):.4f}")

# generate visualizations
report_df = visualize_metrics(y_test, y_pred)