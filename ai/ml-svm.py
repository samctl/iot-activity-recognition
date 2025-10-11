"""
File: ml-svm.py
Created: 2025-10-08
Last Updated: 2025-10-11

Description:
    Support Vector Machines (SVM) Alogrithm for Human Activity Recognition Task

Usage:
    python ml-svm.py

Notes:
    - Completion TBC
    - Peer Review TBC
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
## Imports for Support Vector Machines ##
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Field Names are the columns / types of data the datasets have collected

# Note: These field names likely need to be changed.
fieldNames = ['time', 'Latitude', 'Longitude', 'Altitude (m)', 'Speed (km/h)', 'Total distance (km)', ] 

# Todo: multiple dataframes / data files
# TODO: get all of above from the "../data/group/" directory instead of manually loading
#dataFrames = []

# Load csv file
pima = pd.read_csv("run1_jw.csv", sep='\t', encoding='utf-16', skiprows=2, names=fieldNames)

# Convert speed to a numerical value 
pima['Speed (km/h)'] = pd.to_numeric(pima['Speed (km/h)'], errors='coerce')

# Convert time to datatime to create time interval feature 
pima['time'] = pd.to_datetime(pima['time'], errors='coerce')

# Create an average speed over a 5 second period
pima['averageSpeed'] = (
    pima.set_index('time')['Speed (km/h)']
    .rolling('5s', min_periods=1) # TODO: Test different rolling averages
    .mean()
    .reset_index(drop=True)
)


# TODO: Refine these rules for better accuracy
# TODO: Need a way of not straight away dropping from in_vehicle to walking if moving slow for a second etc
def classify_activity(speed_kmh):
    if pd.isna(speed_kmh):
        return 'unknown'
    if speed_kmh < 1:
        return 'stationary'
    elif speed_kmh < 7:
        return 'walking'
    elif speed_kmh < 18:
        return 'running'
    elif speed_kmh < 72:
        return 'in_vehicle'  # Can include any road vehicles
    else:
        return 'on_train'    # higher speeds


# Apply it
pima['label'] = pima['averageSpeed'].apply(classify_activity)

# Set x and y for train test split
x = pima[['Latitude', 'Longitude', 'Altitude (m)', 'averageSpeed', 'Total distance (km)']]
y = pima['label']

# Split the training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=1)

# Create SVM Pipline Classifer Object
clf = make_pipeline(StandardScaler(), SVC())
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

# predict the response for test dataset
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# TODO: More metrics (precision, recall etc.)

# Print the first 5 rows
#print(pima.head())

