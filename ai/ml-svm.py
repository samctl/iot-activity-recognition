"""
File: ml-svm.py
Author: Sam, 
Created: 2025-10-08
Last Updated: 2025-10-08

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
#dataFrames = []

# Load csv file
pima = pd.read_csv("../data/group/run-walk-mixed-Birmingham.csv", header=None, names=fieldNames, encoding='utf-16')

pima.head()

# Data Classification



