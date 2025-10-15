# Preprocessing folder

Upload point for group HAR data.

As the group is comprised of both Apple and Android users each using slightly different GPS trackers, some formatting of .csv files is required to ensure all files hold the same fields of data. 

These changes are as follows (Applicable to data exported from Apple devices):
* Delete top row with document title
* Rename date column to time 
* Delete Item column
* Delete Elapse (s) column
* Delete Heading column