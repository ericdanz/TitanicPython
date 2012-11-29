#Import whatever
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


#import training data
csv_file_object = csv.reader(open('train.csv', 'rb')) #Load in the csv file
header = csv_file_object.next() #Skip the fist line as it is a header
data=[] #Creat a variable called 'data'
for row in csv_file_object: #Skip through each row in the csv file
    data.append(row) #adding each row to the data variable
data = np.array(data) #Then convert from a list to an array

#scale all data values, remove ticket and name, fix cabin to be just the letter


#do a grid search for c,y on the data, possibly a second better-region-only search


#run an RBF kernel on the data








