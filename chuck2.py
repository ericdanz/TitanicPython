#I'm going to want the forest, a fitter (logistic regression? elastic net?), and whatever this is averaged

import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#put the training data into a big variable

csv_file_object = csv.reader(open('../csv/train.csv', 'rb')) #Load in the csv file
header = csv_file_object.next() #Skip the fist line as it is a header
data=[] #Creat a variable called 'data'
for row in csv_file_object: #Skip through each row in the csv file
    data.append(row) #adding each row to the data variable
data = np.array(data) #Then convert from a list to an array

#split the cabin column to find the last split (the last if there are multiple cabins)
#split the cabin column to find the last split (the last if there are multiple cabins)

for i in data.shape[0]:
	

data[column9] = 
