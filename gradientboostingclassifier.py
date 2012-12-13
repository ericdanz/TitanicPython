#Import whatever
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from collections import Counter
from pandas import *
import efunctions as ef

	
#import training data
csv_file_object = csv.reader(open('train.csv', 'rb')) #Load in the csv file
header = csv_file_object.next() #Skip the fist line as it is a header
train_data=[] #Creat a variable called 'data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the test csv file
header = test_file_object.next() #Skip the fist line as it is a header
test_data=[] #Creat a variable called 'test_data'
for row in test_file_object: #Skip through each row in the csv file
    test_data.append(row) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array
test_data = np.insert(test_data,[0], 0, axis=1)

#normalize data frame, remove ticket and name, fix cabin to be just the letter
#in the future, scale the data by the train set and apply that to test set
train_data = ef.fixdataSVM(train_data)
test_data = ef.fixdataSVM(test_data)
train_test = ef.scaleData(train_data, test_data)
train_data = train_test[0]
test_data = train_test[1]
#print train_data[2]
#print test_data[2]


#do a quick forest
#forest = RandomForestClassifier(n_estimators=100)
#forest = forest.fit(train_data[0::,1::],train_data[0::,0])
#forest_output = forest.predict(test_data[0::,1::])

#open_file_object = csv.writer(open("normalizedforest.csv", "wb"))
#test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the csv file
#test_file_object.next()
#i = 0
#for row in test_file_object:
#    row.insert(0,forest_output[i].astype(np.uint8))
#    open_file_object.writerow(row)
#    i += 1

#GBC
nEst = 1001
lR = .3

gb = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst)
gb.fit(train_data[0::,1::],train_data[0::,0])
gbcResults = gb.predict(test_data[0::,1::])


#Cross-Validate the RF
cvScore = []
savedGBC = []
cv = StratifiedKFold(train_data[0::,0], 15)
for train,test in cv:
	cvGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst).fit(train_data[train,1::].astype(np.float),train_data[train,0].astype(np.float))
	savedGBC.append(cvGBC)
	thisOutput = cvGBC.predict(train_data[test,1::].astype(np.float))
	cvScore.append(ef.compare(thisOutput.astype(np.float),train_data[test,0].astype(np.float)))

print "CV Scores:"
for score in cvScore:
	print score

print "Against the whole training set accuracy:"
sfOutput = []
for s in savedGBC:
	thisOutput = s.predict(train_data[0::,1::].astype(np.float))
	print ef.compare(thisOutput.astype(np.float),train_data[0::,0].astype(np.float))
	sfOutput.append(thisOutput)
	
averageOutput = []
sfOutput = np.array(sfOutput)
averageOutput =  np.mean(sfOutput[0::].astype(np.int), axis=0)
averageOutput = averageOutput.astype(np.int)

print "Averaged accuracy:"
print ef.compare(averageOutput.astype(np.float),train_data[0::,0].astype(np.float))

print "single GBC accuracy:"
print ef.compare (gb.predict(train_data[0::,1::]).astype(np.float),train_data[0::,0].astype(np.float))






#record the GBC results 
'''
open_file_object = csv.writer(open("gbc.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the csv file
test_file_object.next()

i = 0
for row in test_file_object:
	row.insert(0,gbcResults[i].astype(np.uint8))
	open_file_object.writerow(row)
	i += 1
'''



