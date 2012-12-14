#Import whatever
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
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


#scale the data
train_test = ef.scaleData(train_data, test_data)
train_data = train_test[0]
test_data = train_test[1]
bestforest = [RandomForestClassifier(n_estimators=1001),0.0]


#split into male and female
male_train = train_data[train_data[0::,2] == 1, 0::]
female_train = train_data[train_data[0::,2] == 0, 0::]
#print train_data[0::,2]
#print male_train[0::,2]

#cross validation
cv = KFold(len(train_data), k=5, indices=False)

#do a quick forest, iterate five times to get some idea of the range
for i in range(5):
	forest = RandomForestClassifier(n_estimators=101)
	forest = forest.fit(male_train[0::,1::].astype(np.float),male_train[0::,0].astype(np.float))
	print "the normalized male forest accuracy:"
	accuracy = ef.compare (forest.predict(male_train[0::,1::]).astype(np.float),male_train[0::,0].astype(np.float))
	#if accuracy > bestforest[1]:
	#	bestforest[0] = forest
	print accuracy

for i in range(5):
	forest = RandomForestClassifier(n_estimators=101)
	forest = forest.fit(female_train[0::,1::].astype(np.float),female_train[0::,0].astype(np.float))
	print "the normalized fem forest accuracy:"
	accuracy = ef.compare (forest.predict(female_train[0::,1::]).astype(np.float),female_train[0::,0].astype(np.float))
	#if accuracy > bestforest[1]:
	#	bestforest[0] = forest
	print accuracy

for i in range(5):
	forest = RandomForestClassifier(n_estimators=101)
	forest = forest.fit(train_data[0::,1::].astype(np.float),train_data[0::,0].astype(np.float))
	print "the normalized normal forest accuracy:"
	accuracy = ef.compare (forest.predict(train_data[0::,1::]).astype(np.float),train_data[0::,0].astype(np.float))
	#if accuracy > bestforest[1]:
	#	bestforest[0] = forest
	print accuracy

#get a list of forests
forestResults = []
for traincv in cv:
	forest = RandomForestClassifier(n_estimators=1001)
	forest = forest.fit(train_data[0::,1::].astype(np.float),train_data[0::,0].astype(np.float))
	forestResults.append(forest.predict(test_data[0::,1::].astype(np.float)))

forestResults = np.array(forestResults)
averagedResults =  np.mean(forestResults.astype(np.float), axis=0)
#print averagedResults
for i in xrange(np.size(averagedResults)):
	if averagedResults[i] > .4:
		averagedResults[i] = 1
	else:
		averagedResults[i] = 0


	

#forest_output = bestforest[0].predict(test_data[0::,1::])
open_file_object = csv.writer(open("normalizedforest.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the csv file
test_file_object.next()
i = 0
for row in test_file_object:
	row.insert(0,averagedResults[i].astype(np.uint8))
	open_file_object.writerow(row)
	i += 1





