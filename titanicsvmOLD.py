#Import whatever
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from collections import Counter
from pandas import *


def cabin_letter(x):
	return{
		'A':1,
		'B':2,
		'C':3,
		'D':4,
		'E':5,
		'F':6,
		'G':7,
		}.get(x,0)

def num (s):
	try:
		return int(s)
	except ValueError:
		return float(s)
		
def fixdataSVM (data):
#split the cabin column to find the last split (the last if there are multiple cabins)
#then assign a number based on the cabin letter
	cabin_letter_list = []
	for i in data[0::,9]:
		splitter = i.rsplit(' ',1)[-1]
		if splitter:
			splitter = cabin_letter(splitter[0])
		else:
			splitter = 0
		cabin_letter_list.append(splitter)

	data[0::,9] = cabin_letter_list
	
	#print data[0,9]

	#Male = 1, female = 0:
	data[data[0::,3]=='male',3] = 1
	data[data[0::,3]=='female',3] = 0
	#embark c=0, s=1, q=2
	data[data[0::,10] =='C',10] = 0
	data[data[0::,10] =='S',10] = 1
	data[data[0::,10] =='Q',10] = 2
	data[data[0::,10] == '',10] = 3
	#find the most common embark point, put it in the blanks
	data[data[0::,10] == 3,10] = max(Counter(data[data[0::,10] != 3,10]))
	#take means for age for each class

	age = []
	firstclassfare = []
	secondclassfare = []
	thirdclassfare = []
	
	for x in data:
		if x[4]:
			age.append(np.float(x[4]))
		if x[1] == '1' and x[8]:
			firstclassfare.append(np.float(x[8]))
		if x[1] == '2' and x[8]:
			secondclassfare.append(np.float(x[8]))
		if x[1] == '3' and x[8]:
			thirdclassfare.append(np.float(x[8]))

	ageaverage = int(np.mean(age))
	
	firstclassfareaverage = int(np.mean(firstclassfare))
	secondclassfareaverage = int(np.mean(secondclassfare))
	thirdclassfareaverage = int(np.mean(thirdclassfare))
	
	#put those averages back into the '' ages

	for i in xrange(np.size(data[0::,0])):
		try:
			float(data[i,4])
		except ValueError:
			data[i,4] = ageaverage
		try:
			float(data[i,8])
		except ValueError:
			if data[i,1] == '1':
				data[i,8] = firstclassfareaverage
			if data[i,1] == '2':
				data[i,8] = secondclassfareaverage
			if data[i,1] == '3':
				data[i,8] = thirdclassfareaverage
				
	#clean up the name and ticket elements
	data = np.delete(data,[2,7],1)
		
	#change strings to float
	for i in xrange(np.size(data[0::,0])):
		for y in range(9):
			try:
				data[i,y] = num(data[i,y])
			except ValueError:
				print y 
				print data[i,y]
				print '--'
				#data[i,y] = num(0)
	
	return data

def scaleData (trainer,tester):
	deparray = trainer[0::,0]
	df = DataFrame(trainer.astype(np.float))
	df_norm = (df-df.mean())/(df.max()-df.min())
	trainer = np.array(df_norm)
	trainer[0::,0] = deparray
	dfT = DataFrame(tester.astype(np.float))
	dfT_scaled = (dfT-df.mean())/(df.max()-df.min())
	tester = np.array(dfT_scaled)
	tester[0::,0] = 0
	bothArray = [trainer,tester]
	return bothArray
	
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
train_data = fixdataSVM(train_data)
test_data = fixdataSVM(test_data)
train_test = scaleData(train_data, test_data)
train_data = train_test[0]
test_data = train_test[1]
#print train_data[2]
#print test_data[2]


#do a quick forest
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
forest_output = forest.predict(test_data[0::,1::])

open_file_object = csv.writer(open("normalizedforest.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the csv file
test_file_object.next()
i = 0
for row in test_file_object:
    row.insert(0,forest_output[i].astype(np.uint8))
    open_file_object.writerow(row)
    i += 1


#do a grid search for c,y on the data, possibly a second better-region-only search
#do a bigger fold, maybe 10 instead of 3
C_range = 2.0 ** np.arange(-13,30)
gamma_range = 2.0 ** np.arange(-17,13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(train_data[0::,0], 10)
grid = GridSearchCV(SVC(kernel='rbf',cache_size=2000), param_grid = param_grid, cv = cv, n_jobs=5, verbose=2)
grid.fit(train_data[0::,1::],train_data[0::,0])
thebest = ["The best classifier is: ", grid.best_estimator_]


open_file_object = csv.writer(open("thebest.txt", "wb"))
open_file_object.writerow(thebest)
open_file_object.writerow(train_data[5])
open_file_object.writerow(test_data[15])

#run an RBF kernel on the data
myRBF = grid.best_estimator_
rbfResults = myRBF.predict(test_data[0::,1::])


#record the RBF results 
open_file_object = csv.writer(open("rbf.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the csv file
test_file_object.next()

i = 0
for row in test_file_object:
	row.insert(0,rbfResults[i].astype(np.uint8))
	open_file_object.writerow(row)
	i += 1




