#Import whatever
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
nEst = 601
lR = 0.2
subSam = 1.0



gb = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,  subsample = subSam)
gb.fit(train_data[0::,1::],train_data[0::,0])
gbcResults = gb.predict(test_data[0::,1::])


#Cross-Validate the RF
cvScore = []
savedGBC = []
cv = StratifiedKFold(train_data[0::,0], 15)
for train,test in cv:
	cvGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,subsample = subSam).fit(train_data[train,1::].astype(np.float),train_data[train,0].astype(np.float))
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
averageOutput =  np.mean(sfOutput[0::].astype(np.float), axis=0)
for x in xrange(np.size(averageOutput)):
	if averageOutput[x] > .6:
		averageOutput[x] = 1
	else:
		averageOutput[x] = 0

print "Averaged accuracy with threshold .6:"
print ef.compare(averageOutput.astype(np.float),train_data[0::,0].astype(np.float))

print "single GBC accuracy:"
print ef.compare (gb.predict(train_data[0::,1::]).astype(np.float),train_data[0::,0].astype(np.float))

#male and female split
male_train = train_data[train_data[0::,2] == 1,0::]
female_train = train_data[train_data[0::,2] == 0,0::]

#split the test data
male_test = test_data[test_data[0::,2] == 1,0::]
female_test = test_data[test_data[0::,2] == 0,0::]


#cv it
mcv = KFold(len(male_train[0::,0]), 15)
fcv = KFold(len(female_train[0::,0]), 15)

sMG = []
sFG = []

for train,test in mcv:
	mGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,subsample = subSam).fit(male_train[train,1::].astype(np.float),male_train[train,0].astype(np.float))
	male_output = mGBC.predict(male_test[0::,1::])
	sMG.append(male_output)


sMG = np.array(sMG)

male_average = np.mean(sMG, axis=0)
for i in xrange(np.size(male_average)):
	if male_average[i] > .6:
		male_average[i] = 1
	else:
		male_average[i] = 0




for train,test in fcv:
	fGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,subsample = subSam).fit(female_train[train,1::].astype(np.float),female_train[train,0].astype(np.float))
	female_output = fGBC.predict(female_test[0::,1::])
	sFG.append(female_output)


sFG = np.array(sFG)

female_average = np.mean(sFG, axis=0)
for i in xrange(np.size(female_average)):
	if female_average[i] > .6:
		female_average[i] = 1
	else:
		female_average[i] = 0


'''
print 'MALE AVERAGE'
print ef.compare(male_average,male_train[0::,0])

print 'FEMALE AVERAGE'
print ef.compare(female_average,female_train[0::,0])



#cv it
mcv = KFold(len(male_test[0::,0]), 15)
fcv = KFold(len(female_test[0::,0]), 15)

sMG = []
sFG = []


for train,test in mcv:
	mGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,subsample = subSam).fit(male_test[train,1::].astype(np.float),male_test[train,0].astype(np.float))

	male_output = mGBC.predict(male_test[0::,1::].astype(np.float))
	sMG.append(male_output)

sMG = np.array(sMG)

male_average = np.mean(sMG, axis=0)
for i in xrange(np.size(male_average)):
	if male_average[i] > .6:
		#print male_average[i]
		male_average[i] = 1
	else:
		male_average[i] = 0

for train,test in fcv:
	fGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,subsample = subSam).fit(female_test[train,1::].astype(np.float),female_test[train,0].astype(np.float))
	female_output = fGBC.predict(female_test[0::,1::].astype(np.float))
	sFG.append(female_output)

sFG = np.array(sFG)

female_average = np.mean(sFG, axis=0)
for i in xrange(np.size(female_average)):
	if female_average[i] > .6:
		#print female_average[i]
		female_average[i] = 1
	else:
		female_average[i] = 0

'''
test_data[test_data[0::,2] == 1,0] = male_average
test_data[test_data[0::,2] == 0,0] = female_average


'''
mGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,subsample = subSam).fit(male_train[0::,1::].astype(np.float),male_train[0::,0].astype(np.float))

fGBC = GradientBoostingClassifier(learn_rate = lR, n_estimators = nEst,subsample = subSam).fit(female_train[0::,1::].astype(np.float),female_train[0::,0].astype(np.float))


male_output = mGBC.predict(male_train[0::,1::].astype(np.float))

female_output = fGBC.predict(female_train[0::,1::].astype(np.float))

print "male gbc:"
print ef.compare(male_output.astype(np.float),male_train[0::,0].astype(np.float))

print "female gbc:"
print ef.compare(female_output.astype(np.float),female_train[0::,0].astype(np.float))
'''






#record the GBC results based on male/female

open_file_object = csv.writer(open("sexbasedgbc.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the csv file
test_file_object.next()

i = 0
for row in test_file_object:
	row.insert(0,test_data[i,0].astype(np.uint8))
	open_file_object.writerow(row)
	i += 1



