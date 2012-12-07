#I'm going to want the forest, a fitter (logistic regression? elastic net?), and whatever this is averaged

import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

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
		
#put the training data into a big variable

csv_file_object = csv.reader(open('train.csv', 'rb')) #Load in the csv file
header = csv_file_object.next() #Skip the fist line as it is a header
train_data=[] #Creat a variable called 'train_data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

def fixdata (data):
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

	firstclassage = []
	firstclassfare = []
	secondclassage = []
	secondclassfare = []
	thirdclassage = []
	thirdclassfare = []
	
	for x in data:
		if x[1] == '1' and x[4]:
			firstclassage.append(np.float(x[4]))
		if x[1] == '1' and x[8]:
			firstclassfare.append(np.float(x[8]))
		if x[1] == '2' and x[4]:
			secondclassage.append(np.float(x[4]))
		if x[1] == '2' and x[8]:
			secondclassfare.append(np.float(x[8]))
		if x[1] == '3' and x[4]:
			thirdclassage.append(np.float(x[4]))
		if x[1] == '3' and x[8]:
			thirdclassfare.append(np.float(x[8]))

	firstclassageaverage = int(np.mean(firstclassage))
	secondclassageaverage = int(np.mean(secondclassage))
	thirdclassageaverage = int(np.mean(thirdclassage))
	
	firstclassfareaverage = int(np.mean(firstclassfare))
	secondclassfareaverage = int(np.mean(secondclassfare))
	thirdclassfareaverage = int(np.mean(thirdclassfare))
	
	#put those averages back into the '' ages

	for i in xrange(np.size(data[0::,0])):
		try:
			float(data[i,4])
		except ValueError:
			if data[i,1] == '1':
				data[i,4] = firstclassageaverage
			if data[i,1] == '2':
				data[i,4] = secondclassageaverage
			if data[i,1] == '3':
				data[i,4] = thirdclassageaverage
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

#now do the same for the test data
test_file_object = csv.reader(open('test.csv','rb'))
header = test_file_object.next()
test_data = []
for row in test_file_object:
	test_data.append(row)

test_data = np.array(test_data)
test_data = np.insert(test_data,[0], 0, axis=1)


#fix the data
train_data = fixdata(train_data)
test_data = fixdata(test_data)

#FOREST IT UP
forest = RandomForestClassifier(n_estimators=80)
print "Fitting RForest"
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

print "Prediciting RForest"
output = forest.predict(test_data[0::,1::])


open_file_object = csv.writer(open("ericsforest.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rb'))

test_file_object.next()

i = 0
for row in test_file_object:
	row.insert(0,output[i].astype(np.uint8))
	open_file_object.writerow(row)
	print row
	print i
	i += 1
