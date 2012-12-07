#Import whatever
import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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
	
	df = DataFrame(data.astype(np.float))
	df_norm = (df-df.mean())/(df.max()-df.min())
	print df_norm[5]
	
	
	return data

#import training data
csv_file_object = csv.reader(open('train.csv', 'rb')) #Load in the csv file
header = csv_file_object.next() #Skip the fist line as it is a header
train_data=[] #Creat a variable called 'data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

#normalize data frame, remove ticket and name, fix cabin to be just the letter
train_data = fixdataSVM(train_data)
print train_data[4]


#do a grid search for c,y on the data, possibly a second better-region-only search


#run an RBF kernel on the data








