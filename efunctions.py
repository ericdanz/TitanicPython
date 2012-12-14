import csv as csv
import numpy as np

from collections import Counter
from pandas import *

def compare (predicted,actual):
	score = float(0)
	for i in xrange(np.size(predicted)):
		if predicted[i] == actual[i]:
			score += 1
	score = score / np.size(predicted)
	return score

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
	
	#take means for age for each class and gender

	firstClassMAge = []
	secondClassMAge = []
	thirdClassMAge = []
	
	firstClassFAge = []
	secondClassFAge = []
	thirdClassFAge = []
	
	firstclassfare = []
	secondclassfare = []
	thirdclassfare = []
	
	
	for x in data:
		if x[4] and x[1] == '1' and x[3] == '1':
			firstClassMAge.append(np.float(x[4]))
		if x[4] and x[1] == '2' and x[3] == '1':
			secondClassMAge.append(np.float(x[4]))	
		if x[1] == '3' and x[4] and x[3] == '1':
			thirdClassMAge.append(np.float(x[4]))	
			
		if x[1] == '1' and x[4] and x[3] == '0':
			firstClassFAge.append(np.float(x[4]))
		if x[1] == '2' and x[4] and x[3] == '0':
			secondClassFAge.append(np.float(x[4]))	
		if x[1] == '3' and x[4] and x[3] == '0':
			thirdClassFAge.append(np.float(x[4]))	
			
		if x[1] == '1' and x[8]:
			firstclassfare.append(np.float(x[8]))
		if x[1] == '2' and x[8]:
			secondclassfare.append(np.float(x[8]))
		if x[1] == '3' and x[8]:
			thirdclassfare.append(np.float(x[8]))

			
	fcmavg = int(np.mean(firstClassMAge))
	scmavg = int(np.mean(secondClassMAge))
	tcmavg = int(np.mean(thirdClassMAge))
	
	fcfavg = int(np.mean(firstClassFAge))
	scfavg = int(np.mean(secondClassFAge))
	tcfavg = int(np.mean(thirdClassFAge))
	
	firstclassfareaverage = int(np.median(firstclassfare))
	secondclassfareaverage = int(np.median(secondclassfare))
	thirdclassfareaverage = int(np.median(thirdclassfare))
	
	#put those averages back into the '' ages

	for i in xrange(np.size(data[0::,0])):
		try:
			float(data[i,4])
		except ValueError:
			if x[1] == '1' and x[3] == '1':
				data[i,4] = fcmavg
			elif x[1] == '2' and x[3] == '1':
				data[i,4] = scmavg
			elif x[1] == '3' and x[3] == '1':
				data[i,4] = tcmavg
			elif x[1] == '1' and x[3] == '0':
				data[i,4] = fcfavg
			elif x[1] == '2' and x[3] == '0':
				data[i,4] = scfavg
			elif x[1] == '3' and x[3] == '0':
				data[i,4] = tcfavg
			
		try:
			float(data[i,8])
		except ValueError:
			if data[i,1] == '1':
				data[i,8] = firstclassfareaverage
			if data[i,1] == '2':
				data[i,8] = secondclassfareaverage
			if data[i,1] == '3':
				data[i,8] = thirdclassfareaverage
				

	#clean up the name and ticket  and cabin elements
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
	mfArray = trainer[0::,2]
	df = DataFrame(trainer.astype(np.float))
	df_norm = (df-df.mean())/(df.max()-df.min())
	trainer = np.array(df_norm)
	trainer[0::,0] = deparray
	trainer[0::,2] = mfArray
	mfArray = tester[0::,2]
	dfT = DataFrame(tester.astype(np.float))
	dfT_scaled = (dfT-df.mean())/(df.max()-df.min())
	tester = np.array(dfT_scaled)
	tester[0::,0] = 0
	tester[0::,2] = mfArray
	bothArray = [trainer,tester]
	return bothArray
	
