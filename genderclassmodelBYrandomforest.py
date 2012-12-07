import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter



csv_file_objectg = csv.reader(open('genderclasspricebasedmodelpy.csv', 'rb')) #Load in the csv file
csv_file_objectf = csv.reader(open('myfirstforest.csv', 'rb')) #Load in the csv file

gdata = []
fdata = []

for row in csv_file_objectg:
	gdata.append(row)

	
for row in csv_file_objectf:
	fdata.append(row)
	
gdata = np.array(gdata)
fdata = np.array(fdata)
avgsurvived = []
tempavg = ['1','2']

print xrange(np.size(fdata[0::,0]))
print xrange(np.size(gdata[0::,0]))


for i in xrange(np.size(gdata[0::,0])):
	tempavg[0] = gdata[i,0].astype(float)
	tempavg[1] = fdata[i,0].astype(float)
	print tempavg
	print np.round(np.mean(tempavg))
	

