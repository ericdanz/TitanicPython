import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter



csv_file_objectg = csv.reader(open('genderclassmodel2.csv', 'rb')) #Load in the csv file
csv_file_objectf = csv.reader(open('myfirstforest2.csv', 'rb')) #Load in the csv file
csv_file_objectr = csv.reader(open('rbf.csv', 'rb')) #Load in the csv file
csv_file_objectn = csv.reader(open('normalizedforest.csv', 'rb')) #Load in the csv file
csv_file_objectgb = csv.reader(open('sexbasedgbc.csv', 'rb')) #Load in the csv file




gdata = []
fdata = []
rdata = []
ndata = []
gbdata = []
avgdata = []

for row in csv_file_objectn:
	ndata.append(row)

for row in csv_file_objectgb:
	gbdata.append(row)

	
for row in csv_file_objectg:
	gdata.append(row)

	
for row in csv_file_objectf:
	fdata.append(row)

for row in csv_file_objectr:
	rdata.append(row)

	
ndata = np.array(ndata)	
gdata = np.array(gdata)
gbdata = np.array(gbdata)
fdata = np.array(fdata)
rdata = np.array(rdata)
avgsurvived = []
tempavg = ['1','2','3','4','5']
#tempavg = ['1','2','3']
print xrange(np.size(fdata[0::,0]))
print xrange(np.size(gdata[0::,0]))


for i in xrange(np.size(gdata[0::,0])):
	tempavg[0] = gdata[i,0].astype(float)
	tempavg[1] = fdata[i,0].astype(float)
	tempavg[2] = rdata[i,0].astype(float)
	tempavg[3] = gbdata[i,0].astype(float)
	tempavg[4] = ndata[i,0].astype(float)
	#print tempavg
	print np.mean(tempavg)
	if np.mean(tempavg) > .6:
		avgdata.append(1)
	else:
		avgdata.append(0)
	#avgdata.append((np.mean(tempavg)))
	

open_file_object = csv.writer(open("avgforest6threshold.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rb'))

test_file_object.next()

i = 0
for row in test_file_object:
	row.insert(0,avgdata[i])
	open_file_object.writerow(row)
	print i
	i += 1




