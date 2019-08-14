import csv
import operator
import re
import random

filename = "data/data.csv"
opfilename = "data/smalljson.csv"
sampleSize = 10

with open(filename) as csvfile:
	reader = csv.reader(csvfile, delimiter='\t')
	header = next(reader)
	origList = list(reader)

sList = random.sample(origList, sampleSize)

with open(opfilename, 'w') as opfile:
    wr = csv.writer(opfile, delimiter = '\t')
    wr.writerow(header)
    for entry in sList:
    	wr.writerow(entry)
