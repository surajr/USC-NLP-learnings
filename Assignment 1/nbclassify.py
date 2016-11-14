"""
Author: Suraj
Course: CSCI544 - Applied NaturalLanguage Processing
Assignment: 1
Description: Naive Bayes classifier to classify a file as spam/ham
filename: nbclassify - Test the model against test files
"""


import os
import json
import math
import io
import sys

 
#read the trained model
with open("nbmodel.txt","r",encoding="latin1") as model:
	param=json.load(model)

spamcount, hamcount =int(param['spamcount']), int(param['hamcount'])

#print("%d %d" %(spamcount,hamcount))

spamprob = (spamcount /(spamcount+hamcount))
hamprob = (hamcount /(spamcount+hamcount))

#print ("%s %s" % (spamprob ,  hamprob))

spamdict, hamdict = param['spam'], param['ham']
dictionary_words = list(set(list(spamdict.keys()) + list(hamdict.keys())))
dict_total = len(dictionary_words)
spamdict_count, hamdict_count = sum(list(spamdict.values())), sum(list(hamdict.values()))

rootdir=sys.argv[1]

#read the test files and classify each file as spam/ham
output = ""
for dirpath, dirs, files in os.walk(rootdir):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if(fname.endswith('.txt')):
                with open(fname,'r',encoding="latin1") as fopen:
                    file_content = (fopen.read().replace("\n","").split(" "))                  
                    count = []
                    for token in file_content:
                        if token in spamdict:
                            count.append(spamdict[token])
                        elif token in hamdict:
                            count.append(0)
                        else:
                            count.append(-1)
                    count = [c for c in count if c != -1]
                    logodds = []
                    for c in count:
                        logodds.append(math.log((c+1)/(spamdict_count + dict_total)))
                    probspam = sum(logodds) + math.log(spamprob)
                    
                                        
                    count = []
                    for token in file_content:
                        if token in hamdict:
                            count.append(hamdict[token])
                        elif token in spamdict:
                            count.append(0)
                        else:
                            count.append(-1)
                    count = [c for c in count if c != -1]
                    logodds = []

                    for c in count:
                        logodds.append(math.log((c+1)/(hamdict_count + dict_total)))
                    probham = sum(logodds) + math.log(hamprob)
                    
          	    #output the result to a file
                    if probham > probspam:
                        output = output + "ham " + fname + "\n"
                    else: 
                        output = output + "spam " + fname + "\n"

outputfile = open("nboutput.txt","w")
outputfile.write(output)
outputfile.close()