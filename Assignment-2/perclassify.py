"""
Author: Suraj
Course: CSCI544 - Applied NaturalLanguage Processing
Assignment: 2
Description: The goal of this assignment is to get some experience implementing the simple but effective machine learning model, the perceptron, and applying it to a binary text classification task (i.e., spam detection)
filename: perclassify.py - to classify the test data by the trained perceptron model
"""

import os, time
import io
import sys
from collections import defaultdict
import json

rootdir = sys.argv[1]
output_file = sys.argv[2]
output = ""

#read the saved perceptron model
with open("per_model.txt","r",encoding="latin1") as model:
    param=json.load(model)
    
weights = param["weights"]
bias = param["bias"]
testdict = defaultdict()
addweights = lambda x: int(testdict[x]*weights[x]) 
  
#read each file and test the model to classify as spam/ham  
for dirpath, dirs, files in os.walk(rootdir):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if(fname.endswith('.txt')):  
            testdict = defaultdict()
            alpha = 0
            with open(fname,'r',encoding="latin1") as fopen:
                file_content = (fopen.read().replace("\n","").split(" "))                
                for token in file_content:
                    if token not in testdict:
                        testdict[token] = 1
                    else:
                        testdict[token]+=1
                
            for token in testdict:
                if token in weights:
                    alpha+=addweights(token)
            alpha += bias
                
            if alpha >0:
                output+= "spam " + fname + "\n"
            else:
                output+= "ham " + fname + "\n"  

#save the output in the file
outputfile = open(output_file,"w")
outputfile.write(output)
outputfile.close()        
                    
                    