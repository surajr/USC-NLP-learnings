"""
Author: Suraj
Course: CSCI544 - Applied NaturalLanguage Processing
Assignment: 2
Description: The goal of this assignment is to get some experience implementing the simple but effective machine learning model, the perceptron, and applying it to a binary text classification task (i.e., spam detection)
filename: perlearn.py - to learn the perceptron for the given dataset
"""

import os, time
from os.path import join
import sys
from collections import defaultdict
import json
import random
from collections import Counter

class perceptron:
    def __init__(self,filename,data,label):
        self.filename = filename
        self.edict = data
        self.label = label

rootdir = sys.argv[1]
allfiles,filenames = [],[]
weights = defaultdict()

#read each file and create an object with set of features and label 
for dirpath, dirs, files in os.walk(rootdir):
    for filename in files:
        fname = os.path.join(dirpath,filename)        
        if(fname.endswith('.txt')):
            if "spam" in fname:
                label = "spam"
            else:
                label = "ham"
            with open(fname,'r',encoding="latin1") as fopen:
                wdict = defaultdict()
                file_content = (fopen.read().replace("\n","").split(" ")) 
                for token in file_content:                    
                    #Constructing dict for each file
                    if token not in wdict:
                        wdict[token]=1
                    else:                               
                        wdict[token]+=1
                    # if token not in weights dict
                    if token not in weights:
                        weights[token] = 0
                allfiles.append(perceptron(fopen,wdict,label))

#Apply perceptron algorithm
random.shuffle(allfiles)
bias = 0
for iter in range(0,20):
    for eachfile in allfiles:
        label = 1 if eachfile.label == "spam" else -1
        alpha = 0
        for x in eachfile.edict:
            alpha = alpha + (weights[x]*eachfile.edict[x]) 
        alpha += bias
        if alpha * label <= 0:
            for x in eachfile.edict:
                weights[x] = weights[x]+(label*eachfile.edict[x])
            bias = bias + label     

 
#Save the model as per_model.txt               
model={
    "weights":weights,
    "bias":bias
}
with open("per_model.txt",'w',encoding="latin1") as fopen:
    json.dump(model,fopen)
