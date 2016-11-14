"""
Author: Suraj
Course: CSCI544 - Applied NaturalLanguage Processing
Assignment: 2
Description: The goal of this assignment is to get some experience implementing the simple but effective machine learning model, the perceptron, and applying it to a binary text classification task (i.e., spam detection)
filename: avg_perlearn.py - to learn the average perceptron for the given dataset
"""

import os, time
from os.path import join
import sys
from collections import defaultdict
import json
import random
import copy

class perceptron:
    def __init__(self,filename,data,label):
        self.filename = filename
        self.edict = data
        self.label = label

rootdir = sys.argv[1]
allfiles,filenames = [],[]
weights = defaultdict()
u = defaultdict()

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
                        u[token]=0
                allfiles.append(perceptron(fopen,wdict,label))

random.shuffle(allfiles)
bias = 0

#Apply average perceptron algorithm
c=1
beta = 0
for iter in range(0,30):
    for eachfile in allfiles:
        label = 1 if eachfile.label == "spam" else -1
        alpha = 0
        for x in eachfile.edict:
            alpha = alpha + (weights[x]*eachfile.edict[x]) 
        alpha += bias
        if alpha * label <= 0:
            for x in eachfile.edict:
                weights[x] = weights[x]+(label*eachfile.edict[x])                     
                u[x] = u[x]+(label*c*eachfile.edict[x])
            bias = bias + label 
            beta = beta + (label * c)
        c=c+1
for d in u:
    u[d] = weights[d]-((1/c)*u[d])
beta= bias - ((1/c)*beta)
    
#save the model        
model={
    "weights":u,
    "bias":beta
}

#write to a file
with open("per_model.txt",'w',encoding="latin1") as fopen:
    json.dump(model,fopen)
