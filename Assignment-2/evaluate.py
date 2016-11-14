"""
Author: Suraj
Course: CSCI544 - Applied NaturalLanguage Processing
Assignment: 2
Description: The goal of this assignment is to get some experience implementing the simple but effective machine learning model, the perceptron, and applying it to a binary text classification task (i.e., spam detection)
filename: perlearn.py - to learn the perceptron for the given dataset

Construct the confusion matrix and calculate the accuracies
"""

import os, time
import io
from os.path import join
import sys

predictResults =[]
outputfile = sys.argv[1]
with io.open(outputfile) as results:
    for res in results:
        idx = res.find(" ")
        predictlabel = res[0:idx]
        filepath = res[idx+1:]
        filename = filepath[filepath.rfind("/")+1:]
        if "ham" in filename:
            actual = "ham"
        else:
            actual = "spam"
        predictResults  = predictResults  + [(predictlabel, actual)]
        
Matrix = [[0,0],[0,0]]
for result in predictResults:
    if(result[0] == result[1] == "ham"):
        Matrix[0][0] = Matrix[0][0] + 1
    elif(result[0] == "ham" and result[1] == "spam"):
        Matrix[1][0] = Matrix[1][0] + 1
	elif(result[0] == result[1] == "spam"):
        Matrix[1][1] = Matrix[1][1] + 1    
    else:
        Matrix[0][1] = Matrix[0][1] + 1

print(Matrix)

precisionSpam = Matrix[1][1] / (Matrix[1][1] + Matrix[0][1])
recallSpam = Matrix[1][1] / (Matrix[1][1] + Matrix[1][0])
fscoreSpam = 2 * precisionSpam * recallSpam / (precisionSpam + recallSpam)
print("[SPAM]Precision:{0},Recall:{1},FScore:{2}".format(precisionSpam,recallSpam,fscoreSpam))

precisionHam = Matrix[0][0] / (Matrix[0][0] + Matrix[1][0])
recallHam = Matrix[0][0] / (Matrix[0][0] + Matrix[0][1])
fscoreHam = 2 * precisionHam * recallHam / (precisionHam + recallHam)
print("[HAM]Precision:{0},Recall:{1},FScore:{2}".format(precisionHam,recallHam,fscoreHam))