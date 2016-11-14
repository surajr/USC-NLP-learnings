"""
Author: Suraj
Course: CSCI544 - Applied NaturalLanguage Processing
Assignment: 1
Description: Naive Bayes classifier to classify a file as spam/ham
filename: nblearn - Train the model for the given dataset
"""

import io
import os
import json
import sys

spam_file_content, ham_file_content = [], []
spamdict, hamdict ={} , {}
spamcount, hamcount = 0, 0

#ead each file, calculate the number of occurences and construct spam and ham dictionary
rootdir=sys.argv[1]
for dirpath, dirs, files in os.walk(rootdir):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if "spam" in fname and "spam" in  dirpath:
            if(fname.endswith('.txt')):
                with open(fname,'r',encoding="latin1") as fopen:
                    spam_file_content = (fopen.read().replace("\n","").split(" "))                    
                    spamcount = spamcount + 1                    
                    for token in spam_file_content:
                        if token not in spamdict:
                            spamdict[token]=1
                        else:
                            spamdict[token]=spamdict[token]+1
        elif "ham" in fname and "ham" in  dirpath:
            if(fname.endswith('.txt')):
                with io.open(fname,'r',encoding="latin1") as fopen:
                    ham_file_content = (fopen.read().replace("\n","").split(" "))                    
                    hamcount = hamcount + 1                    
                    for token in ham_file_content:
                        if token not in hamdict:
                            hamdict[token]=1
                        else:
                            hamdict[token]=hamdict[token]+1

#Save the model as nbmodel.txt which is required to test on unseen data
model ={
    "spam":spamdict,
    "ham":hamdict,
    "spamcount":spamcount,
    "hamcount":hamcount
}

with open("nbmodel.txt",'w') as fopen:
    json.dump(model,fopen)
