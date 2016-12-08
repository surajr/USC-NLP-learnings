import os
import sys
from hw3_corpus_tool import *
import pycrfsuite
import time

#72.56 previous

def extractFeatures(rootdir):
    features, label = [], []    
    
    for dirpath, dirs, files in os.walk(rootdir):
        for filename in files:
            fname = os.path.join(dirpath,filename)         
            var = get_utterances_from_filename(fname)    
            first_speaker,output = 0, ''
            previous_speaker, first_utterance = None, True
            
            for utterance in var:
                temp = []
                first_speaker = utterance[1]            
                if first_utterance:
                    temp.append("0")
                    temp.append("1")
                    first_utterance = False                

                elif first_speaker != previous_speaker:
                    temp.append("1")
                    temp.append("0")                

                else:
                    temp.append("0")
                    temp.append("0") 

                previous_speaker = first_speaker 
                
                if utterance[0] is not None:
                    act_tag = utterance[0]
                else:
                    act_tag = None       

                if(utterance[2] is not None):
                    for y in range(len(utterance[2])):
                        #all tokens have been converted to lower case to add weight to the repetitive words irrespective of the case.
                        if(utterance[2][y][0] is not None):
                            temp.append('TOKEN_'+utterance[2][y][0])
                            temp.append('POS_'+utterance[2][y][1])
                        #####
                               
                else:
                    temp.append('NONE')
                    temp.append('NONE')   

                features.append(temp)
                label.append(act_tag)
                
    return features, label


def testFeatures(fname):
    features, label = [], []
    
    var = get_utterances_from_filename(fname)    
    first_speaker,output= 0, ''
    previous_speaker, first_utterance = None, True

    for utterance in var:
        temp = []
        first_speaker = utterance[1]            
        if first_utterance:
            temp.append("0")
            temp.append("1")
            first_utterance = False                

        elif first_speaker != previous_speaker:
            temp.append("1")
            temp.append("0")                

        else:
            temp.append("0")
            temp.append("0") 

        previous_speaker = first_speaker 
        
        if utterance[0] is not None:
            act_tag = utterance[0]
        else:
            act_tag = None               

        if(utterance[2] is not None):
            for y in range(len(utterance[2])):
                #all tokens have been converted to lower case to add weight to the repetitive words irrespective of the case.
                if(utterance[2][y][0] is not None):
                    temp.append('TOKEN_'+utterance[2][y][0])
                    temp.append('POS_'+utterance[2][y][1])
                #####
                      
        else:
            temp.append('NONE')
            temp.append('NONE')         

        features.append(temp)

        label.append(act_tag)
        temp = []
    return features, label

start = time.time()
traindir, testdir, outputfile = sys.argv[1],sys.argv[2],sys.argv[3]
print ("baseline start")
features, label = extractFeatures(traindir)

trainer = pycrfsuite.Trainer(verbose=False)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-5,  # coefficient for L2 penalty
    'max_iterations': 100,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.append(features, label)
trainer.train('cs544-assign3.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('cs544-assign3.crfsuite')

output = ''
rootdir = testdir
#accCount, labelCount = 0,0

for dirpath, dirs, files in os.walk(rootdir):
    for filename in files:
        fname = os.path.join(dirpath,filename)        
        testfeatures, testlabels = testFeatures(fname)
        output += "Filename=\""+ filename + "\"\n" + "\n".join(tagger.tag(testfeatures))       
        output += "\n\n"    
        
        #Commented code is for accuracy calculation
        #pred = tagger.tag(testfeatures)
        #labelCount += len(testlabels)
		#for i in range(0, len(pred)):
			#if pred[i] == testlabels[i]:
				#accCount += 1
                
#print ("Accuracy of CRFBaseline is %s"%(float(accCount)/labelCount))
outputfile = open(outputfile,"w")
outputfile.write(output)
print ("baseline crf end")
outputfile.close() 
end = time.time()
elapsed = (end - start)/60
print ("elapsed = %s" % elapsed)