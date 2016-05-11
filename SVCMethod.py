#Adam Wendler
#CMSC 471
#Project 3
#SVM Model
#bitmapping code from:
#https://pythonprogramming.net/automated-image-thresholding-python/?completed=/thresholding-python-function/
import math
import sys
from sklearn.svm import LinearSVC 
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
from statistics import mean


#converts to B/W bitmap
def threshold(imageArray):
    balanceAr=[]
    newAr = imageArray
    
    #averages each pixle's RGB values
    for evryRow in imageArray:
        for evryPix in evryRow:
            avgNum = mean(evryPix[:3])
            balanceAr.append(avgNum)
        
    #averages all pixle averages
    balance = mean(balanceAr)
    for evryRow in newAr:
        for evryPix in evryRow:
            #brighter pixles are made white
            if mean(evryPix[:3]) > balance:
                evryPix[0] = 255
                evryPix[1] = 255
                evryPix[2] = 255
            #darker pixles made black
            else:
                evryPix[0] = 0
                evryPix[1] = 0
                evryPix[2] = 0
    return newAr


#creates example data from test data
def createExamples():
    versions = range(2,7)
    numLables = range(1,6)
    eiarl =[]
    

    print("Loading Training Data")
    
    #get and make numppy array from images
    for everyLable in numLables:
        for everyFile in versions:
            imgFilePath = 'Data/0' + str(everyLable) + '/0' +str(everyFile) +'.jpg'
            ei = Image.open(imgFilePath)
            eiar = np.array(ei)
            eiarl.append(threshold(eiar))
    return eiarl
    
    
    
#train SVC
def train():
    #y data SVC
    yTest = [0,0,0,0,0,
             1,1,1,1,1,
             2,2,2,2,2,
             3,3,3,3,3,
             4,4,4,4,4]
    xAll = []
    
    #get an flattened array for SVC
    eiar = createExamples()
    ii=0
    for everyX in eiar:
        xAll.append(everyX.flatten())
        
    #trian SVC
    print("Beinging Training...")
    checker = LinearSVC()
    checker.fit(xAll,yTest)
    print("Training completed!")

    return checker


#recognizes user input
def recognize(supVectMach, imgFilePath):
    lables = ['Face', 'Hat', 'Hash','Heart' ,'Money']
    #open image file
    try:
        ei = Image.open(imgFilePath)
    except:
        print("please enter valid File Path")
        return
        
    #flatten image
    eiar = np.array(ei)
    eiarl = threshold(eiar)
    Xin = eiarl.flatten()
    
    answer = supVectMach.predict([Xin])
    
    out = "Predicted label: "+ lables[answer[0]]
    print(out)

    
#driver function
def main():
    #train SVC
    supVectMach = train()
    
    
    while True:
        img = input("Please enter a file path to a 100 X 100 jpg: ")
        if img == "stop":
            return 
        recognize(supVectMach,img)
        
main()