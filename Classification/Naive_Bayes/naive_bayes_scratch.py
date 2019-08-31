# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 00:28:46 2019

@author: Bhavya Kala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean,stdev
import math
#def summary(dataset):
#    stdev=dataset.std(axis=0)
#    mean=dataset.mean(axis=0)
#    summaries=list(zip(mean,stdev))
#    del summaries[-1]
#    return summaries
import csv
def loadCsv(filename):
	lines = csv.reader(open(filename, "rt"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

import random
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separatebyclass(dataset):
    separated={}
    for i in range(len(dataset)):
        v = dataset[i]
        if v[-1] not in separated:
            separated[v[-1]]=[]
        separated[v[-1]].append(v)
    return separated    
    
def summarize(dataset):
    summaries= [(mean(x),stdev(x)) for x in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizebyclass(dataset):
    separated=separatebyclass(dataset)
    summaries={}
    for cv,i in separated.items():
        summaries[cv]=summarize(i)
    return summaries   

def gaussianpdf(x,mean,std):
    exponent = math.exp( (-(math.pow(x-mean,2))/(2*math.pow(std,2))) )
    gpdf = 1/((math.sqrt(2*math.pi))*std)*exponent
    return gpdf

def classprobability(summaries, iv):
    probabilities = {}
    for cv,cs in summaries.items():
        probabilities[cv]=1
        for i in range(len(cs)):
            mean, std = cs[i]
            x = iv[i]
            probabilities[cv]*=gaussianpdf(x,mean,std)
    return probabilities

def predict(summaries,iv):
    probabs = classprobability(summaries,iv)
    bestlabel, bestprob = None, -1
    for cv,p in probabs.items():
        if bestlabel is None or p>bestprob:
            bestprob=p
            bestlabel=cv
    return bestlabel
        
def getpredictions(summaries, x_test):
    predictions = []
    for i in range(len(x_test)):
        predictions.append(predict(summaries, x_test[i]))
    return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
    filename = 'diabetes.csv'
    dataset = loadCsv(filename)
    training_set, test_set = splitDataset(dataset, 0.2)
    summaries = summarizebyclass(training_set)
    y_pred = getpredictions(summaries, test_set)
    acc=getAccuracy(test_set, y_pred)
    print(acc)
    
main()     
        
    
        
            
    
    





 
        
    
    
    
    
    
    