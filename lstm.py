#!/usr/bin/python 
'''Script that constructs a simple lstm network , and classifies the dataset of SICURA images as guns and non guns '''

__author__ = 'Saurav , saurav@iupr.com' 


from pybrain.structure.networks.recurrent import RecurrentNetwork 
from pybrain import LinearLayer , FullConnection , LSTMLayer , BiasUnit 
from pybrain.tests import runModuleTestSuite 
from pybrain.utilities import percentError 
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet 
from scipy import diag 
from numpy.random import multivariate_normal 

##imports fixed 
###create the dataset 

def create_dataset():
   '''Create a random dataset to train and test the network on '''
   means = [(-1,0),(2,4),(3,1)]
   cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
   alldata=ClassificationDataSet(2,1,nb_classes=3)
   for n in xrange(400):
      for klass in range(3):
             input=multivariate_normal(means[klass],cov[klass])
             alldata.addSample(input,[klass])
   tstdata,trndata=alldata.splitWithProportion(0.25)
   trndata._convertToOneOfMany() 
   tstdata._convertToOneOfMany() 
   return (trndata,tstdata) 

def buildSimpleLSTMNetwork(peepholes=False): 
   '''Module that builds an LSTM network and returns it '''
   N=RecurrentNetwork('simpleLstmNet') 
   i=LinearLayer(1, name='inp') 
   h=LSTMLayer(1,peepholes=peepholes , name='lstm')
   o=LinearLayer(1, name='outp')
   b =BiasUnit('bias')
   N.addmodule(b) 
   N.addOutputModule(o)
   N.addInputModule(i)
   N.addModule(h) 
   N.addConnection(FullConnection(i,h,name='f1'))
   N.addConnection(FullConnection(b,h,name='f2'))
   N.addConnection(FullConnection(h,o,name='r1'))
   N.sortModules()
   return N 

if __name__ =="__main__":
  runModuleTestSuite(__import__('__main__'))
  trndata , tstdata = create_dataset()
  print "Training data shape : " , trndata.indim , trndata.outdim 
  print "\n A random training data sample :" 
  print trndata['input'][0] , trndata['target'][0] ,trndata['class'][0] 


  
