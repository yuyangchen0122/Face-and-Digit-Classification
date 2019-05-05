# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import mira
import samples
import sys
import util
import timeit
import random

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  
  ##
  """

  #work not good
  '''
  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) == 1:
        features[(x, y, 1)] = 1
        features[(x, y, 0)] = 0
        features[(x, y, 2)] = 0
      elif datum.getPixel(x, y) == 2:
        features[(x, y, 1)] = 0
        features[(x, y, 0)] = 0
        features[(x, y, 2)] = 1
      else:
        features[(x, y, 1)] = 0
        features[(x, y, 0)] = 1
        features[(x, y, 2)] = 0
  '''
  "*** YOUR CODE HERE ***"
  

  
  
  '''
  for x in range(DIGIT_DATUM_HEIGHT):
    black = 0
    for y in range(DIGIT_DATUM_WIDTH):
      black += datum.getPixel(y, x) % 1
    features[(y, x , 1, black)] = 1
    for i in range(DIGIT_DATUM_HEIGHT):
      if(i != black):
        features[(y, x, 1, i)] = 0
  '''
  features = util.Counter()

  flag = 0
  start_row = 0
  end_row = 0
  start_column = 0
  end_column = 0

  #check start row
  for x in range(DIGIT_DATUM_HEIGHT):
    flag = 0
    for y in range(DIGIT_DATUM_WIDTH):
      if(datum.getPixel(y, x) > 0):
        flag = 1
        start_row = x
        break
    if flag != 0:
      break

  #check end row
  for x in range(DIGIT_DATUM_HEIGHT):
    flag = 0
    for y in range(DIGIT_DATUM_WIDTH):
      if(datum.getPixel(DIGIT_DATUM_WIDTH - y - 1, DIGIT_DATUM_HEIGHT - x - 1) > 0):
        flag = 1
        end_row = DIGIT_DATUM_HEIGHT - x - 1
        break
    if flag != 0:
      break
  
  #check start column
  for x in range(DIGIT_DATUM_WIDTH):
    flag = 0
    for y in range(DIGIT_DATUM_HEIGHT):
      if(datum.getPixel(x, y) > 0):
        flag = 1
        start_column = x
        break
    if flag != 0:
      break

  #check end column
  for x in range(DIGIT_DATUM_WIDTH):
    flag = 0
    for y in range(DIGIT_DATUM_HEIGHT):
      if(datum.getPixel(DIGIT_DATUM_WIDTH - x - 1, DIGIT_DATUM_HEIGHT - y - 1) > 0):
        flag = 1
        end_column = DIGIT_DATUM_WIDTH - x - 1
        break
    if flag != 0:
      break

  hight = float(end_row - start_row)
  width = float(end_column - start_column)
  if(hight / width <1.2):
    features['fat'] = 1
    features['mid'] = 0
    features['thin'] = 0

  elif(hight / width >=1.2 and hight / width < 2):
    features['fat'] = 0
    features['mid'] = 1
    features['thin'] = 0

  else:
    features['fat'] = 0
    features['mid'] = 0
    features['thin'] = 1


  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  '''
  for x in range(start_column, end_column + 1):
    grey = 0
    for y in range(start_row, end_row + 1):
      if(datum.getPixel(x, y) == 1):
        grey += 1
    features[(x - start_column,'column', grey)] = 1
  '''
  for x in range(start_row, end_row + 1):
    grey = 0
    for y in range(start_column, end_column + 1):
      if(datum.getPixel(y, x) == 1):
        grey += datum.getPixel(y, x)
    features[(x - start_row , 'row', grey)] = 1

  '''
  for x in range(start_column, end_column + 1):
    black = 0
    for y in range(start_row, end_row + 1):
      black += datum.getPixel(x, y) % 1
    features[(x - start_column, 'bcolumn', black)] = 1
    for i in range(DIGIT_DATUM_HEIGHT):
      if(i != black):
        features[(x -start_column, 'bcolumn',i)] = 0
    '''
  

  # print("start column is {}, end column is {}".format(start_column, end_column))
  # print("start row is {}, end row is {}".format(start_row, end_row))
  #['black', 0, 0, i - 1, i, num] means row i -1 has num more non-white pixels than row i 
  '''
  prev = 0
  curr = 0
  for i in range(start_row, end_row + 1):
    prev = curr
    curr = 0
    for j in range(start_column, end_column + 1):
      if(datum.getPixel(j, i) > 0):
        curr += 1
    if(abs(prev - curr) == 0 ):
      features[('black', 'row', i - 1 - start_row, i - start_row, 0)] = 1
      features[('black', 'row', i - 1 - start_row, i - start_row, 1)] = 0
      features[('black', 'row', i - 1 - start_row, i - start_row, 2)] = 0
      features[('black', 'row', i - 1 - start_row, i - start_row, -1)] = 0
      features[('black', 'row', i - 1 - start_row, i - start_row, -2)] = 0
    elif((abs(prev - curr) <= 3)):
      if(prev - curr > 0):
        features[('black', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, 1)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, 2)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, -1)] = 1
        features[('black', 'row', i - 1 - start_row, i - start_row, -2)] = 0
      else:
        features[('black', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, 1)] = 1
        features[('black', 'row', i - 1 - start_row, i - start_row, 2)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, -1)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, -2)] = 0
    else:
      if(prev - curr > 0):
        features[('black', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, 1)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, 2)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, -1)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, -2)] = 1
      else:
        features[('black', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, 1)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, 2)] = 1
        features[('black', 'row', i - 1 - start_row, i - start_row, -1)] = 0
        features[('black', 'row', i - 1 - start_row, i - start_row, -2)] = 0

  #['white', 0, 0, i - 1, i, num] means row i -1 has num more white pixels than row i 
  curr = 0
  for i in range(start_row, end_row + 1):
    prev = curr
    curr = 0
    for j in range(start_column, end_column + 1):
      rs = start_row
      re = start_row
      if(datum.getPixel(j, i) != 0):
        for k in range(j, end_column):
          if(datum.getPixel(j, i) == 0):
            rs = k
            break
        for k in range(rs, end_column):
          if(datum.getPixel(j, i) != 0):
            re = k - 1
        if(re > rs):
          curr = re - rs
        else:
          curr = 0
    if(abs(prev - curr) == 0 ):
      features[('white', 'row', i - 1 - start_row, i - start_row, 0)] = 1
      features[('white', 'row', i - 1 - start_row, i - start_row, 1)] = 0
      features[('white', 'row', i - 1 - start_row, i - start_row, 2)] = 0
      features[('white', 'row', i - 1 - start_row, i - start_row, -1)] = 0
      features[('white', 'row', i - 1 - start_row, i - start_row, -2)] = 0
    elif((abs(prev - curr) <= 3)):
      if(prev - curr > 0):
        features[('white', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, 1)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, 2)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, -1)] = 1
        features[('white', 'row', i - 1 - start_row, i - start_row, -2)] = 0
      else:
        features[('white', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, 1)] = 1
        features[('white', 'row', i - 1 - start_row, i - start_row, 2)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, -1)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, -2)] = 0
    else:
      if(prev - curr > 0):
        features[('white', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, 1)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, 2)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, -1)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, -2)] = 1
      else:
        features[('white', 'row', i - 1 - start_row, i - start_row, 0)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, 1)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, 2)] = 1
        features[('white', 'row', i - 1 - start_row, i - start_row, -1)] = 0
        features[('white', 'row', i - 1 - start_row, i - start_row, -2)] = 0
  '''
  '''
  #['white', 'column', i - 1, i, num] means column i -1 has num more white pixels than row i 
  curr = 0
  for i in range(start_column, end_column + 1):
    prev = curr
    curr = 0
    for j in range(start_row, end_row + 1):
      if(datum.getPixel(x, y) == 0):
        curr += 1
    if(abs(prev - curr) == 0 ):
      features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 1
      features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
      features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
      features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
      features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
    elif((abs(prev - curr) <= 3)):
      if(prev - curr > 0):
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 1
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
      else:
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 1
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
    else:
      if(prev - curr > 0):
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 1
      else:
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 1
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
  curr = 0
  for i in range(start_column, end_column + 1):
    prev = curr
    curr = 0
    for j in range(start_row, end_row + 1):
      if(datum.getPixel(x, y) == 0):
        curr += 1
    if(abs(prev - curr) == 0 ):
      features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 1
      features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
      features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
      features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
      features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
    elif((abs(prev - curr) <= 3)):
      if(prev - curr > 0):
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 1
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
      else:
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 1
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
    else:
      if(prev - curr > 0):
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 1
      else:
        features[('white', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, 2)] = 1
        features[('white', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('white', 'column', i - 1 - start_column, i - start_column, -2)] = 0
  '''
  '''
  curr = 0
  for i in range(start_column, end_column + 1):
    prev = curr
    curr = 0
    for j in range(start_row, end_row + 1):
      if(datum.getPixel(i, j) > 0):
        curr += 1
    if(abs(prev - curr) == 0 ):
      features[('black', 'column', i - 1 - start_column, i - start_column, 0)] = 1
      features[('black', 'column', i - 1 - start_column, i - start_column, 1)] = 0
      features[('black', 'column', i - 1 - start_column, i - start_column, 2)] = 0
      features[('black', 'column', i - 1 - start_column, i - start_column, -1)] = 0
      features[('black', 'column', i - 1 - start_column, i - start_column, -2)] = 0
    elif((abs(prev - curr) <= 3)):
      if(prev - curr > 0):
        features[('black', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, -1)] = 1
        features[('black', 'column', i - 1 - start_column, i - start_column, -2)] = 0
      else:
        features[('black', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, 1)] = 1
        features[('black', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, -2)] = 0
    else:
      if(prev - curr > 0):
        features[('black', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, 2)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, -2)] = 1
      else:
        features[('black', 'column', i - 1 - start_column, i - start_column, 0)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, 1)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, 2)] = 1
        features[('black', 'column', i - 1 - start_column, i - start_column, -1)] = 0
        features[('black', 'column', i - 1 - start_column, i - start_column, -2)] = 0
    '''

    # for k in range(-1 * DIGIT_DATUM_WIDTH, DIGIT_DATUM_WIDTH):
    #   if k != prev - curr:
    #     features[('black', 0, 0, i - 1 - start_row, i - start_row, k)] = 0




  return features
def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)
  return features

def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  features =  basicFeatureExtractorFace(datum)

  flag = 0
  start_row = 0
  end_row = 0
  start_column = 0
  end_column = 0

  #check start row
  for x in range(FACE_DATUM_HEIGHT):
    flag = 0
    for y in range(FACE_DATUM_WIDTH):
      if(datum.getPixel(y, x) > 0):
        flag = 1
        start_row = x
        break
    if flag != 0:
      break

  #check end row
  for x in range(FACE_DATUM_HEIGHT):
    flag = 0
    for y in range(FACE_DATUM_WIDTH):
      if(datum.getPixel(FACE_DATUM_WIDTH - y - 1, FACE_DATUM_HEIGHT - x - 1) > 0):
        flag = 1
        end_row = FACE_DATUM_HEIGHT - x - 1
        break
    if flag != 0:
      break
  
  #check start column
  for x in range(FACE_DATUM_WIDTH):
    flag = 0
    for y in range(FACE_DATUM_HEIGHT):
      if(datum.getPixel(x, y) > 0):
        flag = 1
        start_column = x
        break
    if flag != 0:
      break

  #check end column
  for x in range(FACE_DATUM_WIDTH):
    flag = 0
    for y in range(FACE_DATUM_HEIGHT):
      if(datum.getPixel(FACE_DATUM_WIDTH - x - 1, FACE_DATUM_HEIGHT - y - 1) > 0):
        flag = 1
        end_column = FACE_DATUM_WIDTH - x - 1
        break
    if flag != 0:
      break

  hight = float(end_row - start_row)
  width = float(end_column - start_column)

  

  for x in range(start_column, end_column + 1):
    black = 0
    for y in range(start_row, end_row + 1):
      black += datum.getPixel(x, y) % 1
    features[(x - start_column , 'column', black)] = 1


  for x in range(start_row, end_row + 1):
    black = 0
    for y in range(start_column, end_column + 1):
      black += datum.getPixel(y, x) % 1
    features[(x - start_row , 'row' , black)] = 1
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print ("===================================")
          print ("Mistake on example %d" % i)
          print ("Predicted %d; truth is %d" % (prediction, truth))
          print ("Image: ")
          print (rawTestData[i])
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print ("new features:", pix)
            continue
      print (image)  

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default= 100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print ("Doing classification")
  print ("--------------------")
  print ("data:\t\t" + options.data)
  print ("classifier:\t\t" + options.classifier)
  if not options.classifier == 'minicontest':
    print ("using enhanced features?:\t" + str(options.features))
  else:
    print ("using minicontest feature extractor")
  print ("training set size:\t" + str(options.training))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print ("Unknown dataset", options.data)
    print (USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training <= 0:
    print ("Training set size should be a positive integer (you provided: %d)" % options.training)
    print (USAGE_STRING)
    sys.exit(2)
    
  if options.smoothing <= 0:
    print ("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
    print (USAGE_STRING)
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print ("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
      print (USAGE_STRING)
      sys.exit(2)

  if(options.classifier == "mostFrequent"):
    classifier = mostFrequent.MostFrequentClassifier(legalLabels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print ("using automatic tuning for naivebayes")
        classifier.automaticTuning = True
    else:
        print ("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "mira"):
    classifier = mira.MiraClassifier(legalLabels, options.iterations)
    if (options.autotune):
        print ("using automatic tuning for MIRA")
        classifier.automaticTuning = True
    else:
        print ("using default C=0.001 for MIRA")
  elif(options.classifier == 'minicontest'):
    import minicontest
    classifier = minicontest.contestClassifier(legalLabels)
  else:
    print ("Unknown classifier:", options.classifier)
    print (USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data  
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
    
  
  # Extract features
  print ("Extracting features...")

  #randomly choose 10% percentage of the dataset from the training set
  percentage = 0.1
  rand_sample = [i for i in sorted(random.sample(range(numTraining), int(numTraining * percentage)))]
  sample_rawTrainingData = [rawTrainingData[i] for i in rand_sample]
  sample_trainingLabels = [trainingLabels[i] for i in rand_sample]
  trainingData = list(map(featureFunction, sample_rawTrainingData))
  validationData = list(map(featureFunction, rawValidationData))
  testData = list(map(featureFunction, rawTestData))
  
  # Conduct training and testing
  print ("Training...")
  start = timeit.default_timer()
  classifier.train(trainingData, sample_trainingLabels, validationData, validationLabels)
  stop = timeit.default_timer()
  print("training time is: ", stop - start)
  print ("Validating...")
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print (str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
  print ("Testing...")
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print (str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  # do odds ratio computation if specified at command line
  if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
    label1, label2 = options.label1, options.label2
    features_odds = classifier.findHighOddsFeatures(label1,label2)
    if(options.classifier == "naiveBayes" or options.classifier == "nb"):
      string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
    else:
      string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)    
      
    print (string3)
    printImage(features_odds)

  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)