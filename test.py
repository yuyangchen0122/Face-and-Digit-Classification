import samples
import util
import numpy as np
import os
from samples import Datum
from samples import readlines
from dataClassifier import DIGIT_DATUM_HEIGHT,DIGIT_DATUM_WIDTH,contestFeatureExtractorDigit
from samples import IntegerConversionFunction

featureFunction = contestFeatureExtractorDigit
rawTrainingData = samples.loadDataFile("digitdata/testimages", 1, 28, 28)
trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 1)
fin = readlines("digitdata/testimages")
fin.reverse()

a = ['+', ' ', '#']
print(IntegerConversionFunction(a))
data = []
items = []
for j in range(28):
  data.append(list(fin.pop()))

for i in range(28):
	print(data[i])

items.append(Datum(data,28, 28))
print(items[0].getPixels())