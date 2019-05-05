# image_recognition
## Acknowledgement: 
This project is based on the one created by Dan Klein and John DeNero that
was given as part of the programming assignments of Berkeley's CS188 course.
In this project, we designed two classifiers: a Naive Bayes classifier and a Perceptron classifier.
Using these classifiers, we will mainly perform two tasks: optical character recognition(OCR) and face
detection. There are two data sets: a set of scanned handwritten digit images and a set of face images
in which edges have already been detected. We will design and extract features from the given image
files using for both classifiers. We will start with 10% of the data points for training, and increase the
training set by 10% each time until we can use 100% of the data points for training and use a fixed
number of 100 samples for testing. After we finish implementing these two classifiers, we will compare
their performances and discuss the results.


## Result:
Except for accomplishing the implementation of building Naive Bayes Algorithm and Perceptron Algorithm from scratch, we also design two different feature enhance algorithms that helps improving the prediction accuracy. Comparing to the baseline algorithm, our first feature enhance algorithm helps increasing the validation accucary from 82% to 90% for digit recognition using the whole 5000 training data, and from 88% to 96% for face binary classification when using the whole 451 training data. Without lossing prediction accuracy, our second feature enhace algorithm helps reducing the time complexity significantly. The runtime drops from 591.72s using perceptron with first feature enhance algorithm, 3 iterations, and 5000 training data, to 208.97s using perceptron with second feature enhance algorithm, 3 iteration, and 5000 training data. Detailed can be found in our report.
