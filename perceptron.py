import numpy as np
from imageloader import ImageDataSet
import matplotlib.pyplot as plt
import timeit


class Perceptron():
	def __init__(self, nfeature, labels):
		print('Initilizing Perceptron')
		self.name = "Perceptron"
		self.weights = np.zeros((len(labels), nfeature))
		self.w0 = np.zeros(len(labels))
		self.labels = labels

	def train(self,featurel, labell , Rat):
		
		for i in range(len(featurel)):
			
			feature = np.array(featurel[i]).flatten()
			label = labell[i]
			for l in range(len(self.labels)):
				label_c = self.labels[l]
				W = self.weights[l]
				f = np.dot(feature, W) 
				if f < 0 and label == label_c:
					for j in range(len(W)):
						W[j] = W[j] + Rat*feature[j]
				elif f >=0 and label != label_c:
					for j in range(len(W)):
						W[j] = W[j] - Rat*feature[j]

	def classify(self, datalist):
		result = []
		for image in datalist:
			data = np.array(image).flatten()
			temp = float("-inf")
			index = 0
			for l in range(len(self.labels)):
				weight = self.weights[l]
				m = np.dot(data, weight) 
				if m >= temp:
					temp = m
					index = l
			result.append(self.labels[index])
		return result


def Accuracy(FirstL, SecondL):
	if len(SecondL) != len(FirstL):
		return 0
	aint = 0
	for x, y  in zip(FirstL,SecondL):
		if x == y:
			aint += 1
	return aint/len(FirstL)

def runProceptron(data_train, Data_test):
    print("--------------")
    ti = int(input("How many times for trainning: "))
    Rat = float(input("Please put Ratio: "))
    tNumbers = data_train.number
    #first - try with ordered datas 
    for p in range(10,101,10):
        # images, labels = data_train.orderedout(p)
        start = timeit.default_timer()
        images, labels = data_train.shuffleout(p)        
        al = []
        il = []
        pc = Perceptron(data_train.width*data_train.height, data_train.labeldomain)
        testmean = 0
        for i in range(ti):
            pc.train(images, labels, Rat)
            x = pc.classify(Data_test.images)
            a = Accuracy(x, Data_test.labels)
            testmean+=a
            al.append(a*100)
            il.append(i+1)
            print(a*100)
        plt.plot(il, al, label="size=%d"%(p*0.01*tNumbers) )
        stop = timeit.default_timer()
        print('Time(mean): {}, std: {}, testmean:{}'.format(((stop-start)/3), np.std(al),testmean/3))
    leg = plt.legend( ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel("times of trainning")
    plt.ylabel("_Accuracy")
    plt.show()

def dataloader_face(Tranum = -1, Testnum = -1):
    data = ImageDataSet(70,60, labeldomain=['0', '1'])
    # data.loadImageData("facedata/facedatatrain", Tranum)
    # data.loadLabelData("facedata/facedatatrainlabels", data.number)
    data.loadImageData("facedata/facedatatrain", Tranum)
    data.loadLabelData("facedata/facedatatrainlabels", data.number)
    Data_test = ImageDataSet(70,60)
    Data_test.loadImageData("facedata/facedatavalidation", Testnum)
    Data_test.loadLabelData("facedata/facedatavalidationlabels", Data_test.number)
    # Data_test.loadImageData("facedata/facedatatest", Testnum)
    # Data_test.loadLabelData("facedata/facedatatestlabels", Data_test.number)   
    return data, Data_test 

def dataloader_digit(Tranum = -1, Testnum = -1):
    data = ImageDataSet(28,28, labeldomain=['0', '1','2','3','4','5','6','7','8','9'])
    # data.loadImageData("digitdata/trainingimages", Tranum)
    # data.loadLabelData("digitdata/traininglabels", data.number)
    data.loadImageData("digitdata/trainingimages", Tranum)
    data.loadLabelData("digitdata/traininglabels", data.number)
    Data_test = ImageDataSet(28,28)
    Data_test.loadImageData("digitdata/validationimages", Testnum)
    Data_test.loadLabelData("digitdata/validationlabels", Data_test.number)
    # Data_test.loadImageData("digitdata/testimages", Testnum)
    # Data_test.loadLabelData("digitdata/testlabels", Data_test.number)
    return data, Data_test

def main():
	train = None
	test  = None
	while True:
		choice = input("(enter 1. for digit, 2. for face or Q to exit): ")
		if choice == '1':
			print("Choosen Digit")
			train, test = dataloader_digit()
			break
		elif choice == '2':
			print("Choosen Face")
			train, test = dataloader_face()
			break
		elif choice == 'Q':
			print("Cya~~!")
			return 
		else:
			print("Please give validate input")
	runProceptron(train,test)

if __name__ == '__main__':
	main()
