import os
import random
class ImageDataSet:

    def __init__(self, height, width, labeldomain=None):
        self.height = height
        self.width = width
        self.number = 0
        if labeldomain != None:
            self.labeldomain = labeldomain


    def loadDataSet(self, Pathim, path, n):
        self.loadImageData(Pathim, n)
        self.loadLabelData(path, n)

    def loadImageData(self, Filename, n):
        self.images = []
        if n != -1:
            self.number = n
            with open(Filename, 'r') as f:
                for x in range(n):
                    one_image = []
                    for y in range(self.height):
                        line = f.readline()
                        if len(line) != self.width + 1:
                            raise Expection("Inavalid ImageData")
                        one_image.append(list(map(charToint,list(line)[:-1])))
                    self.images.append(one_image)
                f.close()
        else:
            number_count = 0
            with open(Filename, 'r') as f:
                one_image = []
                count = 0
                for line in f:
                    one_image.append(list(map(charToint,list(line)[:-1])))
                    count += 1
                    if count == self.height:
                        self.images.append(one_image)
                        one_image = []
                        number_count += 1
                        count = 0
                self.number = number_count
                f.close()
    def loadLabelData(self, labelfilename, n):
        if n != self.number:
            print("Unequal number of image and label, will cause problem")
        self.labels = []
        count = 0
        with open(labelfilename, 'r') as f:
            for line in f:
                self.labels.append(list(line)[0])
                count += 1
                if count == n:
                    break
            f.close()
        if count != n:
            print("load fewer label than input,{},{}",count,n)

    def shuffle(self):
        Tmp = list(zip(self.images, self.labels))
        random.shuffle(Tmp)
        self.images , self.labels = zip(*Tmp)

    def orderedout(self, Perce):
        if Perce > 1:
            Perce = Perce * 0.01
        Finall = int(self.number*Perce -1)
        return self.images[:Finall], self.labels[:Finall]

    def shuffleout(self, Perce):
        if Perce > 1:
            Perce = Perce * 0.01
        Tmp = list(zip(self.images, self.labels))
        random.shuffle(Tmp)
        Finall = int(self.number*Perce -1)
        Tmp = Tmp[:Finall]
        images , labels = zip(*Tmp)

        return images,labels



def charToint(char):
    if char == ' ':
        return 0
    elif char == '+':
        return 0.5
    elif char == '#':
        return 1





