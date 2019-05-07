import math
import time
import statistics

def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))

def getfeature(image,lable, no):
    line = image[no]
    length = image_length/4
    count = 0
    #print(length)
    #print(len(line[0]))
    #print(len(line))
    for i  in range(length):
        for j in range(length):
            if(line[i][j] != ' '):
                count+=1

    one = count
    count = 0
    for i  in range(length):
        for j in range(length,2*length):
            if(line[i][j] != ' '):
                count+=1

    two = count
    count = 0
    for i  in range(length):
        for j in range(length*2,length*3):
            if(line[i][j] != ' '):
                count+=1

    three = count
    count = 0
    for i  in range(length):
        for j in range(length*3,length*4):
            if(line[i][j] != ' '):
                count+=1

    four = count
    count = 0
    for i  in range(length,length*2):
        for j in range(length):
            if(line[i][j] != ' '):
                count+=1

    one2 = count
    count = 0
    for i  in range(length,length*2):
        for j in range(length,2*length):
            if(line[i][j] != ' '):
                count+=1

    two2 = count
    count = 0
    for i  in range(length,length*2):
        for j in range(length*2,length*3):
            if(line[i][j] != ' '):
                count+=1

    three2 = count
    count = 0
    for i  in range(length,length*2):
        for j in range(length*3,length*4):
            if(line[i][j] != ' '):
                count+=1

    four2 = count
    count = 0
    for i  in range(length*2,length*3):
        for j in range(length):
            if(line[i][j] != ' '):
                count+=1

    one3 = count
    count = 0
    for i  in range(length*2,length*3):
        for j in range(length,2*length):
            if(line[i][j] != ' '):
                count+=1

    two3 = count
    count = 0
    for i  in range(length*2,length*3):
        for j in range(length*2,length*3):
            if(line[i][j] != ' '):
                count+=1

    three3 = count
    count = 0
    for i  in range(length*2,length*3):
        for j in range(length*3,length*4):
            if(line[i][j] != ' '):
                count+=1

    four3 = count
    count = 0


    for i  in range(length*3,length*4):
        for j in range(length):
            if(line[i][j] != ' '):
                count+=1

    one4 = count
    count = 0
    for i  in range(length*3,length*4):
        for j in range(length,2*length):
            if(line[i][j] != ' '):
                count+=1

    two4 = count
    count = 0
    for i  in range(length*3,length*4):
        for j in range(length*2,length*3):
            if(line[i][j] != ' '):
                count+=1

    three4 = count
    count = 0
    for i  in range(length*3,length*4):
        for j in range(length*3,length*4):
            if(line[i][j] != ' '):
                count+=1

    four4 = count
    count = 0

    result = [one,two,three,four,one2,two2,three2,four2,one3,two3,three3,four3,one4,two4,three4,four4,lable[no]]
    return result

def euc_distance(a,b):
    result = (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 + (a[3]-b[3])**2 + (a[4]-b[4])**2  + (a[5]-b[5])**2  + (a[6]-b[6])**2  + (a[7]-b[7])**2  + (a[8]-b[8])**2 + (a[9]-b[9])**2 + (a[10]-b[10])**2 + (a[11]-b[11])**2 + (a[12]-b[12])**2 + (a[13]-b[13])**2 + (a[14]-b[14])**2 + (a[15]-b[15])**2
    result = math.sqrt(result)
    return result

def getdistance(feature_result,new_feature,distance_result):
    a = [feature_result[0],feature_result[1],feature_result[2],feature_result[3],feature_result[4],feature_result[5],feature_result[6],feature_result[7],feature_result[8],feature_result[9],feature_result[10],feature_result[11],feature_result[12],feature_result[13],feature_result[14],feature_result[15]]
    b = [new_feature[0],new_feature[1],new_feature[2],new_feature[3],new_feature[4],new_feature[5],new_feature[6],new_feature[7],new_feature[8],new_feature[9],new_feature[10],new_feature[11],new_feature[12],new_feature[13],new_feature[14],new_feature[15]]
    tempdistance = euc_distance(a ,b)
    temphold = feature_result + [tempdistance]
    #print(temphold)
    return temphold

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num

def findknn(distance_result,knumber):
    result = []
    for i in range(knumber):
        result.append(distance_result[i][16])
    #print(result)
    final = most_frequent(result)
    return final



#load image to list
#load feature
#percent = input("what percent of trainingdata you want to use?(Input number 1 - 10)")
acculist = []
timelist = []
for pp in range(1,11):
    percent = pp
    lable=[l[:-1] for l in open("facedata/facedatatrainlabels").readlines()]
    image_length = 28
    all=[l[:-1] for l in open("facedata/facedatatrain").readlines()]
    total = len(lable)
    n_image = total*percent/10
    line = []
    image = []
    for i in range(n_image):
        start = i*image_length
        end = start+image_length
        for j in range(start,end):
            line.append(all[j])
        image.append(line)
        line = []
    #for i in range(image_length):
    #    print(image[0][i])

    #print(lable)
    feature_result = []
    for i in range(n_image):
        no = i
        feature = getfeature(image,lable,no)
        feature_result.append(feature)

    #print(feature_result)



    new_lable=[l[:-1] for l in open("facedata/facedatatestlabels").readlines()]
    new_total = len(new_lable)

    #load image from test
    new_all=[l[:-1] for l in open("facedata/facedatatest").readlines()]

    new_n_image = new_total
    new_line = []
    new_image = []
    for i in range(new_n_image):
        #print(i)
        new_start = i*image_length
        new_end = new_start+image_length
        for j in range(new_start,new_end):
            #print(j)
            new_line.append(new_all[j])
        new_image.append(new_line)
        new_line = []
    #print(len(new_image))
    #for i in range(image_length):
    #    print(new_image[999][i])
    #print(new_lable)
    finalset = []
    knumber = 11
    startime = time.time()

    for j in range(new_n_image):
        new_feature = getfeature(new_image,new_lable,j)
        distance_result = []
        for i in range(n_image):
            #print(i)
            distance = getdistance(feature_result[i],new_feature,distance_result)
            distance_result.append(distance)

        distance_result.sort(key=lambda distance_result: distance_result[17])
            #print(distance_result)

        final = findknn(distance_result,knumber)
        #print(final)
        finalset = finalset+[final]
    #print(finalset)
    dif = differences(finalset, new_lable)
    endtime = time.time()
    finaltime = endtime - startime
    print('percent is :' + repr(percent))
    print("time: "+ repr(finaltime))
    #print(dif)
    correct = new_n_image - dif
    #print(correct)
    rate = 100 * float(correct)/float(new_n_image)
    print("accuracy is " + repr(rate) +"%")
    acculist = acculist+[rate]
    timelist = timelist+[finaltime]
#print(acculist)
#print(timelist)

print("mean accuracy:" + repr(statistics.mean(acculist)))
print("Standard Deviation accuracy:" + repr(statistics.stdev(acculist)))

print("mean time:" + repr(statistics.mean(timelist)))
print("Standard Deviation time:" + repr(statistics.stdev(timelist)))
