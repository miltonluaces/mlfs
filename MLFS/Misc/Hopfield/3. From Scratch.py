# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from Standard import *
import numpy as np
import random
from PIL import Image
import os
import re

#Convert matrix to a vector
def Mat2Vec(x):
    vec = np.zeros(x.shape[0] * x.shape[1])
    k = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vec[k] = x[i,j]
            k=k+1
    return vec


#Create Weight matrix for a single image
def CreateWMat(x):
    if len(x.shape) != 1:
        print("The input is not vector")
        return
    else:
        w = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(i,len(x)):
                if i == j:
                    w[i,j] = 0
                else:
                    w[i,j] = x[i]*x[j]
                    w[j,i] = w[i,j]
    return w


#Read Image file and convert it to Numpy array
def ReadImage(file,size, threshold= 145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN= pilIN.resize(size)
    #pilIN.thumbnail(size,Image.ANTIALIAS)
    imgArray = np.asarray(pilIN,dtype=np.uint8)
    x = np.zeros(imgArray.shape,dtype=np.float)
    x[imgArray > threshold] = 1
    x[x==0] = -1
    return x

#Convert Numpy array to Image file like Jpeg
def Array2Image(data, outFile = None):

    #data is 1 or -1 matrix
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    if outFile is not None:
        img.save(outFile)
    return img


#Update
def Update(w,y_vec,theta=0.5,time=100):
    for s in range(time):
        m = len(y_vec)
        i = random.randint(0,m-1)
        u = np.dot(w[i][:],y_vec) - theta

        if u > 0:
            y_vec[i] = 1
        elif u < 0:
            y_vec[i] = -1

    return y_vec


#The following is training pipeline
#Initial setting
def hopfield(train_files, test_files,theta=0.5, time=1000, size=(100,100),threshold=60, current_path=None):

    #read image and convert it to Numpy array
    print("Importing images and creating weight matrix....")

    #num_files is the number of files
    num_files = 0
    for path in train_files:
        print(path)
        x = ReadImage(file=path,size=size,threshold=threshold)
        x_vec = Mat2Vec(x)
        print(len(x_vec))
        if num_files == 0:
            w = CreateWMat(x_vec)
            num_files = 1
        else:
            tmp_w = CreateWMat(x_vec)
            w = w + tmp_w
            num_files +=1

    print("Weight matrix is done!!")


    #Import test data
    counter = 0
    for path in test_files:
        y = ReadImage(file=path,size=size,threshold=threshold)
        oshape = y.shape
        y_img = Array2Image(y)
        y_img.show()
        print("Imported test data")

        y_vec = Mat2Vec(y)
        print("Updating...")
        y_vec_after = Update(w=w,y_vec=y_vec,theta=theta,time=time)
        y_vec_after = y_vec_after.reshape(oshape)
        if current_path is not None:
            outfile = current_path+"/after_"+str(counter)+".jpeg"
            Array2Image(y_vec_after,outFile=outfile)
        else:
            after_img = Array2Image(y_vec_after,outFile=None)
            after_img.show()
        counter +=1




# Testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#First, you can create a list of input file path
trainPaths = []
trainPath = imgPath +"hopfieldTrain/"
for i in os.listdir(trainPath):
    if re.match(r'[0-9a-zA-Z-]*.jp[e]*g',i):
        trainPaths.append(trainPath+i)

#Second, you can create a list of sungallses file path
testPaths = []
testPath = imgPath +"hopfieldTest/"
for i in os.listdir(testPath):
    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
        testPaths.append(testPath+i)

#Hopfield network starts!
hopfield(train_files=trainPaths, test_files=testPaths, theta=0.5,time=20000,size=(100,100),threshold=60, current_path = trainPath)
