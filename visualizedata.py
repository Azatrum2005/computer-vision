import cv2 as cv 
import numpy as np
import csv
X_train = np.loadtxt('trainlabel.csv', delimiter = ',',dtype=np.float32)[:,:-1]
Y_train = np.loadtxt('trainlabel.csv', delimiter = ',',dtype=np.float16)[:,-1:]

X_train = X_train.reshape(len(X_train),200, 200,1)
Y_train = Y_train.reshape(len(Y_train), 1)
print("Shape of X_train: ", X_train.shape)
# print("Shape of Y_train: ", Y_train.shape)
size=40000
X_train = X_train/255.0
# indices = np.random.permutation(len(X_train))  #shuffle
# X_train = X_train[indices]
# Y_train = Y_train[indices]
# X_train=np.concatenate((X_train,X_train-0.4),axis=0)
for i in range(-45,0):
    # i=random.randint(0,len(Y_train))
    # print(i,Y_train[i])
    cv.putText(X_train[i],str(Y_train[i]),(10,20),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
    cv.imshow('train',X_train[i])
    cv.waitKey(100)
    # imgs=cv.cvtColor(X_train[i],cv.COLOR_RGB2GRAY).reshape(200,200,1)
    # flattened_img = [coord for point in imgs for color in point for coord in color]
    # flattened_img = flattened_img[:size]
    # with open('trainlabelValidGray.csv', 'a+', newline='') as trainlabel_file: #, open('label.csv', 'a+', newline='') as label_file:
    #     csv_writer = csv.writer(trainlabel_file)
    #     # label_writer = csv.writer(label_file)
    #     if len(flattened_img)==size:
    #         csv_writer.writerow(flattened_img)
    #         # label_writer.writerow([0])