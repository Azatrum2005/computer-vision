import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# import tensorflow_hub as hub
import mediapipe as mp
import time
import cv2 as cv
import csv
import math
import random
import threading
import queue
layers=tf.keras.layers
Model=tf.keras.Model 
Input=layers.Input
Sequential=tf.keras.models.Sequential
Conv2D=layers.Conv2D
MaxPooling2D=layers.MaxPooling2D
GlobalAveragePooling2D=layers.GlobalAveragePooling2D
BatchNormalization=layers.BatchNormalization
LeakyReLU=layers.LeakyReLU
Activation=layers.Activation
Dropout=layers.Dropout
Dense=layers.Dense
Flatten=layers.Flatten
EarlyStopping=tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau=tf.keras.callbacks.ReduceLROnPlateau
ImageDataGenerator=tf.keras.preprocessing.image.ImageDataGenerator
tf.config.run_functions_eagerly(False)  # Enable eager execution
# print(tf.config.experimental.list_physical_devices())

cap=cv.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

ct,pt=0,0
mpface=mp.solutions.face_detection
face=mpface.FaceDetection(min_detection_confidence=0.8)
mpdraw=mp.solutions.drawing_utils

# chunk_size = 100
# chunks = pd.read_csv('trainlabel.csv', chunksize=chunk_size, delimiter=',')
# for chunk in chunks:
#     X_train = np.loadtxt(chunk)[:, :-1]
#     X_train = X_train.reshape(100,300, 300,3)
#     print("Shape of X_train: ", X_train.shape)
#     plt.imshow(X_train[60])
#     plt.show()
#     break

# x_train = np.loadtxt('trainlabel.csv', delimiter = ',',dtype=np.float16)[:,:-1]
# # X_test = np.loadtxt('trainlabelValidGray.csv', delimiter = ',',dtype=np.float16)
# # # X_train=np.concatenate((X_train,np.loadtxt('trainlabel.csv', delimiter = ',',dtype=np.float16)[1024:,:-1]),axis=0)
# Y_train = np.loadtxt('trainlabel.csv', delimiter = ',',dtype=np.float16)[:,-1:]
# # Y_test = np.loadtxt('trainlabelValid.csv', delimiter = ',',dtype=np.float16)[:,-1:]
# # # Y_train=np.concatenate((Y_train,np.loadtxt('trainlabel.csv', delimiter = ',',dtype=np.float16)[1024:,-1:]),axis=0)

# X_train = x_train.reshape(len(x_train),200, 200,1)
# # Xt_train=np.zeros(shape=(len(x_train),224,224),dtype=np.float32)
# # X_train=np.zeros(shape=(len(x_train),224,224,1),dtype=np.float32)
# Y_train = Y_train.reshape(len(Y_train), 1)
# # X_test = X_test.reshape(len(X_test),200, 200,1)
# # Y_test = Y_test.reshape(len(Y_test), 1)
# print("Shape of X_train: ", X_train.shape)
# print("Shape of Y_train: ", Y_train.shape)
# # # print("Shape of X_test: ", X_test.shape)
# # # print("Shape of Y_test: ", Y_test.shape)
# # # # print(X_train[11])
# X_train = X_train/255.0
# print("prepared data")
# # X_train=np.concatenate((X_train,X_train-0.3),axis=0)
# # Y_train=np.concatenate((Y_train,Y_train),axis=0)
# # X_test = X_test/255.0

# for i in range(len(x_train)):
#     # i=random.randint(0,len(Y_train))
#     print(i,Y_train[i])
#     # cv.imshow('train',cv.cvtColor(X_train[i],cv.COLOR_RGB2BGR))
#     # cv.imshow('train',x_train[i])
#     # cv.waitKey()
#     # x_train[i]=x_train[i].reshape(200, 200,1)
#     Xt_train[i]=cv.resize(x_train[i],(224,224))
#     X_train[i]=Xt_train[i].reshape(224, 224,1)
#     # cv.imshow('train',X_train[i])
#     # cv.waitKey()
# print("Shape of X_train: ", X_train.shape)
# model = Sequential([
#     Conv2D(32, (3,3), strides=(1,1), padding="same", input_shape=(200, 200, 1)),
#     BatchNormalization(),
#     Activation('swish'),
#     # LeakyReLU(alpha=0.1),
#     MaxPooling2D((2,2), strides=(2,2)),

#     Conv2D(32, (3,3), strides=(1,1), padding="same"),
#     BatchNormalization(),
#     Activation('swish'),
#     # LeakyReLU(alpha=0.1),

#     Conv2D(64, (3,3), strides=(1,1), padding="same"),
#     BatchNormalization(),
#     Activation('swish'),
#     # LeakyReLU(alpha=0.1),
#     MaxPooling2D((2,2), strides=(2,2)),

#     Conv2D(128, (3,3), strides=(1,1), padding="same"),
#     BatchNormalization(),
#     Activation('swish'),
#     # LeakyReLU(alpha=0.1),
#     # Dropout(0.2),
#     MaxPooling2D((2,2), strides=(2,2)),

#     Conv2D(128, (3,3), strides=(1,1), padding="same"),
#     BatchNormalization(),
#     Activation('swish'),
#     # LeakyReLU(alpha=0.1),
#     Dropout(0.2),
#     MaxPooling2D((2,2), strides=(2,2)),

#     GlobalAveragePooling2D(),
#     # Flatten(), 
#     # Dense(128,activation='relu'),#activation='relu'
#     # LeakyReLU(alpha=0.1),
#     # Dropout(0.2),
#     Dense(128,activation='swish'),#activation='relu'
#     # LeakyReLU(alpha=0.1),
#     # Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])
# # feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
# # pretrained_model = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 1), trainable=False)
# # model = Sequential([
# #     pretrained_model,
# #     Dense(100, activation='relu'),
# #     Dense(1, activation='sigmoid')
# # ])
# # model=tf.keras.models.load_model('face_recognitionCNNGray.h5')
# # print(len(model.layers),model.layers[:17])
# # for layer in model.layers[:17]:
# #     layer.trainable = True
# METRICS = [
#       tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#       tf.keras.metrics.Precision(name='precision'),
#       tf.keras.metrics.Recall(name='recall')
# ]
# model.compile(loss = 'binary_crossentropy', optimizer =tf.optimizers.Adam(0.0007), metrics =METRICS  )
# early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
# lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=4)

# indices = np.random.permutation(len(X_train))  #shuffle
# X_train = X_train[indices]
# Y_train = Y_train[indices]
# print("training begins.......")
# model.fit(X_train, Y_train,batch_size=20,epochs=50,callbacks=[early_stopping, lr_scheduler])

# # count=0
# # nolist=[i for i in range(521,526)]
# # # l.append([i for i in range(5580,5596)])
# # # model.fit(X_test, Y_test, 
# #         #  epochs = 1,
# #         #  batch_size = 6, verbose=0)
# # while count<5:
# #     while len(trainlist)>0:
# #         i=random.randint(0,len(trainlist)-1)
# #         a=trainlist.pop(i)
# #         if a not in nolist:
# #             model.fit(X_train[a].reshape(1,200,200,1), Y_train[a].reshape(1,1), 
# #                     #   validation_data=(X_test, Y_test),
# #                       epochs = 1,
# #                     #   callbacks=[early_stopping, lr_scheduler],
# #                       batch_size = 1, verbose=0)
# #     count+=1
# #     print(f"Completed Epoch={count}")
# print(model.evaluate(X_train, Y_train, batch_size=1))
# # print(model.evaluate(X_test, Y_test, batch_size=1))
# model.save('face_recognitionCNNGrayTRY.h5')

# data_queue = queue.Queue()
# def train_thread():
#     while True:
#         if not data_queue.empty():
#             data, label = data_queue.get()
#             frmodelg.fit(data, label, epochs=1, batch_size=1, verbose=1)
#             data_queue.task_done()

frmodelg=tf.keras.models.load_model('face_recognitionCNNGray50.h5')
# extractor_model = Model(inputs=frmodelg.layers[0].input, outputs=frmodelg.layers[20].output)
# GAP2D = extractor_model.predict(X_train)
# print(GAP2D.shape)
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# params = {
#     'C': [0.1, 1, 2,3,4,5,6, 10],
#     'gamma': [1e-4,1e-3, 1e-2, 1e-1, 1,2,3,4],
#     'kernel': ['rbf']
# }
# grid = GridSearchCV(SVC(), param_grid=params, cv=8, scoring='accuracy')
# grid.fit(GAP2D, Y_train)
# print("Best Params:", grid.best_params_)
# print("Best Score:", grid.best_score_)
# svm = SVC(kernel="rbf",C=4,gamma=0.01)
# print(svm.fit(GAP2D, Y_train))
# print(svm.score(GAP2D , Y_train))
# with open('SVMxCNNv2','wb') as file:
#     pickle.dump(svm,file)
with open('SVMxCNN','rb') as file:
    svm = pickle.load(file)

# frmodelg.compile(loss = 'binary_crossentropy', optimizer =tf.optimizers.Adam(0.0007), metrics =METRICS  )
# trainthread = threading.Thread(target=train_thread,daemon=True)
# trainthread.start()
# frmodelc=tf.keras.models.load_model('face_recognitionCNNv4.h5')
while True:
    img_dict={}
    flattened_img=[]
    ret,frame=cap.read()
    # img=frame
    img=cv.flip(frame,1)
    imgrgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=face.process(imgrgb)
    if results.detections:
        for id,d in enumerate(results.detections):
            h,w,c=img.shape
            s=d.score[0]
            v=d.location_data.relative_bounding_box
            co={1:int(v.xmin*w),2:int(v.ymin*h),3:int(v.width*w),4:int(v.height*h)}

            cv.rectangle(img,(co[1],co[2]),(co[3]+co[1],co[4]+co[2]),(200,200,200),2)
            cv.line(img,(co[1]-5,co[2]-5),(co[1]-5,co[2]+30),(0,0,0),5)
            cv.line(img,(co[1]-5,co[2]-5),(co[1]+50,co[2]-5),(0,0,0),5)
            cv.line(img,(co[1]+co[3]+5,co[2]+co[4]+5),(co[1]+co[3]+5,co[2]+co[4]-30),(0,0,0),5)
            cv.line(img,(co[1]+co[3]+5,co[2]+co[4]+5),(co[1]+co[3]-50,co[2]+co[4]+5),(0,0,0),5)

            midp=(co[1]+ int(co[3]/2),co[2]+int(co[4]/2))
            # imgg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            # imgs=imgrgb[midp[1]-200:midp[1]+100,midp[0]-150:midp[0]+150]/255
            imgs=imgrgb[co[2]-50:co[2]+co[4],co[1]-25:co[1]+co[3]+25]
            imgsg=np.array([0])
            if imgs.size>0:
                # imgs=np.array(imgs)
                imgg=cv.resize(imgrgb[co[2]-50:co[2]+co[4],co[1]-25:co[1]+co[3]+25],(200,200))
                imgsc=cv.resize(imgs,(200,200))/255
                imgs=cv.cvtColor(imgs,cv.COLOR_RGB2GRAY)/255
                imgsg=cv.resize(imgs,(200,200))
                imgsg=imgsg.reshape(1,200,200,1)
                imgsc=imgsc.reshape(1,200,200,3)

            pred="Couldn't Detect"
            if(imgsg.shape!=(1,200,200,1)):
                print(pred)
                cv.putText(img,f"{      pred}",(co[1]+7,co[2]-10),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(100,200,0),1)
            if imgsg.shape==(1,200,200,1):
                # pred=model.predict(imgs)
                # pred=frmodelg.predict(imgsg)
                extractor_model = Model(inputs=frmodelg.layers[0].input, outputs=frmodelg.layers[20].output)
                GAP2D = extractor_model.predict(imgsg)
                pred=svm.predict(GAP2D)[0]
                # data_queue.put((imgsg, np.array([1]).reshape(1,1)))
                # frmodelg.fit(imgsg, np.array([1]).reshape(1,1), epochs = 1,batch_size = 1, verbose=1)
                # predc=frmodelc.predict(imgsc)
                # pred=math.sqrt(predg*predc)
                print(pred)
                if(co[3]>=500):
                    m=pred
                    cv.putText(img,str(math.floor(m*100))+"%    Move Away",(co[1]+7,co[2]-10),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(100,200,0),1)
                elif pred>=0.95:
                    print("Murtaza")
                    m=pred
                    cv.putText(img,str(math.floor(m*100))+"%    Murtaza",(co[1]+7,co[2]-10),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(100,200,0),1)
                elif pred<=0.8:
                    print("unkown")
                    m=pred
                    cv.putText(img,str(math.floor(m*100))+"%    Unknown",(co[1]+7,co[2]-10),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(100,200,0),1)
                elif(pred<0.95 and pred>0.8 or co[3]<=200):
                    m=pred
                    cv.putText(img,str(math.floor(m*100))+"%    Unclear",(co[1]+7,co[2]-10),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(100,200,0),1)
    ct=time.time()
    fps=1/(ct-pt)
    pt=ct

    cv.putText(img,"fps:"+str(int(fps)),(10,30),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
    cv.imshow("faceRecognition",img)
    # cv.imshow("face",cv.cvtColor(imgg,cv.COLOR_RGB2BGR))
    key=cv.waitKey(10)
    if key==27:
        break
cap.release()
cv.destroyAllWindows()
# frmodelg.save('face_recognitionCNNGray.h5')