import time
import random
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow import keras

x_train = np.loadtxt('train2.csv', delimiter = ',')/255.0
X_train= np.loadtxt('train2.csv', delimiter = ',').reshape(42,x_train.shape[0])
Y_train = np.loadtxt('labeltrain2.csv', delimiter = ',').reshape(1, X_train.shape[1])
'''x_test = np.loadtxt('on_test.csv', delimiter = ',')/255.0
X_test = np.loadtxt('on_test.csv', delimiter = ',').reshape(63,2233)
Y_test = np.loadtxt('label_test.csv', delimiter = ',').reshape(1, X_test.shape[1])'''


print(X_train.shape)
print(Y_train.shape)
'''print(X_test.shape)
print(Y_test.shape)'''

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def softmax(z):
    expZ = np.exp(z)
    return expZ/(np.sum(expZ, 0))

def relu(Z):
    A = np.maximum(0,Z)
    return A

def tanh(x):
    return np.tanh(x)

def derivative_relu(Z):
    return np.array(Z > 0, dtype = 'float')

def derivative_tanh(x):
    return (1 - np.power(x, 2))

def initialize_parameters(layer_dims):
    
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / (np.sqrt(layer_dims[l-1])) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def forward_propagation(X, parameters, activation):
   
    forward_cache = {}
    L = len(parameters) // 2                  
    
    forward_cache['A0'] = X

    for l in range(1, L):
        forward_cache['Z' + str(l)] = parameters['W' + str(l)].dot(forward_cache['A' + str(l-1)]) + parameters['b' + str(l)]
        
        if activation == 'tanh':
            forward_cache['A' + str(l)] = tanh(forward_cache['Z' + str(l)])
        else:
            forward_cache['A' + str(l)] = relu(forward_cache['Z' + str(l)])
            

    forward_cache['Z' + str(L)] = parameters['W' + str(L)].dot(forward_cache['A' + str(L-1)]) + parameters['b' + str(L)]
    
    if forward_cache['Z' + str(L)].shape[0] == 1:
        forward_cache['A' + str(L)] = sigmoid(forward_cache['Z' + str(L)])
    else :
        forward_cache['A' + str(L)] = softmax(forward_cache['Z' + str(L)])
    
    return forward_cache['A' + str(L)], forward_cache

def compute_cost(AL, Y,parameters,Lambda):
    m = Y.shape[1]
    L = len(parameters)//2    
    if Y.shape[0] == 1:
        cost = (1/m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    else:
        cost = -(1/m) * np.sum(Y * np.log(AL))
    
    for i in range(1,L):
        W=parameters['W'+str(i)]
        cost=cost+(Lambda*np.sum(np.square(W)))/(2*m)
    
    L2cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return L2cost

def backward_propagation(AL, Y, parameters, forward_cache, activation,Lambda):
    
    grads = {}
    L = len(parameters)//2
    m = AL.shape[1]
    
    grads["dZ" + str(L)] = AL - Y
    grads["dW" + str(L)] = 1./m * np.dot(grads["dZ" + str(L)],forward_cache['A' + str(L-1)].T) + (Lambda*np.sum(parameters['W'+str(L)]))/m
    grads["db" + str(L)] = 1./m * np.sum(grads["dZ" + str(L)], axis = 1, keepdims = True)
    
    for l in reversed(range(1, L)):
        if activation == 'tanh':
            grads["dZ" + str(l)] = np.dot(parameters['W' + str(l+1)].T,grads["dZ" + str(l+1)])*derivative_tanh(forward_cache['A' + str(l)])
        else:
            grads["dZ" + str(l)] = np.dot(parameters['W' + str(l+1)].T,grads["dZ" + str(l+1)])*derivative_relu(forward_cache['A' + str(l)])
   
        grads["dW" + str(l)] = 1./m * np.dot(grads["dZ" + str(l)],forward_cache['A' + str(l-1)].T)+(Lambda*np.sum(parameters['W'+str(l)]))/m
        grads["db" + str(l)] = 1./m * np.sum(grads["dZ" + str(l)], axis = 1, keepdims = True)

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters, activation):

    m = X.shape[1]
    y_pred, caches = forward_propagation(X, parameters, activation)
    
    if y.shape[0] == 1:
        y_pred = np.array(y_pred > 0.7, dtype = 'float')
    else:
        y = np.argmax(y, 0)
        y_pred = np.argmax(y_pred, 0)
    
    return np.round(np.sum((y_pred == y)/m), 2)

def model_L2(X, Y, layers_dims, learning_rate = 0.03, activation = 'relu', num_iterations = 3000,Lambda=0.01):

    np.random.seed(1)
    costs = []              
    
    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations+1):

        AL, forward_cache = forward_propagation(X, parameters, activation)

        cost = compute_cost(AL, Y,parameters,Lambda)

        grads = backward_propagation(AL, Y, parameters, forward_cache, activation,Lambda)

        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % (num_iterations//5) == 0:
            print("\niter:{} \t cost: {} \t train_acc:{} ".format(i, np.round(cost, 2), predict(X_train, Y_train, parameters, activation)))

        if i % 40 == 0:
            print("==", end = '')
       
    return parameters

import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Start capturing video from webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7) as hands:
    landmarks_dict={}
    frame_count=0
    layers_dims = [X_train.shape[0],50,Y_train.shape[0]] 
    lr = 0.001
    iters = 5000
    lam=0.0001
    parameters = model_L2(X_train, Y_train, layers_dims, learning_rate = lr, activation = 'relu', num_iterations = iters,Lambda=lam)
    while cap.isOpened():
        flattened_landmarks=[]
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (MediaPipe requires RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image.flags.writeable = False
        
        # Process the image and detect hands
        results = hands.process(image)
        
        # Draw hand landmarks and recognize signs
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks and reshape for model input
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h,w,c=image.shape
                    landmarks.append([int(lm.x*w),int(lm.y*h)])
                
                # Flatten the list of landmarks
                flattened_landmarks = [coord for point in landmarks for coord in point]

                # Ensure that the list is exactly 42 elements (21 landmarks * 2 coordinates)
                flattened_landmarks = flattened_landmarks[:42]

                a=np.array(flattened_landmarks)
                fp,_=forward_propagation(a.reshape(42,1), parameters, 'relu')
                if np.squeeze(fp)<0.9:
                    landmarks_dict[frame_count] = flattened_landmarks
                    frame_count += 1
                print(fp)
                fp=np.squeeze(fp)
                if fp>0.7:
                    print('open')
                    cv2.putText(image,'fist open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA)
                elif fp<0.3:
                    print('closed')
                    cv2.putText(image,'fist closed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow('Hand Sign Recognition', image)
        # Exit the loop when 'q' is pressed
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

with open('train2.csv', 'w', newline='') as train_file, open('labeltrain2.csv', 'w', newline='') as label_file:
    csv_writer = csv.writer(train_file)
    label_writer = csv.writer(label_file)

    # Write landmarks and corresponding labels
    for i in range(len(landmarks_dict)):
        csv_writer.writerow(landmarks_dict[i])
        label_writer.writerow([1])
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
