import pandas as pd
import numpy as np
# indices = np.random.permutation(len(X_train))  #shuffle
# X_train = X_train[indices]
# Y_train = Y_train[indices]
chunk_size = 400
chunks = pd.read_csv('shuffled_trainlabel.csv', chunksize=chunk_size, delimiter=',',dtype='float16')
for df in chunks:
    dfshuffle=df.sample(frac=1)
    dfshuffle.to_csv("shuffled_trainlabel2.csv",mode='a',index=False)
# df= pd.read_csv('trainlabel.csv', delimiter=',',dtype='float32')
# dfshuffle=df.sample(frac=1)
# dfshuffle.to_csv("shuffled_trainlabel.csv",mode='a',index=False)