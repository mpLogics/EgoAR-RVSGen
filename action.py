import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D,Activation
import pandas as pd


df = pd.read_csv("data/Splits/train_split1.csv")
gt_action = df["Action"]

model = Sequential()
model.add(Input(shape=(5,5)))
model.add(Flatten())
model.add(Dense(106))
model.add(Activation('softmax'))
optimizer = Adam(learning_rate=0.001)
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
X = np.load("Actions.npz",allow_pickle=True)
Y = pd.read_csv("data/Splits/train_split1.csv")["Action"]-1
model.fit(X,Y,epochs=10)

