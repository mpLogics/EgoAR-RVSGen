import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D,Activation
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


df = pd.read_csv("data/Splits/train_split1.csv")
gt_action = df["Action"]

model = Sequential()
model.add(Input(shape=(50)))

#model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(106))
model.add(Activation('softmax'))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

X = np.load("Actions.npz",allow_pickle=True)['a']
Y = pd.read_csv("data/Splits/train_split1.csv")["Action"]-1

model.fit(X,Y[:len(X)],epochs=200,validation_split=0.2)

"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X = np.load("Actions.npz",allow_pickle=True)['a']

X_flattened=[]
for i in range(X.shape[0]):
	X_flattened.append(X[i].flatten())
Y = pd.read_csv("data/Splits/train_split1.csv")["Action"]-1
# Always scale the input. The most convenient way is to use a pipeline.
#print(X_flattened)
clf = make_pipeline(StandardScaler(),
	SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X_flattened, Y[:len(X)])

for i in range(8250):
	print(clf.predict(X_flattened[i].reshape(1,25)),",",Y[i])
#Pipeline(steps=[('standardscaler', StandardScaler()),
#                ('sgdclassifier', SGDClassifier())])
#print(clf.predict([[-0.8, -1]]))
"""