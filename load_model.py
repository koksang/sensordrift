import numpy as np
import pandas as pd
import os
import glob
import csv

from pandas import Series, DataFrame
from keras.models import load_model, Sequential
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# setting up current working directory
path = "C:/Users/KOKSANG/Desktop/fnn/dataset/Final"
os.chdir(path)

data_files = glob.glob("*.csv")
dataset = []
data = []
X = []
Y = []
scores = []

for f in data_files:
	data = pd.read_csv(f, sep="[;|,]", engine="python", header=None).astype(float)
	dataset.append(data)
	X.append(data.ix[:, 2:])
	Y.append(data.ix[:, 0])

def base_model():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=129, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

for k in range(0,10):
	pipeline = joblib.load('pipelinesfinal.pkl')
	pipeline.named_steps['mlp'].model = load_model(str(k) + '_final.h5')
	
	#predict with classication accuracy
	score = pipeline.score(X[k], Y[k]) 
		
	#predict with confusion matrix
	Y_pred = pipeline.predict(X[k])
	cm = confusion_matrix(Y[k], Y_pred)
		
	print(score)
	print(cm)
	
	
	
	




	
	
	
	
