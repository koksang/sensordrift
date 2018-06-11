import numpy as np
import pandas as pd
import os
import glob

import matplotlib.pyplot as plt

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

for f in data_files:
	data = pd.read_csv(f, sep="[;|,]", engine="python", header=None).astype(float)
	dataset.append(data)
	X.append(data.ix[:, 2:])
	Y.append(data.ix[:, 0])
	
	
# define baseline model
def base_model():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=129, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
for k in range(0,5):
	df_X = pd.concat((X[i] for i in range (0,k+1)), axis=0).values
	df_Y = pd.concat((Y[i] for i in range (0,k+1)), axis=0).values
	
	# evaluate baseline model with standardized dataset
	np.random.seed(seed)
	estimators = []
	estimators.append(('scaler', Normalizer()))
	estimators.append(('mlp', KerasClassifier(build_fn=base_model, epochs=500, batch_size=100, verbose=0)))
	pipeline = Pipeline(estimators)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(pipeline, df_X, df_Y, cv=kfold)
	
	print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	
	
	pipeline.fit(df_X, df_Y)
	pipeline.named_steps['mlp'].model.save(str(k) + '_final1.h5')
	pipeline.named_steps['mlp'].model = None
	joblib.dump(pipeline, 'pipelinefinal1.pkl')
	
	
	
	
	

