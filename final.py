import numpy as np
import pandas as pd
import os
import glob
import math
import csv
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

from scipy.stats import boxcox

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# setting up current working directory
path = "C:/Users/KOKSANG/Desktop/fnn/dataset/Final/"
data = glob.glob(path + "*.csv")

df = pd.concat(pd.read_csv(i, sep="[;|,]", engine="python", header=None).astype(float) for i in data)

X	= df.values[:, 2:130]
Y	= df.values[:, 0]
bnum= df.values[:, 130]

X 	= X.reshape(16*(len(X)), 8)
col1= X[:, 0]
X	= X[:, 0:]

col1 = col1.reshape(len(col1), 1)

df_X 	= pd.DataFrame(X)
df_Y 	= pd.DataFrame(Y)
df_bnum	= pd.DataFrame(bnum)
df_col1	= pd.DataFrame(col1)

for col in df_X.columns:
	max = df_X[col].max()
	min = df_X[col].min()
	f_min = math.sqrt(min*min)
	
	dft = np.log(1 + df_X[col] - df_X[col].min())
	df_X[col] = dft
	
x_join = df_X.values.reshape(len(df_bnum), 128)
df_X_join = pd.DataFrame(x_join)
print(df_X_join.shape)

final_df = pd.concat([df_Y, df_bnum, df_X_join], axis = 1)

x_train	= final_df.values[0:5000, 1:]
x_test	= final_df.values[5000:, 1:]

y_train	= final_df.values[0:5000, 0]
y_test	= final_df.values[5000:, 0]

# define baseline model
def base_model():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=129, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate baseline model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('scaler', Normalizer()))
estimators.append(('mlp', KerasClassifier(build_fn=base_model, epochs=500, batch_size=100, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
	
pipeline.fit(x_train, y_train)

print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#predict with classication accuracy
score = pipeline.score(x_test, y_test) 
		
#predict with confusion matrix
Y_pred = pipeline.predict(x_test)
cm = confusion_matrix(y_test, Y_pred)
		
print(score)
print(cm)

'''
pipeline.named_steps['mlp'].model.save('batch1.h5')
pipeline.named_steps['mlp'].model = None
joblib.dump(pipeline, 'pipeline.pkl')
'''

