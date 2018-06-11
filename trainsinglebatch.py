import numpy as np
import pandas as pd
import os
import glob

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# setting up current working directory
path = "C:/Users/KOKSANG/Desktop/fnn/dataset/Setting 1/test"
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
	model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu'))
	model.add(Dense(64, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
#concatenate X and Y into df
'''df_X = (pd.concat((X[i] for i in range (0,2)), axis=0)).values
df_Y = (pd.concat((Y[i] for i in range (0,2)), axis=0)).values '''

# evaluate baseline model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=base_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X[0], Y[0], cv=kfold)
	
pipeline.fit(X[0], Y[0])

pipeline.named_steps['mlp'].model.save('batch1.h5')
pipeline.named_steps['mlp'].model = None
joblib.dump(pipeline, 'pipeline.pkl')
	
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


