
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd
import statsmodels.api as sm
import tensorflow
tensorflow.random.set_seed(1)
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow import keras
import time
from kerastuner.tuners import RandomSearch
import pickle
LOG_DIR=f"{int(time.time())}"
dataset = pd.read_csv('Rania Gamal - Design Builder Data.csv')
Z = dataset.iloc[:, 0:10]
X = dataset.iloc[:, 13:14]

y1 = np.array(X)

elv=dataset.iloc[:, 0:1]
mdt=dataset.iloc[:, 1:2]
wtmd=dataset.iloc[:, 2:3]
mdt= dataset.iloc[:, 3:4]
gr=dataset.iloc[:, 4:5]
lw=dataset.iloc[:, 5:6]
ag=dataset.iloc[:, 6:7]
wt=dataset.iloc[:, 7:8]
it=dataset.iloc[:, 8:9]
oln=dataset.iloc[:, 9:10]
ole=dataset.iloc[:, 10:11]
ols=dataset.iloc[:, 11:12]
olw=dataset.iloc[:, 12:13]


greenroof=gr.Green_roof.astype("category").cat.codes
greenroof=pd.Series(greenroof)

livingwall=lw.Living_wall.astype("category").cat.codes
livingwall=pd.Series(livingwall)

opn=oln.openings_N.astype("category").cat.codes
opn=pd.Series(opn)

ope=ole.openings_E.astype("category").cat.codes
ope=pd.Series(ope)

ops=ols.openings_S.astype("category").cat.codes
ops=pd.Series(ops)

opw=olw.openings_W.astype("category").cat.codes
opw=pd.Series(opw)



x1 = np.column_stack((elv,mdt,wtmd,mdt,gr,lw,ag,wt,it,oln,ole,ols,olw))
x1 = sm.tools.tools.add_constant(x1, prepend=True)

X_train, X_val, y_train, y_val = train_test_split(x1, y1)



y_train=np.reshape(y_train, (-1,1))
y_val=np.reshape(y_val, (-1,1))


xtrain_scale=X_train

xval_scale=X_val

ytrain_scale=y_train

yval_scale=y_val



def build_model(hp):
    model = keras.models.Sequential()
    model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(hp.Int("input_units",min_value=10,max_value=50,step=5), activation='relu'))

    model.add(Dense(1, activation='linear'))
    model.summary()
    
    model.compile(loss='mse', optimizer='adam', metrics=["mse"])
    return model


tuner=RandomSearch(
    build_model,
    objective="val_mse",
    max_trials=10,
    executions_per_trial=3,
    directory=LOG_DIR)
tuner.search(x=xtrain_scale,
             y=ytrain_scale,
             epochs=200,
             batch_size=1,
             validation_data=(xval_scale,yval_scale))


with open(f"tuner_{int(time.time())}.pkl","wb") as f:
     pickle.dump(tuner,f)

tuner=pickle.load(open("tuner_1604317993.pkl","rb"))
print(tuner.get_best_models()[0].summary)
print(tuner.results_summary())
object = pd.read_pickle('tuner_1604317993.pkl')
print(object)

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

