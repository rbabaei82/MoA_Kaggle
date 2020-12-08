import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors


np.random.seed(124)
# target dataframe
df = pd.read_csv("D:/DataScience/R/Profile/Kaggle_MoA/train_targets_scored.csv")
# rename index by sig_id
ids = list(df.iloc[:,0])
df.index = ids
df = df.drop(["sig_id"], axis=1)

## check for imbalance label
max(df.sum())
## 832 out of 23814  highly imbalanced


# rmove samples with no target
Y = df.drop(df.sum(axis=1)[df.sum(axis=1) <1].index.values, axis=0)


## get index names to filter Features

idNames = list(Y.index.values)



# features
Features = pd.read_csv("D:/DataScience/R/Profile/Kaggle_MoA/train_features.csv")

# rename index by sig_id
ids = list(Features.iloc[:,0])
Features.index = ids


# remove non-informative features
Features = Features.drop(["sig_id","cp_type"], axis = 1)

# convert categoricals to binary
Features = pd.get_dummies(Features, columns=["cp_time", "cp_dose"])

## select samples based on filtered targets dataframe

X = Features.loc[idNames,:]


## convert dataframe to array

arrX = np.array(X)
arrY = np.array(Y)


## split for training by cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
for train_ix, test_ix in cv.split(arrX):
    X_train, X_test = arrX[train_ix], arrX[test_ix]
    Y_train, Y_test = arrY[train_ix], arrY[test_ix]


### define the first model
model = Sequential()
model.add(Dense(1000, input_dim = 877, kernel_initializer= "he_uniform", activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(206, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

hist = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test), batch_size=32)

## loss: 0.0035 - accuracy: 0.9989 - val_loss: 0.0378 - val_accuracy: 0.9952

plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model loss")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()



## visualize accuracy
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()




### evaluate on the whole set of data
yhat = model.predict(arrX)
yhat = yhat.round()

acc = accuracy_score(arrY, yhat)

print('>%.3f' % acc)
## 0.875




############ handle imbalanced labels with MLSMOTE
def get_tail_label(df):
    """
    get tail label columns of the target dataframe

    Parameters
    ----------
    df : pandas.DataFrame
       

    Returns
    -------
    tail_label: a list of column name of all tail label, minority

    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl)/irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label


def get_index(df):
    """
    get the index of all tail_label rows

    Parameters
    ----------
    df : pandas.DataFrame target df
        

    Returns
    -------
    index: a list of index number of all the tail labels

    """
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label]==1].index)
        index = index.union(sub_index)
    return list(index)

def get_minority_instance(X, y):
    """
    get minority data frame containing all the tail labels

    Parameters
    ----------
    X : pandas.DataFrame
        Features dataframe.
    y : pandas.DataFrame
        target dataframe.

    Returns
    -------
    X_sub: pandas.DataFrame, features of minority.
    y_sub: pandas.DataFrame, target of minority

    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub

def nearest_neighbour(X):
    """
    get index of 5 nearest neighbours of all instances

    Parameters
    ----------
    X : np.array
        

    Returns
    -------
    indices: list of list of index of 5 NN of each element in X

    """
    
    nbs = NearestNeighbors(n_neighbors=5, metric="euclidean",
                           algorithm="kd_tree").fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X,y, n_sample):
    """
    get augmented data using MLSMOTE algorithm

    Parameters
    ----------
    X : pandas.DataFrame
        features dataframe.
    y : pandas.DataFrame
        target dataframe
    n_sample : number of samples to be augmented

    Returns
    -------
    new_X: pandas.DataFrame, aumented features
    target: pandas.DataFrame, augmented target

    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val>2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbour,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio*gap)
    new_X = pd.DataFrame(new_X, columns = X.columns)
    target = pd.DataFrame(target, columns = y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target


#### Getting minority instances

X_sub, y_sub = get_minority_instance(X, Y)

# simulate data from minority samples
X_res, y_res = MLSMOTE(X_sub, y_sub, 50000)

### concatante simulated data and the original data

train_X = pd.concat([X, X_res], axis=0)
train_Y = pd.concat([Y, y_res], axis=0)


## convert to np.array
train_X = np.array(train_X)
train_Y = np.array(train_Y)



# split data for trianing and test

for train_ix, test_ix in cv.split(train_X):
    X_train2, X_test2 = train_X[train_ix], train_X[test_ix]
    Y_train2, Y_test2 = train_Y[train_ix], train_Y[test_ix]




###############
model2 = Sequential()
model2.add(Dense(1000, input_dim = 877, kernel_initializer= "he_uniform", activation="relu"))
model2.add(Dropout(0.2))
model2.add(Dense(512, activation="relu"))
model2.add(Dropout(0.1))
model2.add(Dense(206, activation="sigmoid"))

model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

hist2 = model2.fit(X_train2, Y_train2, epochs=10,
                   validation_data=(X_test2, Y_test2), batch_size=32)

## loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0083 - val_accuracy: 0.9988

## visualize loss
plt.plot(hist2.history["loss"])
plt.plot(hist2.history["val_loss"])
plt.title("Model2 loss")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()



## visualize accuracy
plt.plot(hist2.history["accuracy"])
plt.plot(hist2.history["val_accuracy"])
plt.title("Model2 Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "val"], loc="upper left")
plt.show()



### evaluate on the whole set of data
yhat2 = model2.predict(train_X)
yhat2 = yhat2.round()

acc2 = accuracy_score(train_Y, yhat2)

print('>%.3f' % acc2)
## 0.930

