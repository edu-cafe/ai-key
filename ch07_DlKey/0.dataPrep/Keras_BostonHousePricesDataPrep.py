#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("./housing.csv", delim_whitespace=True, header=None)

print(df.info())    # 506개 dataframe, 14 col
print(df.head())

"""
Features
1. Per capita crime rate.
2. Proportion of residential land zoned for lots over 25,000 square feet.
3. Proportion of non-retail business acres per town.
4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5. Nitric oxides concentration (parts per 10 million).
6. Average number of rooms per dwelling.
7. Proportion of owner-occupied units built prior to 1940.
8. Weighted distances to five Boston employment centres.
9. Index of accessibility to radial highways.
10. Full-value property-tax rate per $10,000.
11. Pupil-teacher ratio by town.
12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
13. % lower status of the population.

타깃은 주택의 중간 가격으로 천달러 단위입니다
"""

#%%
dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

#%%
# 데이터 전처리
np.set_printoptions(linewidth=1000, suppress=True, precision=2) # supress:fixed point notation

print(X_train[0])
mean = X_train.mean(axis=0)
X_train -= mean
print(X_train[0])
std = X_train.std(axis=0)
X_train /= std
print(X_train[0])

X_test -= mean
X_test /= std

#%%
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))    # dafault :  activation='linear'

model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['mae'])

num_epochs = 200
all_mae_histories = []
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=32)

mae_history = history.history['mean_absolute_error']
all_mae_histories.append(mae_history)

test_mse_score, test_mae_score = model.evaluate(X_test, Y_test)
print('-'*50)
print(test_mse_score, test_mae_score)

#%%
import numpy as np
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(average_mae_history)

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Test MAE')
plt.show()


