#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

#%%
# seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("./housing.csv", delim_whitespace=True, header=None)

#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 20)
print(df.info())    # 506개 dataframe, 14 col
print(df.head())
print(df.describe())

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
model = Sequential()
model.add(Dense(1, input_dim=13))  # dafault :  activation='linear'

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

#%%
# new instance where we do not know the answer
Xnew = np.array([[0.02631,0.00,7.090,0,0.4680,6.4220,78.80,4.9681,2,242.0,17.80,376.90,9.15],
                 [0.21114,12.40,7.860,0,0.5230,5.6320,100.00,6.0811,5,310.0,15.10,386.62,29.92]
                    ])  # 21.60  16.50
#Xnew = np.array([[0.02631,0.00,7.090,0,0.4680,4.575,78.80,4.9681,2,242.0,17.80,376.90,9.15],
#                 [0.21114,12.40,7.860,0,0.5230,7.001,100.00,6.0811,5,310.0,15.10,386.62,29.92]
#                    ])  # 21.60  16.50   X_5(방개수)만 변경)
#print(Xnew.shape)  # (2,13)

# make a prediction
ynew = model.predict(Xnew)
#ynew = model.predict_proba(Xnew)
#ynew = model.predict_classes(Xnew)  # 적절치 않음
# show the inputs and predicted outputs
print('-'*50)
#print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
print("X=%s, Predicted=%s" % (Xnew, ynew))
