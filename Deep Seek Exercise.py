# import tensorflow as tf
# import numpy as np

# x = np.array([[0,0],[0,2],[1,0],[1,1]])
# y = np.array([[0],[1],[1],[0]])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(2,activation = 'relu'),
#     tf.keras.layers.Dense(1,activation = 'sigmoid')
# ])

# model.compile(optimizer = 'adam',lass = 'binary_crossnetropy')

# model.fit(x,y, epochs = 1000, verbos = 0)

# prediction = model.predict(x)
# print("Forcast:",prediction.flatten())

import tensorflow as ft
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

x,y = data.data, data.target

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)

model = ft.keras.Sequential([
    ft.keras.layers.Dense(30, activation = 'relu'),
    ft.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 50,verbose = 0)


loss , accuracy = model.evaluate(x_test,y_test,verbose = 0)
print(f'Accuracy:{accuracy:.2%}')