import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split


file = r"D:\Users\Axel\Programmation\Python\NeuralNetworks\iris_dataset.xlsx"
df = pd.read_excel(file, engine='openpyxl')
df.iloc[:, :4] = df.iloc[:, :4].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

df.loc[df["species"] == "setosa"] = 0
df.loc[df["species"] == "versicolor"] = 1
df.loc[df["species"] == "virginica"] = 2

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
X_train = train.iloc[:, :4].values.tolist()
y_train = train.iloc[:, 4].tolist()
X_test = test.iloc[:, :4].values.tolist()
y_test = test.iloc[:, 4].tolist()

y_train_d3 = [[0, 0, 0] for i in range(len(y_train))]
for i in range(len(y_train)):
    y_train_d3[i][y_train[i]] = 1
y_test_d3 = [[0, 0, 0] for i in range(len(y_test))]
for i in range(len(y_test)):
    y_test_d3[i][y_test[i]] = 1

print(len(X_train))

# model = Sequential()
# model.add(Dense(1000, input_dim=4, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, y_train_d3, batch_size=20, epochs=10, verbose=1)
#
# prediction = model.predict(X_test)
# print(prediction)
# length = len(prediction)
# y_label = np.argmax(y_test_d3, axis=1)
# predict_label = np.argmax(prediction, axis=1)
#
# accuracy = np.sum(y_label == predict_label) / length * 100
# print("Accuracy of the dataset", accuracy)
