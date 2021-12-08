import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("data_139.csv")
df = df.fillna(df.mean())

df.iloc[:, 1:-2] = df.iloc[:, 1:-2].apply(np.sign)
#df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
X_train = train.iloc[:, 1:-2].apply(np.sign).values.tolist()
y_train = train.iloc[:, -1].apply(np.sign).astype(int).tolist()
X_test = test.iloc[:, 1:-2].apply(np.sign).values.tolist()
y_test = test.iloc[:, -1].apply(np.sign).astype(int).tolist()

for v in range(len(y_train)):
    y_train[v] = min(1, y_train[v]+1)
for v in range(len(y_test)):
    y_test[v] = min(1, y_test[v] + 1)

y_train_d2 = [[0, 0] for i in range(len(y_train))]
for i in range(len(y_train)):
    y_train_d2[i][y_train[i]] = 1
y_test_d2 = [[0, 0] for i in range(len(y_test))]
for i in range(len(y_test)):
    y_test_d2[i][y_test[i]] = 1

model = Sequential()
# model.add(Dense(1000, input_dim=100, activation='relu'))
# model.add(Dense(800, activation='relu'))
# model.add(Dense(700, activation='relu'))
model.add(Dense(200, input_dim=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_d2, batch_size=20, epochs=10, verbose=1)

prediction = model.predict(X_test)
# print(prediction)
length = len(prediction)
y_label = np.argmax(y_test_d2, axis=1)
predict_label = np.argmax(prediction, axis=1)

# print(y_label)
# print(predict_label)

accuracy = np.sum(y_label == predict_label) / length * 100
print("Accuracy of the dataset", accuracy)
