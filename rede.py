import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

X = np.load('entrada.npy')
Y = np.load('saida.npy')

Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.30,
                                                    random_state=1000)

model = Sequential()

model.add(Dense(10, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.summary()

metrics = model.fit(X_train, Y_train, epochs=30, batch_size=128, verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=1)

print(scores)

res = model.predict_classes(X_test, verbose=1)

Y_cfm = np.argmax(Y_test, axis=1)
print('\nConfusion Matrix')
print(confusion_matrix(Y_cfm, res))
print('\nNormalized Confusion Matrix')
print(confusion_matrix(Y_cfm, res, normalize='true'))
print('\nClassification Report')
target_names = ['Caminhada', 'Corrida']
print(classification_report(Y_cfm, res, target_names=target_names))