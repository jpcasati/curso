import numpy as np

D = np.genfromtxt('dados.csv', delimiter=',')

print(np.shape(D))

X = D[1:, 1:4]

Y = D[1:, 4]

print(np.shape(X))
print(np.shape(Y))

np.save('entrada.npy', X)
np.save('saida.npy', Y)
