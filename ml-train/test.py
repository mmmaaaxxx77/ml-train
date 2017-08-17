import numpy as np
import random
from sklearn import manifold

x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

clf = manifold.LocallyLinearEmbedding(2, n_components=3,
                                      method='standard')
r = clf.fit(x)

print(r)

print(clf.transform([1, 0, 0, 0]))
print(clf.transform([1, 0, 0, 0]))


print([0]*10)

x_data = np.array([[1,2,3],[4,5,6]])
y_data = np.array([[7,8,9],[10,11,12]])
print(np.append(x_data, y_data, axis=1))

print(np.argmax(x_data, axis=1).tolist())

li = [a for a in range(20)]
random.shuffle(li)
print(li)


import random

a = ['a', 'b', 'c']
b = [1, 2, 3]

c = list(zip(a, b))

random.shuffle(c)

a, b = zip(*c)

print(a)
print(b)