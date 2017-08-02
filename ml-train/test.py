from sklearn import manifold

x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

clf = manifold.LocallyLinearEmbedding(2, n_components=3,
                                      method='standard')
r = clf.fit(x)

print(r)

print(clf.transform([1, 0, 0, 0]))
print(clf.transform([1, 0, 0, 0]))
