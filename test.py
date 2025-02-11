from kmeans import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
arr = np.random.rand(100, 3)

model = Kmeans(n_clusters = 3)

model.fit(arr)

colors = ['red', 'green', 'blue']

print (model.best.labels)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for label, elem in zip(model.best.labels, arr):
	ax.scatter(elem[0], elem[1], elem[2], color=colors[label])

for c in model.best.centroids:
	ax.scatter(c[0], c[1], c[2], marker='x', color='black')
plt.show()

print(model.predict([arr[3]]))



