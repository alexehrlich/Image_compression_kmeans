import random as r
import numpy as np
import math
from tqdm import tqdm

class Model:
	def __init__(self):
		self.centroids = None
		self.labels = None

class Kmeans:
	def __init__(self, n_clusters, random_inits=5, max_iter=10):
		self.n_clusters = n_clusters
		self.training_data_shape = None
		self.best = None
		self.random_inits = random_inits
		self.max_iter = max_iter
		self.model_cost = {}

	# slow approach with for loops
	# def cluster_data(self, X, centroids):
	# 	labels  = np.zeros(X.shape[0], dtype=int)
	# 	for j, x in enumerate(X):
	# 			distances = np.zeros(self.n_clusters)
	# 			for i, centroid in enumerate(centroids):
	# 				distances[i] = self._get_distance(x, centroid)
	# 				labels[j] = np.argmin(distances)
	# 	return labels
	
	def _get_distance(self, x, centroid):
		'''
			Use L2 norm (eucledian) distance from numpy
		'''
		return np.linalg.norm(x - centroid)

	def cluster_data(self, X, centroids):
		'''
			Use numpys broadcasting to calculate the distnaces with
			with vectorization for better performance.
		'''
		distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
		labels = np.argmin(distances, axis=1)
		return labels

	def choose_best(self):
		'''
			Iterates over the models (keys) and finds the one
			with the lowest cost.
		'''
		self.best = min(self.model_cost.keys(), key=lambda x : self.model_cost[x])

	def run_kmeans(self, seed, X):
		'''
			Randomly inits n_clusters centroids, calculates the distance from every sample to every
			centroid and choose the centroid fro each sample with the smallest distnace. (labeling)
			The recalculate the centroid as average of all assigned samples.
			Do this until max_iter is reached.
		'''
		np.random.seed(seed)
		model = Model()
		# pick n_cluster random training samples and set them as the current centroids
		model.centroids = np.array([X[idx] for idx in np.random.choice(X.shape[0], self.n_clusters)], dtype=np.float64)
		for i in range (self.max_iter):
			# Calculate distance from every training sample to the current centroids and group set the index of the closest in the distances
			model.labels = self.cluster_data(X, model.centroids)
			# Calculate the average of each group and set the centroid of each group to that value
			model.centroids = np.zeros_like(model.centroids)
			centroid_count = np.zeros(model.centroids.shape[0])
			
			#Update the centroids
			for k in range(self.n_clusters):
				cluster_points = X[model.labels == k] #advanced indexing
				if len(cluster_points > 0):
					model.centroids[k] = np.mean(cluster_points, axis=0
					)
		return model, self.cost(X, model)

	def fit(self, X):
		'''
			Find the best centroids by running k means random_inits times
			with all samples becuase the result depends on the initially
			randomly chosen centroids.
		'''
		self.training_data_shape = X.shape
		self.labels  = np.zeros(X.shape[0], dtype=int)

		print(f"Clustering data into {self.n_clusters} clusters ...")
		for seed in tqdm(range(self.random_inits)):
			model, cost = self.run_kmeans(seed, X)
			self.model_cost[model] = cost
		self.choose_best()

	def predict(self, x):
		x = np.array(x)
		if x.shape[1:] != self.training_data_shape[1:]:
			print("Wrong shape. Must be {self.training_data_shape}")
			return
		return self.cluster_data(x, self.best.centroids)

	def cost(self, X, model):
		'''
			Use np advanved indexing to select all centroids for all samples
			to have the same shape as X for subtraction.
		'''
		distances = np.linalg.norm(X - model.centroids[model.labels], axis=1)
		return np.mean(distances)