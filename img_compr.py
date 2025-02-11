from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from kmeans import *

def main():
	#load the image
	img = plt.imread('./test.jpeg')
	print(f"Shape of original image {img.shape}")

	# reshape image to have the shape (X, 3). X being the number of pixel
	reshaped_img = img.reshape((-1, 3))
	print(f"Shape of reshaped image {reshaped_img.shape}")	

	# Use scikit kmenas: fit the KMeans --> The centroids are the colors
	lib_model = KMeans(n_clusters=16, random_state=42)
	lib_model.fit(reshaped_img)
	
	# Use my own written kemans
	my_model = Kmeans(n_clusters=16)
	my_model.fit(reshaped_img)

	# Transform the original image to have only the cluster colors
	transformed_image_lib = img.copy()
	transformed_image_my = img.copy()

	print("transforming image")
	for row in tqdm( range(img.shape[0])):
		for col in range(img.shape[1]):
			color_idx_lib = lib_model.predict([transformed_image_lib[row][col]])
			color_lib = lib_model.cluster_centers_[color_idx_lib]
			color_idx_my = my_model.predict([transformed_image_my[row][col]])
			color_my = my_model.best.centroids[color_idx_my]
			transformed_image_lib[row][col] = color_lib
			transformed_image_my[row][col] = color_my

	fig1, axes1 = plt.subplots(ncols=3)
	
	axes1[0].set_title("Original image")
	axes1[0].imshow(img)

	axes1[1].set_title("Transoformed image scikit")
	axes1[1].imshow(transformed_image_lib)

	axes1[2].set_title("Transoformed image my kmeans")
	axes1[2].imshow(transformed_image_my)
	plt.show()

	# How to reduce the amout of information when saving??
if __name__ == '__main__':
	main()
