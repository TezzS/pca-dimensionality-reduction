#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:01:42 2024

@author: TejinderPannu
"""

import pandas as pd
import numpy as np
import arff
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

#This is the path were I saved the MNIST Dataset (arff file)
mnist_arff_path = '/Users/user/Documents/Centennial college/SEM6/Unsupervised Learning/Assignments/TejinderPannu_Assignment1/mnist_784.arff'

#Step 1 - Let's load the dataset
with open(mnist_arff_path, 'r') as file:
    dataset = arff.load(file)

#Converting to dataframe
mnist_df = pd.DataFrame(dataset['data'])

#Separate features and labels
X = mnist_df.iloc[:, :-1]  #All columns except the last one (features)
y = mnist_df.iloc[:, -1]   #The last column (labels)

y.value_counts()

#Step 2 - Display each unique digit at least once
#For this step I prefer to display each unique digit at least once instead of displaying random 10
def display_unique_digits(X, y):
    unique_digits = np.unique(y)  #I fetch all unique digits
    plt.figure(figsize=(10, 2))   #10 figsize because 10 digits

    for idx, digit in enumerate(unique_digits):
        #I will be using the first occurrence of each unique digit
        digit_index = np.where(y == digit)[0][0]
        plt.subplot(1, len(unique_digits), idx + 1)
        plt.imshow(X.iloc[digit_index].values.reshape(28, 28), cmap="binary")
        plt.axis("off")
        plt.title(int(digit))

    plt.show()

#Display using the above function
display_unique_digits(X, y)

#Step 3 -  PCA to retrieve the 1st and 2nd principal components and output their explained variance ratio
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
exp_variance_ratio = pca.explained_variance_ratio_

print(f"Explained variance ratio of the first two principal components: {exp_variance_ratio}")

#Step 4 - Plotting the projections of the 1st and 2nd principal components onto a 2D hyperplane
def plot_pca_projections(X_pca, y):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap="jet", alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("Projection of MNIST onto 2D Hyperplane")
    plt.show()

#Use the method to plot whilst passing X_pca and target (y)
plot_pca_projections(X_pca, y)

#Step 5 - Using Incremental PCA, we reduce the dimensionality down to 154 dimensions
ipca = IncrementalPCA(n_components=154, batch_size=200)
X_ipca = ipca.fit_transform(X)
print(f"Shape after dimensionality reduction: {X_ipca.shape}")

#Step 6 - Display the original and compressed digits
def display_compressed_digits(X_original, X_compressed, ipca_model, num_samples=10):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        # Original digit
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X_original.iloc[i].values.reshape(28, 28), cmap="binary")
        plt.axis("off")
        
        # Compressed and reconstructed digit
        compressed_digit = ipca_model.inverse_transform(X_compressed[i].reshape(1, -1))
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(compressed_digit.reshape(28, 28), cmap="binary")
        plt.axis("off")
        
    plt.show()

#Display original and compressed digits
display_compressed_digits(X, X_ipca, ipca, num_samples=10)
