#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:58:03 2024

@author: TejinderPannu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Step 1 - Let's generate Swiss roll using make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2)

#Step 2 - Plot the resulting generated Swiss roll dataset
def plot_swiss_roll(X, t):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.viridis)
    ax.set_title("Swiss Roll Dataset")
    plt.show()

#Plot the Swiss Roll
plot_swiss_roll(X, t)

#Step 3 - Uisng Kernel PCA with linear, RBF, and sigmoid kernels
#One method that would use different kernels
def apply_kpca(X, kernel, gamma=None):
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X)
    return X_kpca

#Step 4 - Method to plot the kPCA results for different kernels
def plot_kpca_results(X_kpca, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap=plt.cm.viridis)
    plt.title(title)
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.show()

#Linear Kernel
X_kpca_linear = apply_kpca(X, kernel='linear')
plot_kpca_results(X_kpca_linear, "Kernel PCA with Linear Kernel")

#RBF Kernel
X_kpca_rbf = apply_kpca(X, kernel='rbf', gamma=0.04)
plot_kpca_results(X_kpca_rbf, "Kernel PCA with RBF Kernel")

#Sigmoid Kernel
X_kpca_sigmoid = apply_kpca(X, kernel='sigmoid', gamma=0.01)
plot_kpca_results(X_kpca_sigmoid, "Kernel PCA with Sigmoid Kernel")

#Step 5 - Use kPCA and first I will use a kernel of my choice and then use GridSearchCV to find the best kernel
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, t > 9.5, test_size=0.2, random_state=42)

#Create a pipeline with StandardScaler, kPCA with an rbf kernel, and Logistic Regression
test_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kpca", KernelPCA(n_components=2, kernel="rbf", gamma=0.1)),  # Try rbf kernel with gamma=0.1
    ("log_reg", LogisticRegression())
])

#Fit the pipeline with the training data
test_pipeline.fit(X_train, y_train)

#Evaluate the performance
train_score = test_pipeline.score(X_train, y_train)
test_score = test_pipeline.score(X_test, y_test)

print(f"Training accuracy: {train_score}")
print(f"Testing accuracy: {test_score}")

#Create a pipeline with StandardScaler, kPCA, and Logistic Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

#Define the parameter grid for GridSearchCV
param_grid = [
    {
        "kpca__kernel": ["linear","rbf", "sigmoid"],
        "kpca__gamma": np.logspace(-4, 0, 10)
    }
]

#Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X_train, y_train)

#Best parameters found by GridSearchCV
print("Best Parameters found by GridSearchCV:", grid_search.best_params_)
#The best accuracy score
print("Best cross-validation accuracy:", grid_search.best_score_)
#Get the best model
best_model = grid_search.best_estimator_
#Calculate training accuracy
train_accuracy = best_model.score(X_train, y_train)
print(f"Training accuracy: {train_accuracy}")
#Calculate test accuracy
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

#Step 6 - Plotting results from GridSearchCV
best_model = grid_search.best_estimator_

X_train_kpca = best_model.named_steps['kpca'].transform(best_model.named_steps['scaler'].transform(X_train))

y_pred_train = best_model.named_steps['log_reg'].predict(X_train_kpca)

plt.scatter(X_train_kpca[:, 0], X_train_kpca[:, 1], c=y_pred_train, cmap=plt.cm.viridis, edgecolors='k', marker='o')
plt.title("Predictions on Training Set after GridSearchCV and KernelPCA")
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()













