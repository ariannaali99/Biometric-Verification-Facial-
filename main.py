''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
import get_landmarks
import numpy as np

''' Import classifier '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
  
# KNN - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# NB - https://scikit-learn.org/stable/modules/naive_bayes.html#
# SVM - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# NN - https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron

''' Load the data and their labels '''
image_directory_netural = 'C:\\Users\\Arian\\OneDrive\\Documents\\Fall 2022\\Mobile Biometrics\\Final Project\\Temporary\\Neutral'
image_directory_non_netural = 'C:\\Users\\Arian\\OneDrive\\Documents\\Fall 2022\\Mobile Biometrics\\Final Project\\Temporary\\Non'
X_neutral, y_neutral = get_images.get_images(image_directory_netural)
X_non, y_non = get_images.get_images(image_directory_non_netural)

''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
X_neutral, y_neutral = get_landmarks.get_landmarks(X_neutral, y_neutral, 'landmarks/', 68, False)
X_non, y_non = get_landmarks.get_landmarks(X_non, y_non, 'landmarks/', 68, False)

''' Matching and Decision '''
# create an instance of the classifier
clf1 = SVC(kernel='poly', gamma=1, C=10, degree=2)
clf2 = SVC(kernel='linear', C=10)

num_correct = 0
labels_correct = []
num_incorrect = 0
labels_incorrect = []

clf1.fit(X_neutral, y_neutral)

for i in range(0, len(y_non)):

    query_img = X_non[i, :]
    query_label = y_non[i]
    y_pred = clf1.predict(query_img.reshape(1, -1))
    
    if y_pred == query_label:
        num_correct += 1
        labels_correct.append(query_label)
    else:
        num_incorrect += 1
        labels_incorrect.append(query_label)

print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect+1)))    

num_correct = 0
labels_correct = []
num_incorrect = 0
labels_incorrect = []

clf2.fit(X_neutral, y_neutral)

for i in range(0, len(y_non)):

    query_img = X_non[i, :]
    query_label = y_non[i]
    y_pred = clf2.predict(query_img.reshape(1, -1))
    
    # Gather results
    if y_pred == query_label:
        num_correct += 1
        labels_correct.append(query_label)
    else:
        num_incorrect += 1
        labels_incorrect.append(query_label)

# Print results
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect+1)))    