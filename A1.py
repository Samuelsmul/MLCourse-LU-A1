import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, ensemble, metrics, svm, model_selection, linear_model


def training_test_split(X, y, test_size=0.3, random_state=None):
    n_samples = len(X)
    n_test = int(test_size * n_samples)
    n_train = n_samples - n_test
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    X_train = X_shuffled[:n_train]
    X_test = X_shuffled[n_train:]
    y_train = y_shuffled[:n_train]
    y_test = y_shuffled[n_train:]
   
    #raise NotImplementedError('Your code here')
    return X_train, X_test, y_train, y_test


def true_positives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels == positive_class  
    pos_predicted = predicted_labels == positive_class 
    match = pos_true & pos_predicted  
    return np.sum(match)  
    #raise NotImplementedError('Your code here')

def false_positives(true_labels, predicted_labels, positive_class):
    pos_predicted = predicted_labels == positive_class  
    neg_true = true_labels != positive_class  
    match = pos_predicted & neg_true  
    return np.sum(match)  
    #raise NotImplementedError('Your code here')


def true_negatives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels != positive_class
    pos_predicted = predicted_labels != positive_class
    match = pos_true & pos_predicted
    return np.sum(match)
    #raise NotImplementedError('Your code here')


def false_negatives(true_labels, predicted_labels, positive_class):
    pos_predicted = predicted_labels != positive_class 
    neg_true = true_labels == positive_class
    match = pos_predicted & neg_true
    return np.sum(match)  
    #raise NotImplementedError('Your code here')


def precision(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FP)


def recall(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FN)
    #raise NotImplementedError('Your code here')


def accuracy(true_labels, predicted_labels, positive_class):
    #raise NotImplementedError('Your code here')
    TP = true_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    return (TP + TN) / (TP + TN + FP + FN)

def specificity(true_labels, predicted_labels, positive_class):
    #raise NotImplementedError('Your code here')
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    return TN / (TN + FP)


def balanced_accuracy(true_labels, predicted_labels, positive_class):
    #raise NotImplementedError('Your code here')
    TP = true_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return (recall + specificity)/2
    
    
def F1(true_labels, predicted_labels, positive_class):
    #raise NotImplementedError('Your code here')
    TP = true_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return 2 * ((precision * recall)/(precision + recall))
    


def load_data(fraction=0.75, seed=None, target_digit=9, appply_stratification=True):
    data = sklearn.datasets.load_digits()
    X = data.data
    y = data.target
    y[y != target_digit] = 11  # we have to do this swap because 1 and 0 also occur as labels in our dataset
    y[y == target_digit] = 12
    y[y == 11] = 0  # negative class
    y[y == 12] = 1  # positive class
    if appply_stratification:
        stratify = y
    else:
        stratify = None
    return sklearn.model_selection.train_test_split(X, y, train_size=fraction, random_state=seed, stratify=stratify)
