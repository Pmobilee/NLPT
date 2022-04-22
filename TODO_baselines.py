# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from model.data_loader import DataLoader
from collections import Counter
import spacy
import textacy
from spacy import displacy
import pandas as pd
import numpy as np
from wordfreq import word_frequency
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
import random
import os
from scipy.stats import pearsonr
import sklearn
import json

cwd = os.getcwd()

print("#######################  TASK 10")
# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.

def majority_baseline(train_sentences, train_labels, testinput, testlabels):
    #find out about majority  class in training data
    predictions = []
    for instance in train_labels:   
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            predictions.append(i)
            
    majority_class = Counter(predictions).most_common()[0][0]
    #assign to each token (word) in test data the majority class as predicted value
    predictions = []
    for instance in testinput:
        tokens = instance.split(" ")
        for _ in tokens:
            predictions.append(majority_class)
    
    test_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            test_labels.append(i)

    #calculate accuracy
    correct_labeled = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct_labeled += 1

    accuracy = round(correct_labeled/len(predictions), 2)
    return accuracy, predictions




def random_baseline(train_sentences, train_labels, testinput, testlabels):
    predictions = []
    for instance in train_labels:   
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            predictions.append(i)
    
    distinct_labels = list(set(predictions))

    test_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            test_labels.append(i)

    #assign random label to test output 100 times and average accuracy
    accuracies = []
    for i in range (100):
        random.seed(i)
        predictions = []
        for instance in testinput:
            tokens = instance.split(" ")
            for _ in tokens:
                predictions.append(random.choice(distinct_labels))
        
        #calculate accuracy after each iteration with new random seed
        correct_labeled = 0
        for i in range(len(predictions)):
            if predictions[i] == test_labels[i]:
                correct_labeled += 1

        random_accuracy = round(correct_labeled/len(predictions), 2)
        accuracies.append(random_accuracy)
    
    accuracy = round(sum(accuracies)/len(accuracies), 2)
    return accuracy, predictions   



def length_baseline(testinput, testlabels, length_threshold):  
    test_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            test_labels.append(i)
    
    predictions = []
    for sentence in testinput:
        tokens = sentence.split(" ")
        for token in tokens:
            if len(token) >= length_threshold:
                predictions.append('C')
            else: 
                predictions.append('N')

    #calculate accuracy
    correct_labeled = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct_labeled += 1

    accuracy = round(correct_labeled/len(predictions), 2)
    return accuracy, predictions  



def frequency_baseline(testinput, testlabels, frequency_threshold):
    test_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            test_labels.append(i)

    predictions = []
    for sentence in testinput:
        tokens = sentence.split(" ")
        for token in tokens:
            if zipf_frequency(token, 'en') <= frequency_threshold:
                predictions.append('C')
            else: 
                predictions.append('N')

    #calculate accuracy
    correct_labeled = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct_labeled += 1

    accuracy = round(correct_labeled/len(predictions), 2)
    return accuracy, predictions  

if __name__ == '__main__':
    dir_path = cwd
    train_path = dir_path + "/data/preprocessed/train/"
    dev_path = dir_path + "/data/preprocessed/val/"          #changed from "/dev/" to "/val/" because there exists no "dev" folder
    test_path = dir_path + "/data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt") as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt") as dev_file:
        dev_sentences = dev_file.readlines()

    with open(dev_path + "labels.txt") as dev_label_file:       #changed from "train_path" to "dev_path"
        dev_labels = dev_label_file.readlines()

    with open(test_path + "sentences.txt") as testfile:
        testinput = testfile.readlines()

    with open(test_path + "labels.txt") as test_label_file:
        testlabels = test_label_file.readlines()
        
    length_threshold = 8   #accuracy best at threshold 8         
    frequency_threshold = 4.52  #accuracy best at zipf-frequency-threshold 4.52
    dev_majority_accuracy, dev_majority_predictions = majority_baseline(train_sentences, train_labels, dev_sentences, dev_labels)
    dev_random_accuracy, dev_random_predictions = random_baseline(train_sentences, train_labels, dev_sentences, dev_labels)
    dev_length_accuracy, dev_length_predictions = length_baseline(dev_sentences, dev_labels, length_threshold)
    dev_frequency_accuracy, dev_frequency_predictions = frequency_baseline(dev_sentences, dev_labels, frequency_threshold)

    test_majority_accuracy, test_majority_predictions = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    test_random_accuracy, test_random_predictions = random_baseline(train_sentences, train_labels, testinput, testlabels)
    test_length_accuracy, test_length_predictions = length_baseline(testinput, testlabels, length_threshold)
    test_frequency_accuracy, test_frequency_predictions = frequency_baseline(testinput, testlabels, frequency_threshold)
 

    baselines = ["Majority", "Random", "Length", "Frequency"]
    acc_dev = [dev_majority_accuracy, dev_random_accuracy, dev_length_accuracy, dev_frequency_accuracy]
    acc_test = [test_majority_accuracy, test_random_accuracy, test_length_accuracy, test_frequency_accuracy]
    baseline_df = pd.DataFrame({"Baseline":baselines, "Accuracy on dev": acc_dev, "Accuracy on test":acc_test})
    
    baseline_df.head()

    print(baseline_df)

    print("SEE THE NICE-LOOKING PANDAS VERSION IN ASSIGNMENT1.IPYNB")