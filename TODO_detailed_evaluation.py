#-> LSTM was evaluated on test data, so we calculate precision, recall and F1 based on the results on the test data

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
from TODO_baselines import random_baseline, majority_baseline, length_baseline, frequency_baseline

cwd = os.getcwd()


def get_lstm_predictions(lstm_output):
    '''
    First column is gold label, Second column is prediction
    '''
    predictions = []
    for instance in lstm_output:
        tokens = instance.split("\t")
        if len(tokens) != 3: continue   #ignore separating rows   
        if tokens[2] == 'N\n':
            predictions.append('N')
        elif tokens[2] == 'C\n':
            predictions.append('C')
    return predictions

if __name__ == '__main__':
    print("#######################  TASK 12")


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
    with open(cwd + "/data/preprocessed/test/labels.txt") as test_label_file:
        testlabels = test_label_file.readlines()

    with open(cwd + "/experiments/base_model/model_output.tsv") as model_output:
        lstm_output = model_output.readlines()

    test_labels = []
    for instance in testlabels:
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            test_labels.append(i)
    
    num_N_labels  = Counter(test_labels).most_common()[0][1]
    num_C_labels = Counter(test_labels).most_common()[1][1]
    print(num_N_labels)
    print(num_C_labels)

    _, test_majority_predictions = majority_baseline(train_sentences, train_labels, testinput, testlabels)
    _, test_random_predictions = random_baseline(train_sentences, train_labels, testinput, testlabels)
    _, test_length_predictions = length_baseline(testinput, testlabels, length_threshold)
    _, test_frequency_predictions = frequency_baseline(testinput, testlabels, frequency_threshold)
    lstm_prediction = get_lstm_predictions(lstm_output)


    # WHEN CLASS 'N' IS TRUE POSITIV
    # what is fp, fn, tp, tn depends on order of labels-argument in confusion matrix function, first element in labels-list is TP
    print(sklearn.metrics.confusion_matrix(test_labels, test_majority_predictions, labels = ['C', 'N']))
    # print(sklearn.metrics.confusion_matrix(test_labels, test_random_predictions, labels = ['N', 'C']))
    # print(sklearn.metrics.confusion_matrix(test_labels, test_length_predictions, labels = ['N', 'C']))
    # print(sklearn.metrics.confusion_matrix(test_labels, test_frequency_predictions, labels = ['N', 'C']))
    # print(sklearn.metrics.confusion_matrix(test_labels, lstm_prediction, labels = ['N', 'C']))

    # pos_label determines TP
    print("Scores with 'N' as TP:")
    print("Precision score majority baseline: " + str(sklearn.metrics.precision_score(test_labels, test_majority_predictions, pos_label='N')))
    print("Precision score random baseline: " + str(sklearn.metrics.precision_score(test_labels, test_random_predictions, pos_label='N')))
    print("Precision score length baseline: " + str(sklearn.metrics.precision_score(test_labels, test_length_predictions, pos_label='N')))
    print("Precision score frequency baseline: " + str(sklearn.metrics.precision_score(test_labels, test_frequency_predictions, pos_label='N')))
    print("Precision score LSTM: " + str(sklearn.metrics.precision_score(test_labels, lstm_prediction, pos_label='N')))
    print()
    print("Recall score majority baseline: " + str(sklearn.metrics.recall_score(test_labels, test_majority_predictions, pos_label='N')))
    print("Recall score random baseline: " + str(sklearn.metrics.recall_score(test_labels, test_random_predictions, pos_label='N')))
    print("Recall score length baseline: " + str(sklearn.metrics.recall_score(test_labels, test_length_predictions, pos_label='N')))
    print("Recall score frequency baseline: " + str(sklearn.metrics.recall_score(test_labels, test_frequency_predictions, pos_label='N')))
    print("Recall score LSTM: " + str(sklearn.metrics.recall_score(test_labels, lstm_prediction, pos_label='N')))
    print()
    print("F1 score majority baseline: " + str(sklearn.metrics.f1_score(test_labels, test_majority_predictions, pos_label='N')))
    print("F1 score random baseline: " + str(sklearn.metrics.f1_score(test_labels, test_random_predictions, pos_label='N')))
    print("F1 score length baseline: " + str(sklearn.metrics.f1_score(test_labels, test_length_predictions, pos_label='N')))
    print("F1 score frequency baseline: " + str(sklearn.metrics.f1_score(test_labels, test_frequency_predictions, pos_label='N')))
    print("F1 score LSTM: " + str(sklearn.metrics.f1_score(test_labels, lstm_prediction, pos_label='N')))

    # WHEN CLASS 'C' IS TRUE POSITIV
    print("\nScores with 'C' as TP:")
    print("Precision score majority baseline: " + str(sklearn.metrics.precision_score(test_labels, test_majority_predictions, pos_label='C')))
    print("Precision score random baseline: " + str(sklearn.metrics.precision_score(test_labels, test_random_predictions, pos_label='C')))
    print("Precision score length baseline: " + str(sklearn.metrics.precision_score(test_labels, test_length_predictions, pos_label='C')))
    print("Precision score frequency baseline: " + str(sklearn.metrics.precision_score(test_labels, test_frequency_predictions, pos_label='C')))
    print("Precision score LSTM: " + str(sklearn.metrics.precision_score(test_labels, lstm_prediction, pos_label='C')))
    print()
    print("Recall score majority baseline: " + str(sklearn.metrics.recall_score(test_labels, test_majority_predictions, pos_label='C')))
    print("Recall score random baseline: " + str(sklearn.metrics.recall_score(test_labels, test_random_predictions, pos_label='C')))
    print("Recall score length baseline: " + str(sklearn.metrics.recall_score(test_labels, test_length_predictions, pos_label='C')))
    print("Recall score frequency baseline: " + str(sklearn.metrics.recall_score(test_labels, test_frequency_predictions, pos_label='C')))
    print("Recall score LSTM: " + str(sklearn.metrics.recall_score(test_labels, lstm_prediction, pos_label='C')))
    print()
    print("F1 score majority baseline: " + str(sklearn.metrics.f1_score(test_labels, test_majority_predictions, pos_label='C')))
    print("F1 score random baseline: " + str(sklearn.metrics.f1_score(test_labels, test_random_predictions, pos_label='C')))
    print("F1 score length baseline: " + str(sklearn.metrics.f1_score(test_labels, test_length_predictions, pos_label='C')))
    print("F1 score frequency baseline: " + str(sklearn.metrics.f1_score(test_labels, test_frequency_predictions, pos_label='C')))
    print("F1 score LSTM: " + str(sklearn.metrics.f1_score(test_labels, lstm_prediction, pos_label='C')))

    # WEIGHTEST F1 SCORES
    print("\nWeighted F1-score majority baseline: "+ str(sklearn.metrics.f1_score(test_labels, test_majority_predictions, average='weighted')))
    print("Weighted F1-score random baseline: "+ str(sklearn.metrics.f1_score(test_labels, test_random_predictions, average='weighted')))
    print("Weighted F1-score length baseline: "+ str(sklearn.metrics.f1_score(test_labels, test_length_predictions, average='weighted')))
    print("Weighted F1-score frequency baseline: "+ str(sklearn.metrics.f1_score(test_labels, test_frequency_predictions, average='weighted')))
    print("Weighted F1-score LSTM: "+ str(sklearn.metrics.f1_score(test_labels, lstm_prediction, average='weighted')))

    
    # print(sklearn.metrics.recall_score(test_labels, lstm_prediction, pos_label='C'))

    # print(sklearn.metrics.f1_score(test_labels, lstm_prediction, pos_label='C'))

    print("#######################  TASK 14")
    with open(cwd + "/experiments/base_model/model_output.tsv") as model_output:
        lstm_output = model_output.readlines()
    
    f = open(cwd + "/experiments/base_model/params.json")
    params = json.load(f)

    lstm_prediction = get_lstm_predictions(lstm_output)

    weighted_F1_score_LSTM = sklearn.metrics.f1_score(test_labels, lstm_prediction, average='weighted')

    #created a new file 'Weighted_F1_scores.txt' to save F1 scores and predictions for differnet hyperparamter values
    with open(cwd + '/Weighted_F1_scores.txt', 'a') as f:
        f.write(str(params['learning_rate']) + " ")
        f.write(str(weighted_F1_score_LSTM) + " ")
        for i in lstm_prediction:
            f.write(str(i) + " ")
        f.write("\n")
    learning_rates = []
    weighted_f1_scores = []
    lstm_predictions = []

    with open(cwd + '/Weighted_F1_scores.txt', 'r') as f:
        LSTM_metrics_output = f.readlines()

    for instance in LSTM_metrics_output:
        output = instance.split(" ")
        learning_rates.append(float(output[0]))
        weighted_f1_scores.append(round(float(output[1]), 3))
        single_model_predictions = []
        for i in range(2, len(output)):
            if output[i] == '\n': continue
            else: single_model_predictions.append(output[i])
        lstm_predictions.append(single_model_predictions)

    print(learning_rates)
    print(weighted_f1_scores)
    print(lstm_predictions)
    plt.plot(learning_rates, weighted_f1_scores, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Weighted F1 Score")
    plt.xticks(rotation=45)
    plt.show()
    print("Plot may not show outside of jupyter notebook")

    with open(cwd + "/data/preprocessed/test/sentences.txt") as test_sentences:
        test_sentences = test_sentences.readlines()

    #get all tokens from test dataset
    test_tokens = []
    for instance in test_sentences:
        tokens = instance.split(" ")
        for i in tokens:         
            if i == 'N\n':
                i = 'N'
            elif i == 'C\n':
                i = 'C'
            test_tokens.append(i)

    #find prediction changes for hyperparameter changes
    for i in range(len(lstm_predictions)-1):
        for j in range(len(lstm_predictions[i])):
            if lstm_predictions[i][j] != lstm_predictions[i+1][j]:
                print("Hyperparam value " + str(i) + " vs Hyperparam value " + str(i+1) + "-> for Token: " + str(test_tokens[j]) + " , the prediction has changed from " + str(lstm_predictions[i][j] + " to " + str(lstm_predictions[i+1][j])))
