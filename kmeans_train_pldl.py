#python kmeans_train.py --train_file data/jobQ123_BOTH/processed/jobQ1_BOTH/split/jobQ1_BOTH_train.json --dev_file data/jobQ123_BOTH/processed/jobQ1_BOTH/split/jobQ1_BOTH_dev.json --lower 2 --upper 12 --iterations 5 --output_file jobQ1_BOTH_split_kmeans  --folder_name data/jobQ1_BOTH/kmeans

#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
 	mpl.use('Agg')
from sklearn.cluster import KMeans
from tqdm import tqdm
import os, math, sys, json, collections
from scipy.stats import entropy
import numpy as np
from sklearn.externals import joblib
from helper_functions_km import write_model_logs_to_json,read_labeled_data_KMeans,save_trained_model_joblib,save_trained_model_joblib_sklearn,create_folder
import argparse
import sys
import pdb
import pandas as pd
from model_eval_transformers import model_selection_for_clustering
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel,LdaMulticore
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer('english')
model_selection_measure = "cross"
from statistics import mean
from sklearn.metrics.scorer import make_scorer

from scipy.stats import entropy #used for KL since it calculates Kl when the parameters are probability distributions
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processe


#Change this function for the file paths if its different.
def load_default_parameters():
    train_file = "data/facebook/processed/fb_train.json"
    dev_file = "data/facebook/processed/fb_dev.json"
    test_file = "data/facebook/processed/fb_test.json"
    lower = 4
    upper = 40
    output_file = "facebook_split_kmeans"
    folder_name = "data/facebook/processed/fb/kmeans_predict"
    return train_file,dev_file,test_file,lower,upper,output_file,folder_name


def lemmatize_stemming(text):
    # return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess_stem_clean(text):
    result = []
    text = remove_url(text)
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result

def remove_url(text):
    text = re.sub(r"http\S+", "", text) #remove URLs from the text
    return text

def clean_text_for_sklean(dataset):
    documents = []
    indexs = []
    cleaned_documents = []
    answer_tokens_list = []
    for message_id in dataset:
        message = dataset[int(message_id)]
        documents.append(message)
        indexs.append(message_id)
        answer_tokens = preprocess_stem_clean(message)
        answer_tokens_list.append(answer_tokens)
        answer_str = ' '.join(answer_tokens)
        cleaned_documents.append(answer_str)
    return documents,indexs,cleaned_documents,answer_tokens_list

def generate_only_labels(dataset):
    labels = []
    for data in dataset:
        label = dataset[str(data)]
        labels.append(label)
    return labels

def embed_to_vect(answer_counters, choices):
    labels = []
    for item in zip(answer_counters, range(choices)):
        labels.append(str(item[0]))
    return labels#''.join(labels).split()

def train_kmeans_labels(train_answer_counters,dev_answer_counters, LOWER, UPPER, output_name, folder_name,label_dict,train_message_dict,dev_message_dict):
    train_messages,train_message_ids,train_cleaned_messages,train_tokens = clean_text_for_sklean(train_message_dict)
    train_vectors = generate_only_labels(train_answer_counters)

    dev_messages,dev_message_ids,dev_cleaned_messages,dev_tokens = clean_text_for_sklean(dev_message_dict)

    dev_labels = generate_only_labels(dev_answer_counters)

    dev_vectors = dev_labels
    results_log_dict = {}
    results_log = []
    # kl_score_function = make_scorer(evaluate_model_kl)
    for n_clusters in tqdm(range(LOWER, UPPER)):

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility
        clusterer = KMeans(n_clusters=n_clusters) #Default 300 iterations

        # for index in tqdm(range(ITERATIONS)):

        # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        train_predict = clusterer.fit_predict(train_vectors)
        dev_predict_vects = clusterer.predict(dev_vectors)
        cluster_distributions = data_in_cluster_sklearn(train_predict,n_clusters,train_message_ids,train_answer_counters)
        dev_predict_labels = map_cluster_to_data(dev_predict_vects, cluster_distributions)
        n_samples = len(dev_answer_counters)
        n_votes = 50
        n = n_samples
        kl = evaluate_model_kl(dev_labels,dev_predict_labels)
        fraction,count,L_Sdash_Set = model_selection_for_clustering(dev_labels,dev_predict_labels,n_samples,kl,n,cluster_distributions,n_votes,n_clusters)
        results_log.append({"no_clusters":n_clusters,"kl":kl,"loss":L_Sdash_Set})
        # write_model_logs_to_json(folder_name + "/logs/models",cluster_distributions,"cluster_info_"+str(n_clusters))
        # save_trained_model_joblib_sklearn_nlp(folder_name + "/logs/models", clusterer, output_name, n_clusters)
    model_selection(results_log,output_name)
    print ("Completed Experiment on KMeans Labels Only")


def evaluate_model_kl(actual,predict):
    kl_total = []
    for actual_value,predict_value in zip(actual,predict):
        actual_value = generate_pd(actual_value)
        predict_value = generate_pd(predict_value)
        kl = entropy(actual_value,qk=predict_value, base=2)
        kl_total.append(kl)
    return mean(kl_total)

def data_in_cluster_sklearn(cluster_predicitions,no_clusters,data_index,answer_counters):

    labels_of_clusters = {}
    for cluster_id,data_i in zip(cluster_predicitions,data_index):
        labels = answer_counters[str(data_i)]
        try:
            labels_of_clusters[cluster_id] = [i+j for i,j in zip (labels_of_clusters[cluster_id],labels)]
        except:
            labels_of_clusters[cluster_id] = labels
    cluster_information = {}
    for cluster in range(no_clusters):
        try:
            cluster_information[str(cluster)] = generate_pd(labels_of_clusters[cluster]).tolist()
        except:
            cluster_information[str(cluster)] = np.zeros(5).tolist()

    return cluster_information

def generate_pd_data(result):
	total = float(sum(result[0]))
	result = result/total
	return result
    
def generate_pd(result):
    original_result = result
    try:
        total = np.sum(result)
        result = np.array(result, dtype='float64')
        result = result/total
    except RuntimeWarning:
        result = original_result
    return result

def map_cluster_to_data(cluster_labels,cluster_info):
    predictions_to_write = []
    for cluster_assignment in zip(cluster_labels):
        label_distribution = cluster_info[str(cluster_assignment[0])]
        predictions_to_write.append(label_distribution)
    return predictions_to_write

def model_selection(results_log,output_path):
    results_pd = pd.DataFrame(results_log)
    model_value = results_pd[results_pd['loss'] == min(results_pd['loss'])]
    print ("Model Selected for KMM p="+str(model_value['no_clusters'].item())+" with KL="+str(model_value['kl'].item()))

def preprocess_data(input_train_file_name,input_dev_file_name,input_test_file_name,folder_name):

    create_folder(folder_name)
    create_folder(folder_name + "/logs")
    create_folder(folder_name + "/logs/models")

    train_answer_counters,train_message_dict,label_dict = read_labeled_data_KMeans(input_train_file_name)

    dev_answer_counters,dev_message_dict,label_dict = read_labeled_data_KMeans(input_dev_file_name)

    test_answer_counters,test_message_dict,label_dict = read_labeled_data_KMeans(input_test_file_name)

    return train_answer_counters,dev_answer_counters,label_dict,train_message_dict,dev_message_dict,test_answer_counters,test_message_dict

def main():
    train_file,dev_file,test_file,lower,upper,output_file,folder_name = load_default_parameters()
    #Reading Data
    train_answer_counters,dev_answer_counters,label_dict,train_message_dict,dev_message_dict,test_answer_counters,test_message_dict = preprocess_data(train_file,dev_file,test_file,folder_name)
    print ("Training on KM - Labels Only")
    train_kmeans_labels(train_answer_counters,dev_answer_counters, lower,upper,output_file,folder_name,label_dict,train_message_dict,dev_message_dict)


if __name__ == '__main__':
	main()
