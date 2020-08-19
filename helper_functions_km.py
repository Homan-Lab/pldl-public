#!/usr/bin/env python
import os, json, nltk, re, string
from sklearn.externals import joblib
import numpy as np
import pickle, gzip
import pdb
import h5py
from collections import defaultdict,OrderedDict
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime
import math
import pandas as pd
import warnings
#LDA on Language
import gensim
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer('english')
#End on LDA Language


def read_json(fname):
    datastore = defaultdict(list)
    if fname:
        with open(fname, 'r') as f:
            datastore = json.load(f)
    return datastore

def get_data_dict (l):
    enuml = enumerate(l)
    fdict = defaultdict(list)
    rdict = defaultdict(list)
    fdict = {k:v for v, k in enuml}
    rdict = {k:v for v, k in fdict.items()}
    return (fdict, rdict)

def write_model_logs_to_json(MODEL_LOG_DIR, results_dict, output_name):
    with open(MODEL_LOG_DIR +"/"+ output_name + ".json", "w") as fp:
        json.dump(results_dict, fp, sort_keys=True, indent=4)
    print ("Saved to "+MODEL_LOG_DIR +"/"+ output_name + ".json")

def read_labeled_data_KMeans(filename):
    answer_counters = defaultdict(list)
    JSONfile = read_json(filename)
    message_dict = compile_tweet_dict(JSONfile["data"])
    (fdict, label_dict) = get_data_dict(JSONfile["dictionary"])
    answer_counters = get_feature_vectors_only(fdict, JSONfile["data"])
    return answer_counters,message_dict,label_dict

def save_trained_model_joblib(MODEL_LOG_DIR, model, output_name, i, j):
    # http://scikit-learn.org/stable/modules/model_persistence.html
    # i in range(LOWER, UPPER)
    # j in range(ITERATIONS)
    model_dir = MODEL_LOG_DIR + '/CL' + str(i) + '/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_dir + "Iter" + str(j) +'.pkl')
    #model.close()

def create_folder(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)


def save_trained_model_joblib_sklearn(MODEL_LOG_DIR, model, output_name, i):
    # http://scikit-learn.org/stable/modules/model_persistence.html
    # i in range(LOWER, UPPER)
    # j in range(ITERATIONS)
    model_dir = MODEL_LOG_DIR + '/CL' + str(i) + '/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_dir + '.pkl')
    #model.close()

def get_feature_vectors_only(fdict, data):
    #output = {}
    output = defaultdict(list)
    for item in data:
        vect = vectorize(fdict, item["labels"])
        total_labels = float(sum(vect))
        vect[:] = [x /total_labels for x in vect]
        item["message_id"] = item["message_id"]
        output[item["message_id"]] = vect
    return output

def compile_tweet_dict(json_list):
    result = {int(x["message_id"]): x["message"] for x in json_list}
    return result
