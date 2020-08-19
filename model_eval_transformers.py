import os
#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import argparse
import sys
import pdb
import numpy as np
from tqdm import tqdm
from statistics import mean,stdev
import math
import random
import pandas as pd
import collections
from collections import defaultdict #https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work

from functools import partial
import multiprocessing
from scipy.stats import multinomial

lda_flag = 1 #gensim starts the clusters from 0 while others start from 1. 


def cluster_sampler(data_pool,n_samples,clusters_dists,n_votes,n_topics):
    # Algorithm 3 Cluster Sampler AI Stats Paper
    # Randomly picks data item then reads it cluster and repeats for the n_samples needed.
    # The empirical labels are read using the message_id in model_selection() function
    total_KL = 0.0
    total_MD = 0.0
    for i in range(n_samples):
        random_cluster = random_selector(data_pool)
        cluster = random_cluster
        predicted_label_distribution = extract_dist_of_cluster(cluster,clusters_dists,n_topics)
        random_sample = sample_from_dist(predicted_label_distribution,n_votes)
        #items_in_cluster = select_items_in_cluster(data_pool,cluster)
        random_label_distribution = generate_pd(random_sample)#bootstrap_sampler(items_in_cluster,10)
        KL = KL_pred2sample(predicted_label_distribution, random_label_distribution)
        # Begin Multinomial Distribution
        MD = multinomial_distribution(random_sample,n_votes)
        #End Multinomial 
        total_KL +=KL
        total_MD +=MD
    L_SDash = float(total_KL/n_samples)
    L_SDash_MD = float(total_MD/n_samples)
    return L_SDash,L_SDash_MD

def KLdivergence(P, Q):
    # from Q to P
    # https://datascience.stackexchange.com/a/26318/30372
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001

    P = P + epsilon
    Q = Q + epsilon

    return np.sum(P * np.log(P/Q))

def generate_pd_of_cluster(random_samples,empirical_labels):
    #needed for Algorithm #3
    empirical_label_set = []
    sampled_label_set = []
    for random_item in random_samples:
        x_id = int(random_item["message_id"])
        empirical_label = empirical_labels[x_id]
        empirical_label_set.append(empirical_label)

    total_of_labels = np.asarray(empirical_label_set)
    total_of_labels = sum(total_of_labels)
    pd_of_labels = generate_pd(total_of_labels)
    return pd_of_labels

def extract_dist_of_cluster(random_cluster,clusters_dists,n_topics):
    random_cluster_in_dist = random_cluster#the cluster assignments start from 0 (predicts) however the cluster assignments are from 1
    raw_dist_of_cluster = random_cluster_in_dist #clusters_dists[str(random_cluster_in_dist)]
    dist_of_cluster = []
    try:
        dist_sum = sum(raw_dist_of_cluster)
        dist_of_cluster = raw_dist_of_cluster
    except:
        for each in raw_dist_of_cluster.split(' + '):
            proba = float(each.split('*')[0])
            choice_index = int(each.split('*')[1].replace('"', ''))
            dist_of_cluster.append(round(proba,2))
    return dist_of_cluster

def cluster_counter(dataset):
    dframe = pd.DataFrame(dataset)
    dframe_cluster = dframe['cluster']
    cluster_counts = dframe_cluster.value_counts()
    cluster_counts = cluster_counts.sort_index()
    return cluster_counts

def multinomial_distribution(item_counts,n_samples):
    item_counts = np.array(item_counts)
    sum_items = item_counts.sum()
    item_counts = item_counts.astype(float) #To avoid any ints not converting to float
    pd_counts = (item_counts/sum_items) #PJ
    # md_value = multinomial.pmf(item_counts, n=sum_items, p=pd_counts)  
    md_value = multinomial.logpmf(item_counts, n=sum_items, p=pd_counts)

    return float(md_value)


def model_selection_for_clustering(empirical_labels,cluster_predict_labels,n_samples,L_S,n,clusters_dists,n_votes,n_topics):
    # Algorithm 2 from AI Stats Paper
    count = 0
    sample_type = "cluster"
    # print ("Model Selection for Pooling, using "+sample_type+" sampler")
    L_Sdash_Set = []
    L_Sdash = 0.0
    MD_Set = []
    tqdm_label = "Sampling "+str(sample_type)
    for i in range(n):
        L_Sdash,MD = cluster_sampler(cluster_predict_labels,n_samples,clusters_dists,n_votes,int(n_topics)) #S_dash
        MD_Set.append(MD)
        L_Sdash_Set.append(L_Sdash)
        if (L_Sdash>L_S):
            count+=1

    count = float(count)
    n = float(n)
    fraction = float(count/n)
    L_S_mean = mean(L_Sdash_Set)
    L_S_stdev = stdev(L_Sdash_Set)
    diff = abs(L_S_mean-L_S)
    sampled_value = (diff)/L_S_stdev
    return fraction,count,sampled_value


def round_up_label_values(labels):
    label = []
    sum_of_labels = sum(labels)

    if (abs(sum_of_labels)>0):
      for item in labels:
        label.append(round(item,2))
    else:
      for item in labels:
        label.append(round(item,1))
    return label

def KL_empirical2pred(empirical_pcts, prediction_proba):
    KLsum = []

    for pair in zip(empirical_pcts, prediction_proba):
        empirical_pct = np.asarray(pair[0])
        prediction_pct = np.asarray(pair[1])

        KL = KLdivergence(empirical_pct, prediction_pct)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)

    KL = np.mean(KLsum)
    #print('KL divergence: ', KL)
    return KL

def KL_pred2sample(predicted_ldl, sampled_ldl):
    predicted_ldl = np.asarray(predicted_ldl)
    sampled_ldl = np.asarray(sampled_ldl)
    KL = KLdivergence(predicted_ldl,sampled_ldl)
    return KL

def KL_empirical2cluster(empirical_pcts, cluster):
    KLsum = []

    for empirical in empirical_pcts:
        empirical_pct = np.asarray(empirical_pcts[empirical])
        empirical_pct = generate_pd(empirical_pct)
        prediction_pct = np.asarray(cluster)

        KL = KLdivergence(empirical_pct, prediction_pct)
        if (math.isnan(KL)):
            KL = 0.0

        KLsum.append(KL)

    KL = np.mean(KLsum)
    #print('KL divergence: ', KL)
    return KL

def sample_from_dist(dist,n_votes):
    no_choices = len(dist)
    #the converstions to the distribution is due to the way how the np.random.choice handles things
    #when the sum is not equal to 1 (absolute) it throws and error
    #in our PDs the sum is 1.0 or 1.00000001 or 0.999999 due to our computations
    #https://stackoverflow.com/questions/25985120/numpy-1-9-0-valueerror-probabilities-do-not-sum-to-1
    dist = round_up_label_values(dist)
    dist = np.array(dist)
    dist /= dist.sum()
    dist = dist.astype('float64')

    try:
        sample_assignments = np.random.choice(no_choices, n_votes, p=dist)
    except:
        dist = [1.00/no_choices for i in range(no_choices)]
        sample_assignments = np.random.choice(no_choices, n_votes, p=dist)
    samples = collections.Counter(sample_assignments)
    sample = []
    for choice in range(no_choices):
        if (samples[choice]):
            sample.append(samples[choice])
        else:
            sample.append(0)
    return sample

def select_items_in_cluster(data_pool,cluster):
    sequence = []

    for data_item in data_pool:
        if data_item['cluster'] == cluster:
            sequence.append(data_item)
    return sequence

def random_selector(data_items):
    item = random.choice(data_items)
    return item

def generate_pd(result):
    original_result = result
    try:
        total = np.sum(result)
        result = np.array(result, dtype='float64')
        result = result/total
    except RuntimeWarning:
        result = original_result
    return result