
#Script to import the data from CF data for Jobs. The variables are declared in the main for the parameters
#change them as required.
import pandas as pd
import pdb
from itertools import groupby
from collections import OrderedDict
from collections import Counter
import json
import argparse
import os
from ldl_utils import read_json
import pickle
import numpy as np
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Input CSV Filename")
    parser.add_argument("--trainpct", help="Train Split percentage",default = 0.5)
    parser.add_argument("--testpct", help="Test Split percentage",default = 0.25)
    parser.add_argument("--colTweetID", help="Tweet ID Columm Name", default="message_id-1")
    parser.add_argument("--colTweetText", help="Tweet Text Column Name", default = "message-1")
    parser.add_argument("--colQuestion", help="Labels Column Name")
    parser.add_argument("--dataDict", help="Label Dictionary")
    parser.add_argument("--id", help="Identifier")
    parser.add_argument("--foldername", help="Main Folder name", default = "data/jobQ123CF")
    args = parser.parse_args()
    input = args.input_file
    col_tweet_ID = args.colTweetID
    col_tweet_text = args.colTweetText
    col_label = args.colQuestion
    id = args.id
    foldername = args.foldername
    dataDict = args.dataDict
    TRAINPCT = float(args.trainpct)
    TESTPCT = float(args.testpct)
    DEVPCT = 1-TRAINPCT - TESTPCT
    data_towrite = {}
    data_towrite['dictionary'] = read_json_data_dict(dataDict,foldername)
    processed_path = foldername+"/processed/"+id+"/"+id+".pkl"
    dframe = pickle.load(open(processed_path, "rb"))
    drf,dfresults = message_dict(dframe,col_tweet_ID,col_tweet_text) #Compile Tweet ID List
    train_items, dev_items, test_items = split_items_train_dev_test(dfresults,TRAINPCT,TESTPCT,DEVPCT) #Data Split

    train_path = foldername+"/processed/"+id+"/split/"+id+"_train.json"
    write_split_data(dframe,col_tweet_ID,col_tweet_text,col_label,train_items,data_towrite['dictionary'],train_path)

    dev_path = foldername+"/processed/"+id+"/split/"+id+"_dev.json"
    write_split_data(dframe,col_tweet_ID,col_tweet_text,col_label,dev_items,data_towrite['dictionary'],dev_path)

    test_path = foldername+"/processed/"+id+"/split/"+id+"_test.json"
    write_split_data(dframe,col_tweet_ID,col_tweet_text,col_label,test_items,data_towrite['dictionary'],test_path)


#CSV to Tweet List
def message_dict(dframeIn,col_tweet_ID,col_tweet_text):
    results = []
    results_dict = []
    for (tweet_id,tweet), bag in dframeIn.groupby([col_tweet_ID,col_tweet_text]):
        results.append(OrderedDict([(tweet_id,tweet)]))
        results_dict.append(tweet_id)
    return results,results_dict

def write_split_data(dframeIn,col_tweet_ID,col_tweet_text,col_label,split_file,label_dict,path):
    results = []
    data_to_write = {}
    for (tweet_id,tweet), bag in dframeIn.groupby([col_tweet_ID,col_tweet_text]):
        if tweet_id in split_file:
            contents_df = bag.drop([col_tweet_text,col_tweet_ID], axis=1)
            labels = Counter(contents_df[col_label])
            #iterate the labels to find labels with /n
            label_list = labels.items()

            for i in label_list:
                if "\n" in i[0]:
                    l_words = i[0].splitlines()
                    l_count = Counter(l_words)
                    labels = labels + l_count
                    del labels[i[0]]

            results.append(OrderedDict([("message_id", tweet_id),
                                        ("message", tweet),
                                        ("labels", labels)]))
    data_to_write['dictionary'] = label_dict
    data_to_write['data'] = results
    save_to_json(data_to_write,path)

def save_to_json(labels,outputdir):
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'w') as outfile:
        outfile.write(json.dumps(labels, indent=4))
        print "JSON file saved to "+outputdir

def read_json_data_dict(dataDict,foldername):
    data_dict = []
    with open(foldername + "/"+dataDict, 'r') as f:
        data_dict = json.load(f)
    return data_dict["dictionary"]

def split_items_train_dev_test(tweetid_askey_dict,TRAINPCT,TESTPCT,DEVPCT):
    # Should be working with either of these data structures
    # tweetid_answer_counter[message_id] = answers_counter
    # tweetid_worker_responses[message_id][worker] = v[worker]
    #dataitems = list(tweetid_askey_dict.keys())
    dataitems = tweetid_askey_dict
    print(len(dataitems))

    # https://docs.python.org/3/library/random.html#random.shuffle
    random.shuffle(dataitems)

    # https://stackoverflow.com/a/38251213/2709595
    # 50% - train set
    # 75% - (train + dev) set
    train_items, dev_items, test_items = np.split(dataitems, [int(TRAINPCT * len(dataitems)), int((TRAINPCT+DEVPCT) * len(dataitems))])
    print(len(train_items), len(train_items)/float(len(dataitems)))
    print(len(dev_items), len(dev_items)/float(len(dataitems)))
    print(len(test_items), len(test_items)/float(len(dataitems)))

    #write_datasplit_to_json(train_items, dev_items, test_items, output_name)

    return train_items, dev_items, test_items

if __name__== "__main__":
    main()
