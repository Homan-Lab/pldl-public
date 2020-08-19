import pandas as pd
import pdb
from itertools import groupby
from collections import OrderedDict,defaultdict
from collections import Counter
import json
import argparse
import os
import numpy as np
import pickle
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
import time
import os
from tqdm import tqdm
# from split_data import split_items_train_dev_test
credentials_path = "" #refer to the readme

def split_items_train_dev_test(tweetid_askey_dict,TRAINPCT,TESTPCT,DEVPCT):
    # Should be working with either of these data structures
    # tweetid_answer_counter[message_id] = answers_counter
    # tweetid_worker_responses[message_id][worker] = v[worker]
    #dataitems = list(tweetid_askey_dict.keys())
    
    dataitems = tweetid_askey_dict.sample(frac=1)
    print(len(dataitems))

    # https://stackoverflow.com/a/38251213/2709595
    # 50% - train set
    # 75% - (train + dev) set
    train_items, dev_items, test_items = np.split(dataitems, [int(TRAINPCT * len(dataitems)), int((TRAINPCT+DEVPCT) * len(dataitems))])
    print(len(train_items), len(train_items)/float(len(dataitems)))
    print(len(dev_items), len(dev_items)/float(len(dataitems)))
    print(len(test_items), len(test_items)/float(len(dataitems)))

    #write_datasplit_to_json(train_items, dev_items, test_items, output_name)

    return train_items, dev_items, test_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Data folder",default="/data")
    parser.add_argument("--col_id", help="Message ID columm name for CSV", default="status_id")
    parser.add_argument("--col_message", help="Text column name for CSV", default = "status_message")
    # parser.add_argument("--colQuestion_mt", help="Labels column name")
    # parser.add_argument("--colQuestion_f8", help="Labels CSV column name")
    parser.add_argument("--label_dict", help="Label dictionary",default="fb_label_dict.json")
    #parser.add_argument("--trainpct", help="Train Split percentage",default = 0.5)
    #parser.add_argument("--testpct", help="Test Split percentage",default = 0.25)
    # parser.add_argument("--input_split_file", help="Input split JSON file")
    parser.add_argument("--id", help="Identifier",default="fb")
    parser.add_argument("--foldername", help="Output Folder name", default = "data/facebook/processed")
    args = parser.parse_args()
    input_file = args.input_folder
    col_message_ID = args.col_id
    col_message_text = args.col_message
    id = args.id
    foldername = args.foldername
    label_dict_json = args.label_dict
    data_towrite = {}
    label_dict = read_json_data_dict(label_dict_json)
    data_towrite['dictionary'] = label_dict

    # dframe = csv_read(input_file,col_message_ID,col_message_text,label_dict)
    dframe = read_folder("data/facebook/raw",col_message_ID,col_message_text,label_dict)
    # Code to select top 3000 items
    dframe = dframe.sort_values(by=['total_reactions'],ascending=False)
    dframe = dframe.head(3000)
    del dframe["total_reactions"]
    # pdb.set_trace()
    # dframe = dframe.sample(n=3000,replace=False,random_state=1)
    dframe = detect_language_eliminate(dframe)
    dframe = dframe.sample(n=2000,replace=False,random_state=1)
    # pdb.set_trace()
    TRAINPCT = 0.5
    TESTPCT = 0.25
    DEVPCT = 0.25
    dframe = rename_fb_dframe(dframe)
    train_items, dev_items, test_items = split_items_train_dev_test(dframe,TRAINPCT,TESTPCT,DEVPCT) #Data Split

    del label_dict[0] # removing num_likes
    sum_of_cols(dframe,label_dict)
    write_results(train_items,label_dict,foldername+"/"+id+"_train.json")
    write_results(dev_items,label_dict,foldername+"/"+id+"_dev.json")
    write_results(test_items,label_dict,foldername+"/"+id+"_test.json")
    # pdb.set_trace()

def sum_of_cols(dframe,label_dict):
    count = 0.0
    for label in label_dict:
        # print("Sum of "+label+" = "+str(sum(dframe[label])))
        count = count + sum(dframe[label])
    
    for label in label_dict:
        print("Sum of "+label+" = "+str(sum(dframe[label]))+" | "+str(100.0*sum(dframe[label])/count)+"%")
        # count = count + sum(dframe[label])
    print("Total:"+str(count))

def read_folder(foldername,col_message_ID,col_message_text,label_dict):
    result = pd.DataFrame()
    count=0
    for filename in os.listdir(foldername):
        if filename.endswith(".csv") or filename.endswith(".py"): 
            # if count>15:
            #     break
            print(filename)
            result_pd = csv_read(foldername+"/"+filename,col_message_ID,col_message_text,label_dict)
            if result_pd.empty:
                continue
            result = pd.concat([result,result_pd])
            count=count+1
        else:
            continue
    return result

def csv_read_full(csv_location,col_message_ID,col_message_text,col_label_dict):

    dframe = pd.read_csv(csv_location)
    #Cleaning remove NaN
    dframe = dframe.dropna()
    dframe_labels = dframe[dframe.columns[-len(col_label_dict):]]
    dframe_labels["id"] = dframe_labels.index+1
    dframe_data = dframe[[col_message_ID,col_message_text]]
    dframe_data["id"] = dframe_data.index+1
    dframe_combined = dframe_data.merge(dframe_labels)
    del dframe_combined["id"]
    return dframe_combined

def csv_read(csv_location,col_message_ID,col_message_text,col_label_dict):

    dframe = pd.read_csv(csv_location)
    if not dframe.empty:
        #Cleaning remove NaN
        dframe = dframe.dropna()
        dframe_labels = dframe[dframe.columns[-len(col_label_dict):]]
        dframe_labels["id"] = dframe_labels.index+1
        dframe_data = dframe[[col_message_ID,col_message_text]]
        dframe_data["id"] = dframe_data.index+1
        dframe_combined = dframe_data.merge(dframe_labels)
        del dframe_combined["id"]
        del dframe_combined["num_likes"]
        dframe_combined['total_reactions'] = dframe_combined['num_hahas']+dframe_combined['num_sads']+dframe_combined['num_loves']+dframe_combined['num_wows']+dframe_combined['num_angrys']
        dframe_combined = dframe_combined[dframe_combined.total_reactions != 0]
        # dframe_combined = dframe_combined[dframe_combined.total_reactions == 50]
        # del dframe_combined["total_reactions"]
        # dframe_combined = dframe_combined.sample(n=30,replace=False,random_state=1)
    else:
        dframe_combined=dframe
    return dframe_combined

def detect_language_eliminate(dataset):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    translate_client = translate.Client(credentials=credentials)
    count = 0
    # text_data = dataset["status_message"]
    for i, row in tqdm(dataset.iterrows(),total=dataset.shape[0],desc="Detecting Language via Google Translate API"):
        if count%500==0 and count>0:
            time.sleep(100)
        try:
            detected = translate_client.detect_language(row["status_message"])
        except:
            time.sleep(31)
            detected = translate_client.detect_language(row["status_message"])
        language = detected["language"]
        if language!="en":
            # print (language)
            status_message_id = row['status_id']
            dataset = dataset[dataset.status_id != status_message_id]
        count +=1

    return dataset
    
# text = "hello world"
# result = translate_client.detect_language(text)
# print('Text: {}'.format(text))
# print('Confidence: {}'.format(result['confidence']))
# import pandas as pd

def rename_fb_dframe(dframe):
    new_columns = dframe.columns.values
    dframe.columns = dframe.columns.str.replace('status_id','message_id')
    dframe.columns = dframe.columns.str.replace('status_message','message')
    return dframe

def read_json_data_dict(label_dict):
    data_dict = []
    with open(label_dict, 'r') as f:
        data_dict = json.load(f)
    return data_dict["dictionary"]

def write_results(dframe_items,label_dict,output_name):
    output = {}
    output['dictionary'] = label_dict
    data = json.loads(dframe_items.to_json(orient='records'))
    output['data'] = prepare_output(data,label_dict)
    save_to_json(output,output_name)
    
def prepare_output(dframe,label_dict):
    result = []
    for row in dframe:
        row_value = {}
        labels = {}
        status_ids = row['message_id'].split("_")
        row_value['message_id'] = status_ids[1]
        row_value['message'] = row['message']
        for label in label_dict:
            labels[label] = row[label]
        row_value['labels'] = labels
        result.append(row_value)
    return result


def save_to_json(data,outputdir):
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))
    with open(outputdir, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))
        print ("JSON file saved to "+outputdir)

if __name__== "__main__":
    main()
