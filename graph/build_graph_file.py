import pandas as pd
from tqdm import tqdm
import datetime as time
import random
import numpy as np
import utils

import warnings
warnings.filterwarnings("ignore")

path_to_data = "../data/"
MAX_DAY = 400
n_cluster = [20]
n_samples = 5000
file_name = "graph_file_tag_20_only_tags_"

# Events Vectorization
print("Events Vectorization")
from vectorizer import TimeTagVectorizer

vectorizer = TimeTagVectorizer(20)

# La vectorisation est une opération couteuse en temps, si elle a déjà été réalisé on la reprend
try:
    dataset = vectorizer.load("dataset_tag_anticipation_dim_20.csv")

# Sinon on la réalise 
except:
    shows = pd.read_csv(path_to_data+"shows.csv")[["delight_show_id", "event_provider_types"]]
    events = pd.read_csv(path_to_data+"events.csv")
    events = events.merge(shows, how="left", on="delight_show_id")[["delight_event_id", "event_provider_types"]]
    time_events = pd.read_csv(path_to_data+"time_events_above_20.csv")
    dataset = time_events.merge(events, how="left", on="delight_event_id").set_index("delight_event_id")

    del events,shows,time_events

    # Features liées à l'anticipation
    delta_cols = [] #list(dataset.filter(like="delta_",axis=1).columns)

    # Features liées au calendrier
    cal_cols = [] #list(dataset.filter(like="tr_d",axis=1).columns)
    
    dataset = vectorizer.transform(dataset[delta_cols], dataset.event_provider_types)
    vectorizer.df = (vectorizer.df-vectorizer.df.mean())/vectorizer.df.std()
    vectorizer.df = vectorizer.df.dropna()
    vectorizer.save("dataset_tag_anticipation_dim_20.csv")

    dataset = vectorizer.df

# Clustering
print("Clustering")
from sklearn.cluster import KMeans,MiniBatchKMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage,fcluster

occurences = pd.read_table(path_to_data+"occurences_customers.txt",sep=" ",names=["id", "occ"])
occurences = random.sample(occurences[occurences.occ >=  20][occurences.occ <=  100].id.tolist(),n_samples)
ids = pd.Series(range(100000,100000+n_samples), index = occurences)

logs_iter = pd.read_csv(path_to_data+'transactions_above_15.csv', low_memory=False, iterator=True, chunksize=100000)
possible_events = set(dataset.index)

dataset = dataset.loc[possible_events]

cl,centers,co,_ = utils.k_means_equi(dataset.as_matrix(),4,100,800)
print(co.describe())

ids.to_csv("customers_"+file_name+"%d.csv" % max(centers.keys()))
file = open(file_name+"%d.txt" % max(centers.keys()), "w") 

dataset["clustering_%d" % max(centers.keys())] = cl
dataset[["clustering_%d" % max(centers.keys())]].to_csv("events_clustering_%d.csv" % max(centers.keys()))

# Users extraction
print("Extraction")

from datetime import datetime
def timestamp(time):
    relative = time-datetime(2016,1,1)
    return int(60*24*relative.days)

for logs in tqdm(logs_iter):
    logs.order_datetime = pd.to_datetime(logs.order_datetime)
    logs = logs[logs.order_datetime.isnull() == False]
    logs = logs[logs.order_datetime.apply(lambda t : t.year) >= 2016][logs.order_datetime.apply(lambda t : t.year) <= 2017]
    for line in logs.itertuples():
        if line.delight_customer_id in occurences and line.delight_event_id in possible_events:
            file.write("%d %d %d\n" % (timestamp(line.order_datetime), ids[line.delight_customer_id], dataset.loc[line.delight_event_id]["clustering_%d" % max(centers.keys())]))
    del logs

file.close()

# rearranging files
df = pd.read_table(file_name+"%d.txt" % max(centers.keys()), sep=" ", names="ABC").sort_values("A")
df.to_csv(file_name+"%d.txt"% max(centers.keys()), sep=" ",header=False, index=False)
del df
