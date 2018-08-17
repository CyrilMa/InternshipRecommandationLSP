import pandas as pd
import numpy as np
import collections

path_to_data = "../data/"

""" Useful data """

weekdays = ["Mon","Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
prenoms = pd.read_csv(path_to_data+"prenoms.csv")

""" General functions on series """

def unique_values_count(serie, sort=False):
    d = dict()
    for key in serie.unique():
        d[key]=0
    for x in serie:
        try:
            d[x]+=1
        except:
            ()
    if(sort):
        return(pd.Series(d).sort_values(ascending=False))
    return(pd.Series(d))

""" Tags related functions """

def tags_extractor(cell):
    import unidecode
    try:
        chains = cell.split(",")
        tags = []
        for c in chains:
            tags.append(unidecode.unidecode(c.replace("[", "").replace("]", "").replace("\"","")))
        return(tags)
    except:
        return([])

def tags_serie(each_tags):
    tags_dict = dict()
    for s_tag in each_tags:
        for tag in s_tag:
            if tag in tags_dict:
                tags_dict[tag]+=1
            else:
                tags_dict[tag]=1

    serie = pd.Series(tags_dict)
    serie = serie.sort_values(ascending=False)
    return(serie)

def zieglerILS(g, u, in_metric):
    # Improving Recommendation Lists Through Topic Diversification - Ziegler - 2005
    if type(in_metric) is dict:
        metric = (lambda u,v : in_metric[u][v])
    else:
        metric = in_metric

    ils = 0
    for v in g.neighbors(u):
        for w in g.neighbors(u):
            if w != v:
                ils+=g[u][w]["weight"]*metric(v,w)
    return(ils/2)

def metric_dict(generator):
    d = dict()
    for u,v,p in generator:
        if u not in d:
            d[u] = dict()
        d[u][v]=p
    return(d)

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score

def k_means_equi(vectors, prod = 4, min_size = 100, max_size = 500):
    MIN_SIZE = min_size
    MAX_SIZE = max_size
    clusters = np.zeros(vectors.shape[0],dtype=int)
    counter = pd.Series(collections.Counter(clusters))
    last_max = np.inf
    cluster_centers = dict()

    while max(counter.values)>MAX_SIZE and last_max != max(counter.values):
        last_max = max(counter.values)
        last_n_cluster = max(clusters)

        i = counter[counter>MAX_SIZE].sort_values(ascending=False).keys()[0]
        km = KMeans(prod*counter[i]//MAX_SIZE, init="random")

        reduced_vectors = vectors[list(np.where(clusters == i)[0])]
        reduced_clusters = km.fit_predict(reduced_vectors)
        reduced_counter = pd.Series(collections.Counter(reduced_clusters))
        while(min(reduced_counter.values)<MIN_SIZE):
            j = reduced_counter[reduced_counter<MIN_SIZE].sort_values().keys()[0]
            clusters_dist = pd.Series([np.linalg.norm(km.cluster_centers_[j]-km.cluster_centers_[k]) for k in reduced_counter.index],index = [k for k in reduced_counter.index])
            clusters_dist[j] = np.inf
            k = clusters_dist.sort_values().keys()[0]
            km.cluster_centers_[k] = (reduced_counter[k] * km.cluster_centers_[k] + reduced_counter[j] * km.cluster_centers_[j])/(reduced_counter[k]+reduced_counter[j])
            km.cluster_centers_[j] = np.inf
            np.place(reduced_clusters, reduced_clusters == j, k)
            reduced_counter = pd.Series(collections.Counter(reduced_clusters))
        clusters[list(np.where(clusters == i)[0])] = last_n_cluster + reduced_clusters + 1
        counter = pd.Series(collections.Counter(clusters))
        for i in np.unique(reduced_clusters):
            cluster_centers[last_n_cluster+i] = km.cluster_centers_[i]

    return(clusters, cluster_centers, counter, calinski_harabaz_score(vectors, clusters))