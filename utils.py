import pandas as pd


""" Useful data """

weekdays = ["Mon","Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

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
