import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def build_expected_repartition(sample, expected):
    dict_repartition = dict()

    for line in expected.itertuples():
        if line.customer in sample:
            if (line.customer,line.cluster) in dict_repartition.keys():
                dict_repartition[line.customer,line.cluster] += 1
            else:
                dict_repartition[line.customer,line.cluster] = 1
    expected_repartition = pd.Series(list(dict_repartition.values()),index = pd.MultiIndex.from_tuples(dict_repartition.keys(), names=["customer","cluster"]))
    vol_by_cust = expected_repartition.groupby(level=0).count()

    for cust,cl in expected_repartition.index:
        dict_repartition[cust,cl] = dict_repartition[cust,cl]/vol_by_cust.loc[cust]
                
    expected_repartition = pd.Series(
        list(dict_repartition.values()),
        index = pd.MultiIndex.from_tuples(dict_repartition.keys(), 
        names=["customer","cluster"]))
    
    return(expected_repartition)

def build_predicted_repartition(sample, predicted):
    dict_repartition = dict()

    for line in predicted.itertuples():
        s,c = line.customer, line.cluster
        if line.volume>0.01:
            if s in sample:
                dict_repartition[s,c]=np.round(line.volume,2)
    
    predicted_repartition = pd.Series(
        list(dict_repartition.values()),
        index = pd.MultiIndex.from_tuples(dict_repartition.keys(), 
        names=["customer","cluster"]))
    
    return(predicted_repartition)

def sse(df):
    return (sum((df.expected-df.predicted)**2))


def fast_scoring(prediction_file, expected_file, config_dict):
    
    # Load expected file
    datafile = pd.read_table(expected_file, sep=" ",names=["time","customer","cluster"])

    expected = datafile[datafile.time >= config_dict["tstartpred"]][datafile.time <= config_dict["tendpred"]]
    before =  datafile[datafile.time >= config_dict["tstartobs"]][datafile.time <= config_dict["tendobs"]]
    sample = set(expected.customer.unique()) & set(before.customer.unique())

    # Load prediction file
    predicted = pd.read_table(prediction_file,sep=" ",names=["customer", "cluster", "volume"])

    del datafile, before
    expected_repartition = build_expected_repartition(sample, expected)
    predicted_repartition = build_predicted_repartition(sample, predicted)

    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(set(predicted_repartition.index)|set(expected_repartition.index)))
    df["expected"] = expected_repartition
    df["predicted"] = predicted_repartition
    df = df.fillna(0)
    del expected_repartition, predicted_repartition
    
    each_sse = df.groupby(level=0).apply(sse)
    return(each_sse.mean())
