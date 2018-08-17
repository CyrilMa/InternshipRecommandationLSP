import pandas as pd
import numpy as np

def jaccard(tags_s, show_tags):
    sim_dict = dict()
    for tag_i in tags_s.index:
        sim_dict[tag_i] = dict()
        for tag_j in tags_s.index:
            sim_dict[tag_i][tag_j] = 0

    for tags in show_tags:
        for tag_i in tags:
            for tag_j in tags:
                sim_dict[tag_i][tag_j]+=1

    for tag_i in tags_s.index:
        for tag_j in tags_s.index:
            sim_dict[tag_i][tag_j] /= (tags_s[tag_i] + tags_s[tag_j] - sim_dict[tag_i][tag_j])

    return(pd.DataFrame(sim_dict))

def BP(tags_s, show_tags):
    S = len(show_tags)
    sim_dict = dict()

    for tag_i in tags_s.index:
        sim_dict[tag_i] = dict()
        for tag_j in tags_s.index:
            sim_dict[tag_i][tag_j] = 0

    for tags in show_tags:
        for tag_i in tags:
            for tag_j in tags:
                sim_dict[tag_i][tag_j]+=1

    for tag_i in tags_s.index:
        for tag_j in tags_s.index:
            ki = tags_s[tag_i]
            kj = tags_s[tag_j]
            sim_dict[tag_i][tag_j] = (sim_dict[tag_i][tag_j] - ki*kj/S)/(np.sqrt(ki*(1-ki/S)*kj*(1-kj/S)))

    return(pd.DataFrame(sim_dict))
