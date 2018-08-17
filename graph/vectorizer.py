import pandas as pd
import numpy as np

# Dim reduction models
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

# Local imports
import metrics
import utils

class TimeTagVectorizer():

    _default_metrics = {"bp": metrics.BP,"jaccard": metrics.jaccard}
    _default_models = {"isomap": lambda k : Isomap(15, k),"pca": PCA}

    def __init__(self, k=10, min_occ_tag = 30, sim_metrics = "bp", model_dim_reduction = "isomap", **args):
        self.k = k
        self.min_occ_tag = min_occ_tag

        # Mesure de similarité
        if isinstance(sim_metrics, str):
            self.sim_metrics_ = TimeTagVectorizer._default_metrics[sim_metrics]
        else:
            self.sim_metrics_ = sim_metrics

        # Mesure de similarité
        if isinstance(sim_metrics, str):
            self.sim_metrics_ = TimeTagVectorizer._default_metrics[sim_metrics]
        else:
            self.sim_metrics_ = sim_metrics

        # Modèle pour la réduction de dimension
        if isinstance(model_dim_reduction, str):
            self.model_dim_reduction_ = TimeTagVectorizer._default_models[model_dim_reduction]
        else:
            self.model_dim_reduction_ = model_dim_reduction

    def transform(self, time_df, tag_col, calendrier=None):
        df_tag = pd.DataFrame(time_df)

        # tags
        sim_df = self.sim_matrix(tag_col)
        mapped = self.dim_reduction(sim_df)
        for i in range(mapped.shape[1]):
            sim_df["tag_component_"+str(i)] = mapped[:,i]
        vectors = []
        for i in range(len(self.each_show_kept_tags)):
            line = self.each_show_kept_tags[i]
            vectors.append([np.mean([sim_df.loc[tag]["tag_component_"+str(i)] for tag in line]) for i in range(mapped.shape[1])])
        self.df = pd.concat([time_df, pd.DataFrame(vectors, index = time_df.index, columns = ["tag_component_"+str(i) for i in range(mapped.shape[1])])],axis=1,join ="inner")


        # cal
        if calendrier:
            pca = PCA(self.k)
            mapped = pca.fit_transform(calendrier.as_matrix())

            self.df = pd.concat([self.df, pd.DataFrame(mapped, index = time_df.index, columns = ["cal_component_"+str(i) for i in range(mapped.shape[1])])],axis=1,join ="inner")

        return(self.df)

    def sim_matrix(self, tag_col):

        # Tag extraction
        each_show_tags = [utils.tags_extractor(cell) for cell in tag_col]
        tags_s = utils.tags_serie(each_show_tags)
        self.kept_tags_s = tags_s[tags_s>self.min_occ_tag]
        self.each_show_kept_tags = [[tag for tag in tags if tag in self.kept_tags_s.index] for tags in each_show_tags]

        sim_df = self.sim_metrics_(self.kept_tags_s, self.each_show_kept_tags)
        return(sim_df)

    def dim_reduction(self, sim_df):
        model = self.model_dim_reduction_(self.k)
        model.fit(sim_df.as_matrix())
        return(model.transform(sim_df))

    def load(self, filename):
        return(pd.read_csv(filename, index_col = "delight_event_id"))

    def save(self, filename):
        self.df.to_csv(filename)
