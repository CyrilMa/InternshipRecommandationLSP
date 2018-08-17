# Pour commencer

## Introduction

Nos travaux implémentent la méthode décrite dans le document "report.pdf" présent dans le dossier principal

- Le dossier principal est consacré à la prédiction d'activité sur les flots de liens
- Le dossier "data/" regroupe les fichiers de données utilisés
- Le dossier "exports/" regroupe les outputs (scoring, fichier de liens prédits) des algorithmes de prédiction d'activité
- Le dossier "graph/" regroupe les fichiers consacrés à la construction de flots de liens

## Prerequis

```
Python3
Numpy
Scipy
Matplotlib
Pandas
Tqdm
Random
NLTK
stop_words
warnings
```

# Fichiers "Flots de Liens"

## Structure

Flots de liens biparties :

```
t u v
...
```

\<float:t\> : time of the link

\<int:u\>,\<int:v\> : pair of nodes

## Construction des Fichiers "Flots de Liens"

La construction d'un fichier de "Flots de Liens" est réalisé dans le dossier "graph"

Comme décrit dans le rapport. Nous procédons à plusieurs étapes ayant chacun leur lot de paramètres :

### Vectorisation d'évènement(Event Vectorisation)

- Les variables lié à l'ancitipation et au calendrier sont à intégré suivant le modèle respectivement ligne 38 et 41 (build_graph_file.py)
- La dimension de la vectorisation est un paramètre de TimeTagVectorizer
- La métrique de similarité et la méthode de reduction de dimension sont à implémenter suivant le modèle dans vectorizer.py et metrics.py

### Clustering

- Il est possible de rapidement modifier la méthode de clustering employé en suivant le modèle fourni

### Intervalle de construction du flot de liens

L'intervalle de construction du flot de liens est à modifier ligne 85 (build_graph_file.py)

# Link Stream Prediction

Repris de https://github.com/ThibaudA/linkstreamprediction

Activity prediction in link streams algorithm

## Default settings

Prediction with and without classes

3 classes by pair activity:
```
  C0: without classes
  C1: pair without interaction during observation
  C2: less than classthreshold=5 links during observation
  C3: more than classthreshold=5 links during observation
  AllClasses: Union of C1, C2 and C3
```

* Activity extrapolation during training: Activity during training prediction period
* Activity extrapolation during real prediction: Extrapolation of observation period activity
* Gradient descent initiation: Random exploration of the parameters space between the parameters indicated in the configuration file for each metric


## Running the prediction

```
cat <data_file>  | python main.py <config_file> <export_file_name>
```
see examples in run.sh

Configuration file structure:
```
<float:tstartobsT> #start time of observation training period
<float:tendobsT> #end time of observation training period
<float:tstartpredT> #start time of prediction training period
<float:tendpredT> #end time of prediction training period
<float:tstartobs> #start time of observation
<float:tendobs> #end time of observation
<float:tendpred> #end time of pred
Metrics #Metrics used:
Metric1 [parameters]
Metric2 [parameters]
Metric3 [parameters]
EndMetrics
[Options]
Commentaries:
Bla bla
```

Metrics available:

```
PairActivityExtrapolation
commonNeighbors
jaccardIndex
PairActivityExtrapolationNbLinks<int:k>
PairActivityExtrapolationTimeInter<float:k>
```

parameters: <float>,<float>

## Output:

By default the algorithm output the prediction quality and the metric combination used by during the prediction by classes.
