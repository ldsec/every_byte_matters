import copy
import random
import json
import sys
import glob
import numpy as np
from constants import DATASET_DIR, TEST_PERCENTAGE, N_FOLDS, AGING_DATA_DIR
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import lib.build_datasets as build_datasets
import lib.defenses as defenses
import lib.plot_builder as plot_builder
from lib.defenses import CostAggregate

REBUILD = False
CACHE_FILE = 'aging2'
CACHE_FILE2 = 'aging_xy2'
CACHE_FILE3 = 'aging_xy3'
DEVICES_TO_INCLUDE = ['HuaweiWatch2']

random.seed(0)




# ===============================
# Functions
def extract_features(xy, unique_from=46, unique_to=1006,
                     unique_granularity=1, unique_deltas=[1005, 46, 0], to_withdraw=[]): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size

    xs = xy['xs']
    ys = xy['ys']
    f = dict()
    
    bins = np.arange(0, 1000, step = unique_granularity)

    def deltas(serie):
        out = []
        i = 1
        while i<len(serie):
            out.append(serie[i]-serie[i-1])
            i += 1
        return out

    def extract_bins(x):
        if x > bins[-1]:
            b = bins[-1] + 10
        else:
            b = bins[np.digitize(x, bins, right = True)]
        return b

    def take(arr, n=30):
        if len(arr) > n:
            return arr[:30]
        return arr

    def stats(key, data):
        if len(data) == 0:
            data=[-1]
        f['min_'+key] = np.min(data)
        f['mean_'+key] = np.mean(data)
        f['max_'+key] = np.max(data)
        f['count_'+key] = len(data)
        f['std_'+key] = np.std(data)
        #f['kurtosis_'+key] = kurtosis(data)


    # general statistics
    stats("sizes_non_null", [abs(y) for y in ys if abs(y) > 0])
    stats("sizes_outgoing", [abs(y) for y in ys if y > 0])
    stats("sizes_incoming", [abs(y) for y in ys if y < 0])
    stats("sizes_outgoing2", [abs(y) for y in ys if y > unique_from])
    stats("sizes_incoming2", [abs(y) for y in ys if y < -unique_from])

    # unique packet lengths [Liberatore and Levine; Herrmann et al.]
    lengths = dict()
    for i in range(unique_from, unique_to):
        lengths[str(i)] = 0
    for y in ys:
        if str(abs(y)) in lengths:
            lengths[str(abs(y))] += 1

    lengths_array = list(lengths.values())
    stats("count_sizes_", lengths_array)

    # global stats about len
    for l in lengths:
        f['count_sizes_'+str(l)] = extract_bins(lengths[l])


    # statistics about timings
    for u in unique_deltas:
        xs_filtered = [xs[i] for i, y in enumerate(ys) if abs(y) > u]
        x_deltas = []
        i=0
        while i<len(xs_filtered):
            x_deltas.append(xs_filtered[i]-xs_filtered[i-1])
            i += 1
        stats("time_deltas_"+str(u), x_deltas)
    for feature in to_withdraw:
        if feature in f:
            del f[feature]

    return f


def build_features_labels_dataset(events, defense=None):
    data = []
    labels = []
    features_names = []
    # shape: data is a [[Features], [Features], ...]
    cost_aggregator = CostAggregate()

    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                for event in events[device][app][action]:
                    features_dict = dict()
                    cost = None

                    if defense is not None:
                        event_defended, cost = defenses.apply_defense(event, defense)
                        features_dict = extract_features(event_defended)
                        cost_aggregator.add_cost(cost, label=app)

                    else:
                        features_dict = extract_features(event)
                    features = list(features_dict.values())
                    features_names = list(features_dict.keys())
                    

                    name = app
                    if name == "Mobills":
                        app = "Mobilis"

                    data.append(features)
                    labels.append(app)

    return [data, labels, features_names, cost_aggregator]



# Fit Random Forest
def random_forest(features_names, X_train, y_train, X_test, y_test, n_trees=1000, rfe_nfeatures=800, rfe_steps=10):
    clf=RandomForestClassifier(n_jobs=8, n_estimators=n_trees, random_state=0)
    selector = RFE(estimator=clf, n_features_to_select=rfe_nfeatures, step=rfe_steps)
    selector = selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    
    clf2=RandomForestClassifier(n_jobs=8, n_estimators=n_trees, random_state=0)
    clf2.fit(X_train, y_train)
    y_pred=clf2.predict(X_test)
    predicted_probas = clf2.predict_proba(X_test)

    scores = dict(
        accuracy = metrics.accuracy_score(y_test, y_pred),
        precision = metrics.precision_score(y_test, y_pred, average='micro'),
        recall = metrics.recall_score(y_test, y_pred, average='micro'),
        f1score = metrics.f1_score(y_test, y_pred, average='micro'),
    )
    selected_features = []
    i = 0
    while i<len(features_names):
        if selector.support_[i] == 1:
            selected_features.append(features_names[i])
        i+=1

    feature_importance = sorted(zip(clf2.feature_importances_, selected_features), reverse=True)

    return scores, feature_importance, y_pred, predicted_probas

def rf_folds(events, defense=None, rfe_nfeatures=50, rfe_steps=100, n_trees=30):
    
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names, cost = build_features_labels_dataset(events, defense=defense)

    X = np.array(X)
    y = np.array(y)

    scores = []
    y_test_all = []
    y_pred_all = []
    feature_ranks = dict()

    i = 0
    for train_index, test_index in sss.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        s, features, y_pred, predicted_probas  = random_forest(feature_names, X_train, y_train, X_test, y_test, n_trees, rfe_nfeatures=10, rfe_steps=100)

        y_test_all.extend(y_test) 
        y_pred_all.extend(y_pred.tolist())
        
        for proba, feature in features:
            if not feature in feature_ranks:
                feature_ranks[feature] = []

            feature_ranks[feature].append(proba)

        scores.append(s)
        i += 1

    # average scores
    score = {k: [np.mean([value[k] for value in scores]), np.std([value[k] for value in scores])] for k in scores[0]}

    # average features importance
    for f in feature_ranks:
        if len(feature_ranks[f]) < N_FOLDS:
            feature_ranks[f].extend([0] * (N_FOLDS - len(feature_ranks[f])))

    features_and_percentages = []
    for f in feature_ranks:
        features_and_percentages.append((f, np.mean(feature_ranks[f]), np.std(feature_ranks[f])))

    return score, features_and_percentages, y_test_all, y_pred_all, cost

def print2(train_d, test_d, data):
    toprint = {}
    for d in data:
        toprint[d] = data[d]
    toprint['train_day'] = train_d
    toprint['test_day'] = test_d

    print(json.dumps(toprint)+",")



# ===============================
# Entrypoint


APP_DISCARD = ['Calm', 'AppInTheAir', 'Camera', 'MapMyFitness', 'Fit', 'FitWorkout', 'FitBreathe', 'Qardio']

n_trees = 200
rfe_nfeatures = 100
rfe_steps = 100
repeat = 50

data3 = build_datasets.load_cache(CACHE_FILE3)
if REBUILD or data3 is None: 

    data2 = build_datasets.load_cache(CACHE_FILE2)
    if REBUILD or data2 is None: 

        data = build_datasets.load_cache(CACHE_FILE)
        if REBUILD or data is None: 

            # "Aged" data
            def include2(d):
                fname = d[d.rfind("/")+1:].replace('.csv', '')
                parts = fname.split("_")
                device = parts[0].strip()
                app = parts[1].strip()
                return app not in APP_DISCARD and device in DEVICES_TO_INCLUDE

            days = glob.glob(AGING_DATA_DIR+"day-*", recursive=False)

            days_mapped = dict()
            for d in days:
                i = d[d.rfind("-")+1:]
                days_mapped[int(i)] = d+"/"

            sources_flat = []
            sources_per_day = dict()
            for i in days_mapped:
                sources_per_day[i] = build_datasets.find_sources(days_mapped[i])

                sources_per_day[i] = list(filter(include2, sources_per_day[i]))
                random.shuffle(sources_per_day[i], random=lambda: 0) #deterministic shuffle for consistent results
                sources_flat.extend(sources_per_day[i])

            # Reference (day 0)
            day0_sources = build_datasets.find_sources(DATASET_DIR)

            def include1(d):
                fname = d[d.rfind("/")+1:].replace('.csv', '')
                parts = fname.split("_")
                device = parts[0].strip()
                number = int(parts[-1])
                app = parts[1].strip()
                return ("_Open_" in d) and device in DEVICES_TO_INCLUDE and app not in APP_DISCARD and (number >= 6000 and number <= 7000)

            day0_sources = list(filter(include1, day0_sources))        
            random.shuffle(day0_sources, random=lambda: 0) #deterministic shuffle for consistent results
            sources_flat.extend(day0_sources)
            sources_per_day[0] = day0_sources

            # parse sources into events
            build_datasets.find_common_columns(sources_flat)


            days = list(sorted(sources_per_day.keys()))

            events_per_day = dict()
            for i in days:
                if i == 0 :
                    events, _ = build_datasets.cut_all_datasets_in_events(sources_per_day[i], folder=DATASET_DIR)
                else:
                    events, _ = build_datasets.cut_all_datasets_in_events(sources_per_day[i], folder=days_mapped[i])
                events_per_day[i] = events

            data = [events_per_day]
            
            build_datasets.cache(CACHE_FILE, data)
        else:
            print("Loaded cache file")

        [events_per_day] = data

        Xy = dict()
        for key in events_per_day:
            X, y, feature_names, _ = build_features_labels_dataset(events_per_day[key], defense=None)
            Xy[key] = dict(X=X, y=y, feature_names=feature_names)

        data2 = [Xy]
        build_datasets.cache(CACHE_FILE2, data2)

    else:
        print("Loaded cache file (2)")

    [Xy] = data2

    # train on initial
    X_train_0 = Xy[0]['X']
    y_train_0 = Xy[0]['y']
    feature_names = Xy[0]['feature_names']
    training_labels = list(sorted(set(y_train_0)))
    n_labels = len(training_labels)

    print('Training labels:', n_labels, ",".join(training_labels))


    # high-volume apps
    highvolume = ['AppInTheAir', 'Bring', 'Calm', 'Camera', 'ChinaDaily', 'Citymapper', 'DCLMRadio', 'DiabetesM', 'Endomondo', 'FITIVPlus', 'FindMyPhone', 'Fit', 'FitBreathe', 'FitWorkout', 'FoursquareCityGuide', 'Glide', 'KeepNotes', 'Krone', 'Lifesum', 'MapMyFitness', 'MapMyRun', 'Maps', 'Meduza', 'Mobilis', 'Outlook', 'PlayStore', 'Qardio', 'Running', 'SalatTime', 'Shazam', 'SleepTracking', 'SmokingLog', 'Spotify', 'Strava', 'SalatTime', 'Telegram', 'Translate', 'WashPost', 'Weather']

    days = list(sorted(Xy.keys()))


    self_acc = []
    cumul_acc = []
    cumul_true = []
    cumul_pred = []
    cumul_self_true = []
    cumul_self_pred = []

    for day in days:
        print(day)

        Y_new = Xy[day]['X']
        y_new = Xy[day]['y']

        for _ in range(repeat):
        
            # Accuracy vs Day 0
            X_past_train, _, y_past_train, _ = train_test_split(X_train_0, y_train_0, stratify=y_train_0, train_size=n_labels*7)

            clf_past=RandomForestClassifier(n_estimators=n_trees, random_state=None)
            clf_past.fit(X_past_train, y_past_train)

            X_new_train, X_new_true, y_new_train, y_new_true = train_test_split(Y_new,  y_new, stratify=y_new, test_size=n_labels*3)

            y_new_pred = clf_past.predict(X_new_true)
            accuracy = metrics.accuracy_score(y_new_true, y_new_pred)

            cumul_acc.append(accuracy)
            cumul_true.append(y_new_true)
            cumul_pred.append(y_new_pred)

            # Self accuracy
            clf_new=RandomForestClassifier(n_estimators=n_trees, random_state=None)
            clf_new.fit(X_new_train[:7*n_labels], y_new_train[:7*n_labels])
            y_new_pred = clf_new.predict(X_new_true)
            accuracy = metrics.accuracy_score(y_new_true, y_new_pred)

            cumul_self_true.append(y_new_true)
            cumul_self_pred.append(y_new_pred)
            self_acc.append(accuracy)

        
    data3 = [self_acc, cumul_acc, cumul_true, cumul_pred, cumul_self_true, cumul_self_pred, clf_past]
    build_datasets.cache(CACHE_FILE3, data3)


else:
    print("Loaded cache file (3)")


[self_acc, cumul_acc, cumul_true, cumul_pred, cumul_self_true, cumul_self_pred, clf_past] = data3

cumul_acc_avg = np.array(cumul_acc).reshape((-1,repeat)).mean(axis = 1)
self_acc_avg = np.array(self_acc).reshape((-1,repeat)).mean(axis = 1)
        
cumul_true_flat = [t for ts in cumul_true for t in ts]
cumul_pred_flat = [t for ts in cumul_pred for t in ts]

cumul_true_self_flat = [t for ts in cumul_self_true for t in ts]
cumul_pred_self_flat = [t for ts in cumul_self_pred for t in ts]

_, _, fscore,_ = sklearn.metrics.precision_recall_fscore_support(cumul_true_flat, cumul_pred_flat)
_, _, fscore_self,_ = sklearn.metrics.precision_recall_fscore_support(cumul_true_self_flat, cumul_pred_self_flat)

print("Mean accuracy over 32 days", metrics.accuracy_score(cumul_true_flat, cumul_pred_flat))

#cm, fig, ax = plot_builder.plot_confusion_matrix(cumul_true_flat, cumul_pred_flat)
#plt.show()
#sys.exit(1)

f1_diff = fscore-fscore_self
sorted_indice = np.argsort(f1_diff)
sorted_class = clf_past.classes_[sorted_indice]
sorted_class= [str(c) for c in sorted_class]
f1_diff_sorted = f1_diff[sorted_indice]


print("Worst classes :", sorted_class[:5], "F1 scores", fscore[sorted_indice][:5])


json_data = dict(f1_diff=f1_diff.tolist(), f1_diff_sorted=f1_diff_sorted.tolist(), sorted_class=sorted_class)
json_str = json.dumps(json_data)
with open('plots/aging_per_class.json', 'w') as f:
    f.write(json_str)
