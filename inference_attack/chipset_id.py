import random
import sys

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit

import lib.build_datasets as build_datasets
import lib.defenses as defenses
import lib.plot_builder as plot_builder
from constants import DATASET_DIR, TEST_PERCENTAGE, N_FOLDS

REBUILD = True
CACHE_FILE = 'chipset_id_eq'
DEVICES_TO_EXCLUDE = [''] 

random.seed(0)

CHIPSET_MAP = dict()
CHIPSET_MAP['SamsungGalaxyWatch'] = 'Broadcomm'
CHIPSET_MAP['FossilExploristHR'] = 'Qualcomm'
CHIPSET_MAP['AppleWatch'] = 'Apple'
CHIPSET_MAP['HuaweiWatch2'] = 'Broadcomm'
CHIPSET_MAP['FitbitVersa2'] = 'Cypress'
CHIPSET_MAP['Airpods'] = 'Apple'
CHIPSET_MAP['MDR'] = 'Qualcomm'

CHIPSET_MAP['BeurerAS80'] = 'TexasInstruments'
CHIPSET_MAP['FitbitCharge2'] = 'MicroElectronics'
CHIPSET_MAP['FitbitCharge3'] = 'Cypress'
CHIPSET_MAP['H2BP'] = 'Nordic'
CHIPSET_MAP['HuaweiBand3'] = 'RivieraWaves'
CHIPSET_MAP['MiBand2'] = 'Dialog'
CHIPSET_MAP['MiBand3'] = 'Dialog'
CHIPSET_MAP['MiBand4'] = 'Dialog'
CHIPSET_MAP['PanoBike'] = 'TexasInstruments'
CHIPSET_MAP['Qardio'] = 'Qualcomm'
CHIPSET_MAP['SW170'] = 'Nordic'

def get_chipset(device_name):
    global CHIPSET_MAP
    if not device_name in CHIPSET_MAP:
        print("Error, device name", device_name, "is not in CHIPSET_MAP")
        sys.exit(1)
    return CHIPSET_MAP[device_name]

# ===============================
# Load or rebuild datasets

events_chipset = build_datasets.load_cache(CACHE_FILE)
if REBUILD or events_chipset is None: 
    sources = build_datasets.find_sources(DATASET_DIR)

    def exclude(d):
        device = d.replace('data/', '')
        device = device[:device.find('_')]
        return device not in DEVICES_TO_EXCLUDE

    all_sources_files = list(filter(exclude, sources))
    random.shuffle(all_sources_files, random=lambda: 0) #deterministic shuffle for consistent results

    build_datasets.rebuild_all_datasets(sources_files=all_sources_files, force_rebuild=False)

    print("Total # of samples:", len(all_sources_files))

    events, counts = build_datasets.cut_all_datasets_in_events(all_sources_files, folder=DATASET_DIR)
    events_eq = build_datasets.equilibrate_events_across_chipset(events, CHIPSET_MAP)

    build_datasets.cache(CACHE_FILE, events_eq)
    events_chipset = events_eq

# ===============================
# Functions

def extract_features(xy, adversary_capture_duration=-1): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size
    xs = []
    ys = []
    i = 0
    while i < len(xy['ys']) and (adversary_capture_duration == -1 or xy['xs'][i]<adversary_capture_duration):
        xs.append(xy['xs'][i])
        ys.append(xy['ys'][i])
        i += 1

    f = dict()

    def deltas(serie):
        out = []
        i = 1
        while i<len(serie):
            out.append(serie[i]-serie[i-1])
            i += 1
        return out
    
    def take(arr, n=30):
        if len(arr) > n:
            return arr[:30]
        return arr

    def sum(arr):
        x = 0
        for v in arr:
            x += v
        return x

    def stats(key, data):
        if len(data) == 0:
            data=[-1]
        f['min_'+key] = np.min(data)
        f['mean_'+key] = np.mean(data)
        f['max_'+key] = np.max(data)
        f['count_'+key] = len(data)
        f['std_'+key] = np.std(data)

    def avgIPT(xs, ys, incoming=True):
        def condition(y):
            if incoming and y>=0:
                return True
            if not incoming and y<0:
                return True
            return False
        zipped = [xy for xy in zip(xs, ys) if condition(xy[1])]
        P = len(zipped)

        if P == 1:
            return zipped[0][1]
        nom = sum(deltas([xy[0] for xy in zipped]))

        return nom/(P-1)

    f['avgipt_incoming'] = avgIPT(xs, ys, True)
    f['avgipt_outgoing'] = avgIPT(xs, ys, False)
    

    # statistics about timings
    x_deltas = []
    i = 1
    while i<len(xs):
        x_deltas.append(xs[i]-xs[i-1])
        i += 1

    stats("sizes_non_null", [abs(y) for y in ys if y != 0])
    stats("sizes_outgoing", [abs(y) for y in ys if y > 0])
    stats("sizes_incoming", [abs(y) for y in ys if y < 0])
    stats("time_deltas", x_deltas)
    
    # unique packet lengths [Liberatore and Levine; Herrmann et al.]
    lengths = dict()
    for i in range(1,10):
        lengths[str(i)] = 0

    for y in ys:
        i = 0
        y = abs(y)
        i = y // 10
        if i > 10:
            i == 10
        if str(i) in lengths:
            lengths[str(i)] += 1

    for k in lengths:
        f["count_sizes_"+str(10*int(k))+"_"+str(10*(int(k)+1)-1)] = lengths[k]

    return f


def build_features_labels_dataset(events, defense=None):
    data = []
    labels = []
    features_names = []
    # shape: data is a [[Features], [Features], ...]
    for chipset in events:
        for device in events[chipset]:
            for app in events[chipset][device]:
                for action in events[chipset][device][app]:
                    for event in events[chipset][device][app][action]:
                        event_defended, cost = defenses.apply_defense(event, defense)
                        features_dict = extract_features(event_defended)
                        features = list(features_dict.values())
                        features_names = list(features_dict.keys())

                        data.append(features)
                        labels.append(chipset)

    return [data, labels, features_names]

# (subroutine) Fit Random Forest on X_train/y_train
def random_forest(features_names, X_train, y_train, X_test, y_test, seed=0, n_trees=10, rfe_nfeatures=3, rfe_steps=10):
    clf=RandomForestClassifier(n_jobs=8, n_estimators=n_trees, random_state=seed)
    selector = RFE(estimator=clf, n_features_to_select=rfe_nfeatures, step=rfe_steps)
    selector = selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    
    
    clf2=RandomForestClassifier(n_jobs=8, n_estimators=n_trees, random_state=seed)
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

# Plot Confusion Matrix
def confusion_matrix(events, seed=0, n_trees=10, output="confusion-matrix.png", rfe_nfeatures=10, rfe_steps=1, adversary_capture_duration=-1):
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names = build_features_labels_dataset(events, defense=None)

    X = np.array(X)
    y = np.array(y)

    accuracies = []
    y_test_all = []
    y_pred_all = []
    pred_probas = []

    i = 0
    for train_index, test_index in sss.split(X, y):
        print("Fold", i, end="\r")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        acc, features, y_pred, predicted_probas  = random_forest(feature_names, X_train, y_train, X_test, y_test, seed, n_trees, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)

        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred.tolist())
        pred_probas.extend(predicted_probas)
        accuracies.append(acc)

        #print("Fold", i, "accuracy:", acc)
        i += 1

    print(np.mean([x['accuracy'] for x in accuracies]))

    plot_builder.confusion_matrix(output, y_test_all, y_pred_all, "", folder=DATASET_DIR)
    #plot_builder.rocauc(output.replace("confusion-matrix", "auc"), y_test_all, pred_probas, folder=DATASET_DIR)

def rf_accuracy(events, seed=0, n_trees=10, defense=None, rfe_nfeatures=10, rfe_steps=10):

    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names = build_features_labels_dataset(events, defense=defense)

    X = np.array(X)
    y = np.array(y)

    accuracies = []
    y_test_all = []
    y_pred_all = []

    i = 0
    for train_index, test_index in sss.split(X, y):
        print("Fold", i, end="\r")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        acc, features, y_pred, predicted_probas = random_forest(feature_names, X_train, y_train, X_test, y_test, seed, n_trees, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)

        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred.tolist())
        accuracies.append(acc)
        i += 1

    mean_accuracy = np.mean([x['accuracy'] for x in accuracies])

    return mean_accuracy


def rank_features(events, output="feature-importance.png", seed=0, n_trees=10, rfe_nfeatures=10, rfe_steps=1, adversary_capture_duration=-1):
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names = build_features_labels_dataset(events, defense=None)

    X = np.array(X)
    y = np.array(y)

    feature_ranks = dict()
    i = 0
    for train_index, test_index in sss.split(X, y):
        print("Fold", i, end="\r")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        acc, features, y_pred, predicted_probas = random_forest(feature_names, X_train, y_train, X_test, y_test, seed, n_trees, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)

        for proba, feature in features:
            if not feature in feature_ranks:
                feature_ranks[feature] = []

            feature_ranks[feature].append(proba)

        i += 1

    for f in feature_ranks:
        if len(feature_ranks[f]) < N_FOLDS:
            feature_ranks[f].extend([0] * (N_FOLDS - len(feature_ranks[f])))

    features_and_percentages = []
    for f in feature_ranks:
        features_and_percentages.append((f, np.mean(feature_ranks[f]), np.std(feature_ranks[f])))

    print(features_and_percentages)

    plot_builder.feature_importance(output, features_and_percentages, folder=DATASET_DIR)



# ===============================
# Entrypoint

if True:
    confusion_matrix(events_chipset, output="chipset-id-cm")

if False:
    rank_features(events_chipset, output="chipset-id-fi")
