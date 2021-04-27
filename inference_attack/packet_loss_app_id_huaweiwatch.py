import copy
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
from lib.defenses import CostAggregate

REBUILD = False
CACHE_FILE = 'packet_loss'
DEVICES_TO_INCLUDE = ['HuaweiWatch2']

random.seed(0)

# ===============================
# Load or rebuild datasets

def filter_by_length(events, minimum_payload=500, ratio_app_not_satisfing_minimum_payload_length=0.25, printInfo=False, keep_low_volume=False):
    results = copy.deepcopy(events)
    for watch in events:
        for app in events[watch]:
            for action in events[watch][app]:
                total_event = len(events[watch][app][action])
                below_minimum_payload = 0
                for sample in events[watch][app][action]:

                    payload_length = sum([abs(s) for s in sample["ys"]])
                    if payload_length < minimum_payload:
                        below_minimum_payload += 1

                ratio_below = below_minimum_payload / total_event
                if not keep_low_volume and ratio_below > ratio_app_not_satisfing_minimum_payload_length:
                    if printInfo:
                        print("total_event: ", total_event, " - below threshold: ", below_minimum_payload)
                        print(app + "_" + action + " removed")
                        print(" ratio_below = ", ratio_below)
                    del results[watch][app][action]
                if keep_low_volume and ratio_below < ratio_app_not_satisfing_minimum_payload_length:
                    if printInfo:
                        print("total_event: ", total_event, " - below threshold: ", below_minimum_payload)
                        print(app + "_" + action + " removed")
                        print(" ratio_below = ", ratio_below)
                    del results[watch][app][action]

            if len(results[watch][app]) == 0:
                del results[watch][app]
        
        if len(results[watch]) == 0:
            del results[watch]

    return results

data = build_datasets.load_cache(CACHE_FILE)
if REBUILD or data is None: 
    sources = build_datasets.find_sources(DATASET_DIR)

    def include(d):
        device = d.replace(DATASET_DIR, '')
        device = device[:device.find('_')]
        return ("_Open_" in d) and device in DEVICES_TO_INCLUDE

    all_sources_files = list(filter(include, sources))
    random.shuffle(all_sources_files, random=lambda: 0) #deterministic shuffle for consistent results
    build_datasets.find_common_columns(all_sources_files)

    events, counts = build_datasets.cut_all_datasets_in_events(all_sources_files, folder=DATASET_DIR)
    events = build_datasets.roughly_equilibrate_events_across_action(events)

    events_low_volume = filter_by_length(events, keep_low_volume=True)
    events_high_volume = filter_by_length(events, keep_low_volume=False)

    if 'SalatTime' in events_low_volume['HuaweiWatch2']:
        events_high_volume['HuaweiWatch2']['SalatTime'] = events_low_volume['HuaweiWatch2']['SalatTime']
        del events_low_volume['HuaweiWatch2']['SalatTime']


    data = [events, events_low_volume, events_high_volume]
    
    build_datasets.cache(CACHE_FILE, data)
else:
    print("Loaded cache file")

[events, events_low_volume, events_high_volume] = data

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



# ===============================
# Entrypoint

def rf_folds(events, defense=None, rfe_nfeatures=50, rfe_steps=100, n_trees=30):
    
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names, cost = build_features_labels_dataset(events, defense=defense)

    #print("Number of classes", len(set(y)))
    #print("Number of features", len(X[0]))

    X = np.array(X)
    y = np.array(y)

    scores = []
    y_test_all = []
    y_pred_all = []
    feature_ranks = dict()

    i = 0
    for train_index, test_index in sss.split(X, y):
        print("Fold", i, end="\r")

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
    score = {k: np.mean([value[k] for value in scores]) for k in scores[0]}

    # average features importance
    for f in feature_ranks:
        if len(feature_ranks[f]) < N_FOLDS:
            feature_ranks[f].extend([0] * (N_FOLDS - len(feature_ranks[f])))

    features_and_percentages = []
    for f in feature_ranks:
        features_and_percentages.append((f, np.mean(feature_ranks[f]), np.std(feature_ranks[f])))

    return score, features_and_percentages, y_test_all, y_pred_all, cost


def confusion_matrix(events, output, n_trees=30, defense=None, rfe_nfeatures=100, rfe_steps=100):
    scores, _, y_test_all, y_pred_all, _ = rf_folds(events, n_trees=n_trees, defense=None, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)
    
    return scores

    #plot_builder.print_classifier_score(scores)
    #plot_builder.confusion_matrix(output, y_test_all, y_pred_all, "", folder=DATASET_DIR)
    #plot_builder.rocauc(output.replace("confusion-matrix", "auc"), y_test_all, pred_probas, folder=DATASET_DIR)

def rank_features(events, n_trees=30, output="confusion-matrix.png", rfe_nfeatures=100, rfe_steps=100):
    _, features, _, _, _ = rf_folds(events, n_trees=n_trees, defense=None, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)
    plot_builder.feature_importance(output, features, folder=DATASET_DIR)

def apply_all_defenses(events, output="defense"):
    def_result = dict()

    defenses.build_size_distribution(events)
    for defense in [None, 'pad', 'delay_group', 'add_dummies']:
        print("Defense:", defense, end='\r')

        scores, features, y_test_all, y_pred_all, cost = rf_folds(events, defense=defense, n_trees=30, rfe_nfeatures=10, rfe_steps=100)
        name = defense
        if name is None:
            name = "none"
        plot_builder.confusion_matrix("app-id-huaweiwatch-def-"+name+"-cm", y_test_all, y_pred_all, "", folder=DATASET_DIR)
        plot_builder.feature_importance("app-id-huaweiwatch-def-"+name+"-fi", features, folder=DATASET_DIR)

        print("Defense:", defense, scores['accuracy'], cost)
        def_result[defense] = dict(scores=scores, costs=cost)

    plot_builder.defense_plot(output, def_result, folder=DATASET_DIR)

def find_best_front_params(events):
    Ws = list(range(0,30))
    dummies_s = [1, 10, 100, 300, 1000]

    scores, _, _, _, cost = rf_folds(events)
    print("['app-id', 'None',", scores['accuracy'], ",'", cost, "'],")

    defenses.build_size_distribution(events)

    def r(i):
        return str(round(i, 2))

    for W in Ws:
        for n_dummies in dummies_s:
            defenses.front_params = [W, W+1, n_dummies, n_dummies+100]
            
            scores, _, _, _, cost = rf_folds(events, defense='add_dummies')
            print("['app-id', 'Front',", W, ",", n_dummies, ",", r(scores['accuracy']), ",'", cost, "'],")


def apply_uniform_packet_loss(events, uniform_packet_loss = 0.0):

    events_copy = dict()
    events_copy['HuaweiWatch2'] = dict()

    for app in events['HuaweiWatch2']:
        events_copy['HuaweiWatch2'][app] = dict()
        events_copy['HuaweiWatch2'][app]['Open'] = []

        for sample in events['HuaweiWatch2'][app]['Open']:

            filtered_sample = dict(xs=[], ys=[], source=sample['source'])

            i = 0
            while i<len(sample['xs']):
                if random.uniform(0, 1) >= uniform_packet_loss:
                    filtered_sample['xs'].append(sample['xs'][i])
                    filtered_sample['ys'].append(sample['ys'][i])
                i += 1

            events_copy['HuaweiWatch2'][app]['Open'].append(filtered_sample)

    return events_copy

# ===============================
# Entrypoint

def r(i):
    return str(round(i, 2))
        
for loss in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for repeat in range(1, 10):

        events_loss1 = apply_uniform_packet_loss(events, uniform_packet_loss=loss)
        events_loss2 = apply_uniform_packet_loss(events_low_volume, uniform_packet_loss=loss)
        events_loss3 = apply_uniform_packet_loss(events_high_volume, uniform_packet_loss=loss)
        acc1 = confusion_matrix(events_loss1, output="")
        acc2 = confusion_matrix(events_loss2, output="")
        acc3 = confusion_matrix(events_loss3, output="")

        s1 = ",".join([k+": "+r(acc1[k]) for k in acc1])
        s2 = ",".join([k+"_low: "+r(acc2[k]) for k in acc2])
        s3 = ",".join([k+"_high: "+r(acc3[k]) for k in acc3])

        print("{loss:", loss, ",repeat:", repeat, ",", s1, s2, s3, "}", end='\n')
        # todo: automatic plotting