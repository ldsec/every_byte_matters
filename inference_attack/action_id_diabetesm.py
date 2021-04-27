import random

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

REBUILD = True
CACHE_FILE = 'app_id_event_all_devices_non_mixed_eq'

random.seed(0)

# ===============================
# Load or rebuild datasets

events = build_datasets.load_cache(CACHE_FILE)
if REBUILD or events is None: 
    sources = build_datasets.find_sources(DATASET_DIR)

    def include(d):
        return "_DiabetesM_" in d and 'HuaweiWatch2' in d and not "_Open_" in d and not "_Close_" in d

    all_sources_files = list(filter(include, sources))
    random.shuffle(all_sources_files, random=lambda: 0) #deterministic shuffle for consistent results

    events, counts = build_datasets.cut_all_datasets_in_events(all_sources_files, folder=DATASET_DIR)

    if False:
        for e in events['HuaweiWatch2']['DiabetesM']:
            print(e, len(events['HuaweiWatch2']['DiabetesM'][e]))
            while len(events['HuaweiWatch2']['DiabetesM'][e]) > 20:
                del events['HuaweiWatch2']['DiabetesM'][e][0]
            print(e, len(events['HuaweiWatch2']['DiabetesM'][e]))


    n_actions = 0
    for dev in events:
        for app in events[dev]:
            for act in events[dev][app]:
                print(dev, app, act, len(events[dev][app][act]))
                n_actions += 1

    print("Total:", n_actions, "actions")

    build_datasets.cache(CACHE_FILE, events)

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
    stats("count_sizes", lengths_array)

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
        

    return f

    

def build_features_labels_dataset(events, defense=None):
    data = []
    labels = []
    features_names = []
    cost_aggregator = CostAggregate()

    # shape: data is a [[Features], [Features], ...]
    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                for event in events[device][app][action]:
                    features_dict = dict()
                    cost = None

                    if defense is not None:
                        event_defended, cost = defenses.apply_defense(event, defense)
                        features_dict = extract_features(event_defended)
                        cost_aggregator.add_cost(cost, label=app+"_"+action)

                    else:
                        features_dict = extract_features(event)

                    features = list(features_dict.values())
                    features_names = list(features_dict.keys())
                    
                    data.append(features)
                    labels.append(app+"_"+action)

    return [data, labels, features_names, cost_aggregator]


# Fit Random Forest
def random_forest(features_names, X_train, y_train, X_test, y_test, n_trees=100, rfe_nfeatures=3, rfe_steps=10):
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


def rf_folds(events, defense=None, rfe_nfeatures=50, rfe_steps=100, n_trees=100):
    
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


def confusion_matrix(all_events, output, n_trees=30, defense=None, rfe_nfeatures=50, rfe_steps=100):
    scores, _, y_test_all, y_pred_all, _ = rf_folds(events, n_trees=n_trees, defense=None, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)
        
    plot_builder.print_classifier_score(scores)
    plot_builder.confusion_matrix(output, y_test_all, y_pred_all, "", folder=DATASET_DIR)
    #plot_builder.rocauc(output.replace("confusion-matrix", "auc"), y_test_all, pred_probas, folder=DATASET_DIR)

def rank_features(events, n_trees=30, output="confusion-matrix.png", rfe_nfeatures=10, rfe_steps=100):
    _, features, _, _, _ = rf_folds(events, n_trees=n_trees, defense=None, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)
    plot_builder.feature_importance(output, features, folder=DATASET_DIR)


def apply_all_defenses(events, output="defense"):
    def_result = dict()
    defenses.build_size_distribution(events)
    for defense in [None, 'pad', 'delay_group', 'add_dummies']:
        print("Defense:", defense, end='\r')
        
        scores, features, y_test_all, y_pred_all, cost = rf_folds(events, defense=defense, n_trees=30, rfe_nfeatures=50, rfe_steps=100)
        name = defense
        if name is None:
            name = "none"
        plot_builder.confusion_matrix("action-id-diabetesm-def-"+name+"-cm", y_test_all, y_pred_all, "", folder=DATASET_DIR)
        plot_builder.feature_importance("action-id-diabetesm-def-"+name+"-fi", features, folder=DATASET_DIR)

        print("Defense:", defense, scores['accuracy'], cost)
        def_result[defense] = dict(scores=scores, costs=cost)

    plot_builder.defense_plot(output, def_result, folder=DATASET_DIR)

def find_best_front_params(events):

    defenses.build_size_distribution(events)
    
    for W in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        defenses.front_params = [W+4, W+5, 1, 100]

        print(defenses.front_params)
        scores, _, _, _, cost = rf_folds(events, defense='add_dummies')
        print("W", W, scores['accuracy'], cost)


# ===============================
# Entrypoint

if True:
    # Simply runs the best attack and builds confusion matrices
    confusion_matrix(events, output="action-id-diabetesm-cm")

if False:
    # Runs RFE slowly to rank features
    print("*" * 20)
    print("Ranking features....")
    rank_features(events, output="action-id-diabetesm-fi")

if False:
    print("*" * 20)
    print("Applying all defenses....")
    apply_all_defenses(events, output="action-id-diabetesm-def")

if True:
    print("*" * 20)
    print("Varying FRONT parameters....")
    find_best_front_params(events)