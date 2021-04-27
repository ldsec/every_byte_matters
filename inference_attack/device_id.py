import pickle
import random
import tempfile

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

import lib.build_datasets as build_datasets
import lib.defenses as defenses
import lib.plot_builder as plot_builder
import lib.pretty_print as pp
from constants import DATASET_DIR, TEST_PERCENTAGE, N_FOLDS

REBUILD = True
CACHE_FILE = 'device_id_event_all_devices_non_mixed_eq'
DEVICES_TO_EXCLUDE = ['']

random.seed(0)

# ===============================
# Load or rebuild datasets

events = build_datasets.load_cache(CACHE_FILE)
if REBUILD or events is None: 
    sources = build_datasets.find_sources(DATASET_DIR)

    def exclude(d):
        device = d.replace(DATASET_DIR, '')
        device = device[:device.find('_')]
        return device not in DEVICES_TO_EXCLUDE

    all_sources_files = list(filter(exclude, sources))
    random.shuffle(all_sources_files, random=lambda: 0) #deterministic shuffle for consistent results


    # split BLE/Classic and Test/Train (1 file = around 20% test)
    build_datasets.find_common_columns(all_sources_files)
    sources_classic, sources_ble = build_datasets.split_classic_ble(all_sources_files, folder=DATASET_DIR)

    pp.new_table()
    pp.table_push("Total # of samples:", len(all_sources_files))
    pp.table_push("CLA Total # of samples:", len(sources_classic))
    pp.table_push("BLE Total # of samples:", len(sources_ble))
    pp.table_print()

    # actually parse the files
    events_classic, counts_classic = build_datasets.cut_all_datasets_in_events(sources_classic, folder=DATASET_DIR)
    events_ble, counts_ble = build_datasets.cut_all_datasets_in_events(sources_ble, folder=DATASET_DIR)

    events_classic = build_datasets.equilibrate_events_across_devices(events_classic)

    events = [events_classic, events_ble]
    build_datasets.cache(CACHE_FILE, events)

[events_classic, events_ble] = events

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

def build_features_labels_dataset(events, adversary_capture_duration=-1, defense=None):
    data = []
    labels = []
    features_names = []
    # shape: data is a [[Features], [Features], ...]
    cost_aggregator = defenses.CostAggregate()

    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                for event in events[device][app][action]:
                    features = dict()
                    cost = None

                    if defense is not None:
                        event_defended, cost = defenses.apply_defense(event, defense)
                        features_dict = extract_features(event_defended)
                        cost_aggregator.add_cost(cost, label=device)
                    else:
                        features_dict = extract_features(event)
                    features = list(features_dict.values())
                    features_names = list(features_dict.keys())

                    data.append(features)
                    labels.append(device)

    return [data, labels, features_names, cost_aggregator]

def get_model_size(rf):
    s = pickle.dumps(rf)
    return len(s)

    fo = tempfile.NamedTemporaryFile()	
    fname = fo	
    joblib.dump(clf, fname)
    size = os.path.getsize(fname) 
    fo.close()

    return size

# (subroutine) Fit Random Forest on X_train/y_train
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

def rf_folds(events, n_trees=100, adversary_capture_duration=-1, defense=None, rfe_nfeatures=10, rfe_steps=10):

    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names, cost = build_features_labels_dataset(events, adversary_capture_duration, defense=defense)

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
        
        s, features, y_pred, predicted_probas = random_forest(feature_names, X_train, y_train, X_test, y_test, n_trees, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)

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


def confusion_matrix(events, n_trees=100, output="confusion-matrix.png", rfe_nfeatures=10, rfe_steps=10, adversary_capture_duration=-1):
    scores, _, y_test_all, y_pred_all, _ = rf_folds(events, n_trees=n_trees, adversary_capture_duration=adversary_capture_duration, defense=None, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)
        
    plot_builder.print_classifier_score(scores)
    plot_builder.confusion_matrix(output, y_test_all, y_pred_all, "", folder=DATASET_DIR)
    #plot_builder.rocauc(output.replace("confusion-matrix", "auc"), y_test_all, pred_probas, folder=DATASET_DIR)

def rank_features(events, n_trees=100, output="confusion-matrix.png", rfe_nfeatures=10, rfe_steps=10, adversary_capture_duration=-1):
    _, features, _, _, _ = rf_folds(events, n_trees=n_trees, adversary_capture_duration=adversary_capture_duration, defense=None, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)
    plot_builder.feature_importance(output, features, folder=DATASET_DIR)


def find_best_kf_params(events, rfe_nfeatures=10, rfe_steps=10):
    n, result = [], []
    
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names, _ = build_features_labels_dataset(events)

    X = np.array(X)
    y = np.array(y)

    for n_trees in [1,2,5,10,20,30,40,50,60,70,80,90,100]:
        i = 0
        scores = []
        for train_index, test_index in sss.split(X, y):
            print("Fold", i, end="\r")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            s, _, _, _ = random_forest(feature_names, X_train, y_train, X_test, y_test, n_trees, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)

            scores.append(s)
            i += 1

        # average scores
        score = {k: np.mean([value[k] for value in scores]) for k in scores[0]}

        n.append(n_trees)
        result.append(score)
        print("N:", n_trees, "Score:", score)

    return n, [r['accuracy'] for r in result]


def find_best_rfe_params(events, n_trees=10, rfe_steps=1):
    n, result = [], []
    
    sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
    X, y, feature_names, _ = build_features_labels_dataset(events)

    X = np.array(X)
    y = np.array(y)

    for rfe_nfeatures in [1,2,5,10,15,20,25,30]:
        print("RFE", rfe_nfeatures)
        i = 0
        scores = []
        for train_index, test_index in sss.split(X, y):
            print("Fold", i, end="\r")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            s, _, _, _ = random_forest(feature_names, X_train, y_train, X_test, y_test, n_trees=n_trees, rfe_nfeatures=rfe_nfeatures, rfe_steps=rfe_steps)

            scores.append(s)
            i += 1

        # average scores
        score = {k: np.mean([value[k] for value in scores]) for k in scores[0]}

        n.append(rfe_nfeatures)
        result.append(score)
        print("N:", rfe_nfeatures, "Score:", score)

    return n, [r['accuracy'] for r in result]


def capture_duration(events_classic, events_ble, n_trees=100):

    # Get the mean accuracy with RF
    def capture_duration_routine(events, adversary_capture_duration=-1):
        sss = StratifiedShuffleSplit(n_splits=N_FOLDS, test_size=TEST_PERCENTAGE, random_state=0)
        X, y, _, _ = build_features_labels_dataset(events, adversary_capture_duration)
        clf=RandomForestClassifier(n_jobs=8, n_estimators=n_trees, random_state=0)
        scores_shuffle = cross_val_score(clf, X, y, cv=sss)

        return scores_shuffle.mean(), scores_shuffle.std()

    xs = []
    ts_classic = []
    ts_classic_e = []
    ts_ble = []
    ts_ble_e = []

    durations = list(np.geomspace(0.1,30,20))
    for duration in tqdm(durations):        
        mean1, std1 = capture_duration_routine(events_classic, adversary_capture_duration=duration)
        mean2, std2 = capture_duration_routine(events_ble, adversary_capture_duration=duration)

        xs.append(duration)
        ts_classic.append(mean1)
        ts_classic_e.append(std1)
        ts_ble.append(mean2)
        ts_ble_e.append(std2)

    plot_builder.accuracy_over_time('device-id-capture-duration', xs, ts_classic, ts_classic_e, ts_ble, ts_ble_e, folder=DATASET_DIR)


def apply_all_defenses(events, output="defenses"):
    def_result = dict()
    defenses.build_size_distribution(events)
    for defense in [None, 'pad', 'delay_group', 'add_dummies']:
        print("Defense:", defense, end='\r')

        scores, features, y_test_all, y_pred_all, cost = rf_folds(events, defense=defense, n_trees=10, rfe_nfeatures=10, rfe_steps=10)
        name = defense
        if name is None:
            name = "none"
        plot_builder.confusion_matrix(output+"-"+name+"-cm", y_test_all, y_pred_all, "", folder=DATASET_DIR)
        plot_builder.feature_importance(output+"-"+name+"-fi", features, folder=DATASET_DIR)

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
    print("-------- Classic --------")
    confusion_matrix(events_classic, rfe_nfeatures=10, rfe_steps=1, output="device-id-cla-cm")
    print("-------- BLE --------")
    confusion_matrix(events_ble, rfe_nfeatures=10, rfe_steps=1, output="device-id-ble-cm")

if False:
    # Runs RFE slowly to rank features
    print("*" * 20)
    print("Ranking features....")
    rank_features(events_classic, rfe_nfeatures=10, rfe_steps=10, output="device-id-cla-fi")
    rank_features(events_ble, rfe_nfeatures=10, rfe_steps=10, output="device-id-ble-fi")

if False:
    # Vary number of features kept by RFE to see how accuracy varies
    print("*" * 20)
    print("Varying kF parameters....")
    X, Y_cla = find_best_kf_params(events_classic)
    X, Y_ble = find_best_kf_params(events_ble)

    plot_builder.accuracy_vs_nsamples("device-id-ntrees", X, Y_cla, Y_ble, folder=DATASET_DIR)

if False:
    # Vary number of features kept by RFE to see how accuracy varies
    print("*" * 20)
    print("Varying RFE parameters....")
    X, Y_cla = find_best_rfe_params(events_classic)
    X, Y_ble = find_best_rfe_params(events_ble)

    plot_builder.accuracy_vs_nsamples("device-id-nfeatures", X, Y_cla, Y_ble, folder=DATASET_DIR)

if False:
    print("*" * 20)
    print("Applying all defenses....")
    print("-------- Classic --------")
    apply_all_defenses(events_classic, output="device-id-def-cla")
    print("-------- BLE --------")
    apply_all_defenses(events_ble, output="device-id-def-ble")

if False:
    print("*" * 20)
    print("Varying FRONT parameters....")
    print("-------- Classic --------")
    find_best_front_params(events_classic)
    print("-------- BLE --------")
    find_best_front_params(events_ble)