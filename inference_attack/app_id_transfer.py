import random

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

import lib.build_datasets as build_datasets
import lib.defenses as defenses
import lib.plot_builder as plot_builder
from constants import DATASET_DIR

REBUILD = True
CACHE_FILE = 'app_id_transfer_wear_os'
random.seed(0)


LOW_CONFIDENCE_APPS = ['ASB','Alarm','AthkarOfPrayer','Battery','DailyTracking','DuaKhatqmAlQuran','Flashlight','GooglePay','HealthyRecipes','HeartRate','Medisafe','NoApp','Phone','PlayMusic','Reminders','Sleep','WearCasts','Workout', 'SalatTime']

# ===============================
# Load or rebuild datasets

events = build_datasets.load_cache(CACHE_FILE)
if REBUILD or events is None: 
    
    def get_device(d):
        device = d.replace(DATASET_DIR, '')
        return device[:device.find('_')]

    def get_app(s):
        app = s.replace(DATASET_DIR, '')
        app = app[app.find('_')+1:]
        app = app[:app.find('_')]
        return app

    sources = build_datasets.find_sources(DATASET_DIR)
    random.shuffle(sources, random=lambda: 0) #deterministic shuffle for consistent results

    sources_open = [s for s in sources if "_Open_" in s]
    sources1 = [s for s in sources_open if get_device(s) == "HuaweiWatch2" and "_6" in s]
    sources2 = [s for s in sources_open if get_device(s) == "FossilExploristHR" and "_6" in s]

    apps1 = set([get_app(s) for s in sources1])
    apps2 = set([get_app(s) for s in sources2])
    common_apps = [s for s in apps1 if s in apps2]
    print("Identified", len(common_apps), "apps out of", len(apps1), "and", len(apps2))
    print(", ".join(["\\app{"+x+"}" for x in sorted(list(common_apps))]))

    sources1 = [s for s in sources1 if get_app(s) in common_apps]
    sources2 = [s for s in sources2 if get_app(s) in common_apps]

    events1, counts1 = build_datasets.cut_all_datasets_in_events(sources1, folder=DATASET_DIR)
    events2, counts2 = build_datasets.cut_all_datasets_in_events(sources2, folder=DATASET_DIR)

    events = [events1, events2]
    build_datasets.cache(CACHE_FILE, events)

[events1, events2] = events

# ===============================
# Functions

def extract_features(xy, unique_from=46, unique_to=1006,
                     unique_granularity=1, unique_deltas=[1005, 46, 0], to_withdraw=[]): # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size

    xs = xy['xs']
    ys = xy['ys']
    f = dict()
    
    bins = np.arange(0, 1000, step = unique_granularity)

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
    stats("non_null", [abs(y) for y in ys if abs(y) < unique_from])
    stats("outgoing", [abs(y) for y in ys if y > unique_from])
    stats("incoming", [abs(y) for y in ys if y < -unique_from])

    # unique packet lengths [Liberatore and Levine; Herrmann et al.]
    lengths = dict()
    for i in range(unique_from, unique_to):
        lengths[str(i)] = 0
    for y in ys:
        if str(abs(y)) in lengths:
            lengths[str(abs(y))] += 1

    lengths_array = list(lengths.values())
    stats("unique_lengths", lengths_array)

    # global stats about len
    for l in lengths:
        f['unique_lengths_'+str(l)] = extract_bins(lengths[l])


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
    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                for event in events[device][app][action]:
                    features_dict = dict()
                    if defense is None:
                        features_dict = extract_features(event)
                    else:
                        event_defended, cost = defenses.apply_defense(event, defense)
                        features_dict = extract_features(event_defended)
                    features = list(features_dict.values())
                    features_names = list(features_dict.keys())
                    
                    data.append(features)
                    labels.append(app)

    return [data, labels, features_names]


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


# ===============================
# Entrypoint

#for e in events1['HuaweiWatch2']:
#    print(e, len(events1['HuaweiWatch2'][e]['Open']), len(events2['FossilExploristHR'][e]['Open']))


if True:   
    X_train, y_train, feature_names1 = build_features_labels_dataset(events1)
    X_test, y_test, feature_names2 = build_features_labels_dataset(events2)

    score, features, y_pred, predicted_probas  = random_forest(feature_names1, X_train, y_train, X_test, y_test, n_trees=1000, rfe_nfeatures=800, rfe_steps=10)
    print(score)
    output = "app-id-transfer1-cm"
    plot_builder.confusion_matrix(output, y_test, y_pred, "", folder=DATASET_DIR)
    #plot_builder.rocauc(output.replace("confusion-matrix", "auc"), y_test, predicted_probas, folder=DATASET_DIR)

if True:
    X_train, y_train, feature_names1 = build_features_labels_dataset(events2)
    X_test, y_test, feature_names2 = build_features_labels_dataset(events1)
    
    score, features, y_pred, predicted_probas  = random_forest(feature_names1, X_train, y_train, X_test, y_test, n_trees=1000, rfe_nfeatures=800, rfe_steps=10)
    print(score)
    output = "app-id-transfer2-cm"
    plot_builder.confusion_matrix(output, y_test, y_pred, "", folder=DATASET_DIR)
    #plot_builder.rocauc(output.replace("confusion-matrix", "auc"), y_test, predicted_probas, folder=DATASET_DIR)
