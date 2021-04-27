import csv
import glob
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import lib.build_datasets as build_datasets
import lib.defenses as defenses
import lib.pretty_print as pp
from constants import DATASET_DIR

DATASET_LONGRUN_DIR = '/home/jwhite/d/git/wearables_dataset/data/'
DATASET_LONGRUN_DIR = '/home/jwhite/d/git/wearables_dataset/data_aux/24h-captures/'
DATASET_LONGRUN_HUMAN_PATTERNS_DIRS = [DATASET_LONGRUN_DIR + "user-interaction-pattern-" + str(i)+"/" for i in range(1,4)]
GROUND_TRUTH_PATH_EXTENTION="ground-truth/" # subfolder name, relative to longrun folder, where the ground truth is to be found
NOISE_LABEL = 'NoApp_NoAction'

# Discard:  'FitBreathe_open','FitWorkout_open','Fit_open'.
# Because got updated and does not send data anymore (Fit).
DISCARDED_ACTION = [ 'FitBreathe_Open', 'FitWorkout_Open', 'Fit_Open', 'HealthyRecipes_Open']

LOWER = 0
UPPER = 18.5

CACHE_FILE='tmp'
CACHE_FILE2='tmp2'

REBUILD=True

POSSIBLE_MASTERS = ["Pixel 2", "Nexus 5", "Ludovic.s iPhone"]
master = None

def extract_master(comm):
    return comm[comm.find('"') +1 : comm.find('"', comm.find('"')+1)]

# ================================================================

def plot_prec_recall_by_slot(all_correct_pred_dict, all_wrong_pred_dict, all_missed_gt_dict,
                             by='all', remove_force_stop=True, savefig=None, beta=1):

    tp, fp, fn = Counter(), Counter(), Counter()

    def apps_count(preds):
        app_counter = Counter()
        for c in preds:
            for pred in preds[c]:
                pred_cop = pred[2] if type(pred[2]) == list else [pred[2]]
                app_counter[pred_cop[0]] += 1
        return app_counter


    if by == 'all':
        # Return precision. recall and f1 score
        for c in all_correct_pred_dict:
            tp['tp'] += len(all_correct_pred_dict[c])
            fp['fp'] += len(all_wrong_pred_dict[c])
            fn['fn'] += len(all_missed_gt_dict[c])

        print(tp)

        p = 1 if (tp['tp'] + fp['fp']) == 0 else tp['tp'] / (tp['tp'] + fp['fp'])
        #p = tp['tp'] / (tp['tp'] + fp['fp'])
        r = tp['tp'] / (tp['tp'] + fn['fn'])
        f1 = 0 if p + r == 0 else (1 + beta**2) * (p * r)/((beta**2 * p) + r)
        return p, r, f1

    tp = pd.Series(tp)
    fp = pd.Series(fp)
    fn = pd.Series(fn)


    tpfpfn = pd.DataFrame({'tp': tp, 'fp': fp, 'fn':fn}).fillna(0)

    precision = tpfpfn["tp"] / (tpfpfn["tp"] + tpfpfn["fp"])
    recall = tpfpfn["tp"] / (tpfpfn["tp"] + tpfpfn["fn"])


    prec_recall = pd.DataFrame({'recall': recall, 'precision': precision})

    if savefig:
        if by == 'category':
            by = "action types"

        ax = tpfpfn.plot(kind='bar', fontsize=15, figsize=(7,5))
        plt.ylabel("score", fontsize=16)
        plt.xlabel(by, fontsize=16)
        plt.title("tp fp fn for by " + by, fontsize = 16)
        plt.legend(fontsize=13)
        #plt.xticks(rotation=0)
        plt.savefig(savefig+'_1tpfpfn', dpi= 80)


        title="Recall and Precision Score by " + by
        ax = prec_recall.plot(kind='bar', fontsize=15, ylim=[0,1.02], figsize=(7,5))
        plt.ylabel("score", fontsize=16)
        plt.xlabel(by, fontsize=16)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=13, loc=(0.4,0.82)) #loc='upper center')
        #plt.xticks(rotation=0)
        plt.savefig(savefig+"_"+by+"_pr", dpi= 80)


    return prec_recall, tpfpfn


def compute_eval_metric(prediciton, ground_truth, clf, method="majority_voting",
                        top_n=3, match_method="first",
                        print_details=False, skipping_force_stop=False,
                        RETURN_PROB=False, target_apps_only=False,
                        threshold=None
                       ):
    """
    calculate accuarcy, precision and recall
    Args:
        prediction : [(start, stop, action),]. List of predicition with time boundaries
        ground_truth : [(start, stop, action),]. List of ground_turh associated with the prediction
        method : str. The method to use to classify the probability output.
                      The list of possible methods are the folowing:
                      'majority_voting': return the label with the most vote in the RF
                      'top_n_majority_voting': return the top_n
                      'threshold_majority_voting': same as majority_voting with threshold cutoff
        match_method : str. The matching method to be use if 2 or more prediction overlaps a correct ground truth.
                            The list of possible methods are the folowing:
                            'first' : The closest to the begining of the action
                            'overlap' : The one that overlap the most the ground-truth
        RETURN_PROB : bool. Return Probabilities vector instead of
        skipping_force_stop : bool. Remove all force stop from predicted and gt.
    Return (tp, fp, fn) True Positive, False Positive and False Negative
    """

    match_method = "best_match_" + match_method

    if skipping_force_stop and method=="top_n_majority_voting":
        print("WARNING: {} cannot be combined with skipping_force_stop=True".format(method))
        sys.exit(1)

    if method == "threshold_majority_voting" and threshold is None:
        print("WARNING: {} cannot be combined with threshold=None".format(method))
        sys.exit(1)

    tp, fp, fn = 0, 0, 0
    i_pred, i_gt = 0, 0
    correct_pred, wrong_pred, missed_gt, correct_gt = [], [], [], []
    last_run = False


    def add_wrong_pred(pred, fp):
        fp += 1
        wrong_pred.append(pred)
        if print_details:
            print(pred, " wrong prediction (fp + 1 = ", fp, ")")
        return fp

    def add_missed_gt(gt, fn):
        fn += 1
        missed_gt.append(gt)
        if print_details:
            print(gt, " missed prediction (fn + 1 = ", fn,")")
        return fn

    def add_correct_pred(pred, tp):
        tp += 1
        correct_pred.append(pred)
        if print_details:
            print(pred, " correct pred (tp + 1 = ", tp,")")
        return tp


    def majority_voting(pred_vec):
        return [clf.classes_[np.argmax(pred_vec)]]

    def threshold_majority_voting(pred_vec):
        if max(pred_vec) < threshold:
            return None
        return majority_voting(pred_vec)


    def top_n_majority_voting(pred_vec):
        top_args = np.argsort(-pred_vec)[:top_n].tolist()
        return [clf.classes_[top_arg] for top_arg in top_args]

    def remove_force_stop(prediction):
        prediciton_out = list()
        for pred in prediction:
            pred_cop = pred[2] if type(pred[2]) == list else [pred[2]]
            if any([True for x in pred_cop if "NoApp_NoAction" in x]):
                continue
            prediciton_out.append(pred)
        return prediciton_out

    def best_match_first(match):
        "Return a list sorted by best match according to the criteria on the function name"
        return match

    def best_match_overlap(match):
        "Return a list sorted by best match according to the criteria on the function name"
        return sorted(match,key=lambda item:-item[1])



    matched_pred = []
    for gt in ground_truth:
        t, action_gt = gt
        start_gt, real_gt, stop_gt = t
        if target_apps_only:
            action_gt = action_gt.split("_")[0]

        match = []
        for pred in prediciton:
            start_pred, stop_pred, pred_vec = pred
            action_pred = eval(method)(pred_vec)
            if action_pred is None:
                continue

            if overlaps((start_pred, stop_pred), (start_gt, stop_gt)) and any([True for x in action_pred if action_gt in x]):

                chosen_pred = (start_pred, stop_pred, action_pred)
                match.append((chosen_pred, overlap_length((start_pred, stop_pred), (start_gt, stop_gt))))

        if len(match) == 0:
            gt2 = [gt[0][0], gt[0][2], gt[1]]
            fn = add_missed_gt(gt2, fn)
            continue

        best_matchs = eval(match_method)(match)

        for bm in best_matchs:
            pred = bm[0]
            if not pred in correct_pred:
                tp = add_correct_pred(pred, tp)
                break

    # compute wrong pred:
    for pred in prediciton:
        start_pred, stop_pred, pred_vec = pred
        action_pred = eval(method)(pred_vec)

        # prediction skipped by the Decision Maker
        if action_pred is None:
            continue
        chosen_pred = (start_pred, stop_pred, action_pred)
        if chosen_pred not in correct_pred:
            fp = add_wrong_pred(chosen_pred, fp)

    if skipping_force_stop:
        correct_pred = remove_force_stop(correct_pred)
        wrong_pred = remove_force_stop(wrong_pred)
        missed_gt = remove_force_stop(missed_gt)

    return correct_pred, wrong_pred, missed_gt


def eval_all(predicted, content_ground_truth, clf, method='majority_voting', top_n=3,
             match_method='first', output_dict=False, dprint=False, target_apps_only=False,
             threshold=None, skipping_force_stop=False):

    all_correct_pred = []
    all_wrong_pred = []
    all_missed_gt = []

    all_correct_pred_dict = dict()
    all_wrong_pred_dict = dict()
    all_missed_gt_dict = dict()

    def dprint(s):
        if dprint:
            print(s)

    for capture in predicted:
        if capture == "longrun_uniform_20-04-20_11-28-32":
            continue
        (correct_pred, wrong_pred, missed_gt) = compute_eval_metric(predicted[capture],
                                                                    content_ground_truth[capture],
                                                                    clf,
                                                                    method=method,
                                                                    threshold=threshold,
                                                                    match_method=match_method,
                                                                    top_n=top_n,
                                                                    target_apps_only=target_apps_only,
                                                                    print_details=True, skipping_force_stop=skipping_force_stop)

        all_correct_pred_dict[capture] = correct_pred
        all_wrong_pred_dict[capture] = wrong_pred
        all_missed_gt_dict[capture] = missed_gt

        all_wrong_pred += wrong_pred
        all_correct_pred += correct_pred
        all_missed_gt += missed_gt

        dprint(capture)
        tp, fp, fn = len(correct_pred), len(wrong_pred), len(missed_gt)
        #if tp + fp != 0 or tp + fn !=0:
        precision = 1 if tp + fp == 0 else tp / (tp + fp)
        recall = 1 if tp + fn == 0 else tp / (tp + fn)

        dprint("precision = {}, recall = {}".format(precision, recall))
        dprint(" ")
    if output_dict:
        return all_correct_pred_dict, all_wrong_pred_dict, all_missed_gt_dict
    return all_correct_pred, all_wrong_pred, all_missed_gt

def find_critical_point(time_serie, window_size=20, min_space_between_critical_points=5,
                        minimum_window_payload=200):
    """
    Find Critcal points: where we have more than minimum_window_payload Bytes
    data sum over a period of window_size seconds with min_space_between_critical_points seconds inter space
    """
    ts = pd.Series(data = time_serie['ys'], index = pd.to_timedelta(time_serie["xs"], 'sec'))

    # filter out packets with no payload length and (or not) the ones that contains < 26 bits
    ts = ts.map(abs)[ts != 0]

    def extract_indexes_in_groups(x):
        return x.index.tolist()

    def time_delta_to_float(td):
        if len(td) == 0:
            return None
        return float(str(td[0].seconds) +"." + str(td[0].microseconds))

    # Compute the moving sum of window_size seconds head in data PayloadLengt
    def rolling_forward(ts, window_size):
        def forward_window(x, ts, window_size):
            return ts[x['index'] : x['index'] + pd.Timedelta(seconds=window_size)].sum()
        roll_forward = ts.reset_index().apply(lambda x: forward_window(x, ts, window_size), axis=1).values
        return pd.Series(roll_forward, ts.index)

    stw = rolling_forward(ts, window_size)
    stw = stw[stw > minimum_window_payload]  # filter out minimum minimum_window_payload Bytes payload (banned app)
    stw = stw.resample(str(min_space_between_critical_points)+'s').apply(extract_indexes_in_groups) # 5 seconds jump
    critical_points = stw.map(time_delta_to_float).dropna().values

    return critical_points



def find_action_end(xs_capt, ys_capt, minimum_payload_size=46 ,max_delay_between_packets=5):
    """
    return the potential end of the action that begins at indicce j
    xs_capt : time elements
    ys_capt : length elements
    minimum_payload_size : do not take length <= n
    max_delay_between_packets ; Do not take into accout

    return indices of the en
    """
    xs_no_zeros = [x  for y, x in zip(ys_capt, xs_capt) if abs(y) > minimum_payload_size]
    for i, x0 in enumerate(xs_no_zeros):
        if i + 1 == len(xs_no_zeros):
            return xs_capt[-1]  # reached the end
        x1 = xs_no_zeros[i + 1]
        inter_time = x1-x0
        if inter_time > max_delay_between_packets:
            return x0
    return xs_capt[-1]

def find_x_indices(xs_capt, j, xs_end):
    "Return indice associated to the time in the serie that begin at indice j"
    for i, x in enumerate(xs_capt[j:]):
        if xs_end <= x:
            return i + j
    return len(xs_capt) - 1


def predict(time_serie, window_size=20, min_space_between_critical_points=5,
            minimum_window_payload=200,
            minimum_payload_size=46, max_delay_between_packets=5):

    """
    Make Prediction on a longrun capture
    Args:
        time_series : dict['xs', 'ys'] -> [x,],[y,]. Dictonary having 2 entries: xs for the time and ys of packet length
        window_size : int. Size in seconds of the Sliding Window.
        min_space_between_critical_points : int. Minimum spacing between two critical points.

    """

    critical_points = find_critical_point(time_serie,  window_size, min_space_between_critical_points,
                        minimum_window_payload)

    cap_predict = []  # tuple list of
    critical_points_i = 0
    xs_end = -1
    xs_capt = time_serie["xs"]
    ys_capt = time_serie["ys"]



    for i, _ in enumerate(xs_capt):


        current_xs = xs_capt[i]
        critical_point = critical_points[critical_points_i]
        if current_xs > critical_point and current_xs > xs_end:

            j = i-1   # take previous one since we are one step further

            xs_start = xs_capt[i]
            xs_end = find_action_end(xs_capt[i:], ys_capt[i:], minimum_payload_size, max_delay_between_packets)
            end_indice = find_x_indices(xs_capt, j, xs_end)
            xy = dict()
            xy["xs"] = xs_capt[j:end_indice+1]
            xy["ys"] = ys_capt[j:end_indice+1]
            features_dict = extract_features(xy)
            features = list(features_dict.values())
            y = clf.predict_proba(np.array(features).reshape(1,-1))

            cap_predict.append((xs_start, xs_end, y[0]))

            while critical_points[critical_points_i] < xs_end:
                critical_points_i +=1
                if critical_points_i == len(critical_points):
                    break




        if critical_points_i == len(critical_points) or xs_end == xs_capt[-1]:
            break
    return cap_predict

def predict_all(captures, window_size=20, min_space_between_critical_points=1,
                minimum_window_payload=200,
                minimum_payload_size=46, max_delay_between_packets=5, dprint=False,
               ):

    predicted = dict()
    for capture in captures:
        if dprint:
            print(capture)
        predicted[capture] = predict(captures[capture], window_size, min_space_between_critical_points,
                                     minimum_window_payload,
                                     minimum_payload_size, max_delay_between_packets)
    return predicted


def plot_f1_prec_rec(predicted, content_ground_truth, clf):
    linspace = np.arange(0, 1, 0.025)
    ps, rs, f1s = [], [], []

    pp.new_table()
    for sensitivity in linspace:

        all_correct_pred_dict, all_wrong_pred_dict, all_missed_gt_dict = eval_all(predicted,
                                                                                  content_ground_truth,
                                                                                  clf,
                                                                                  method='threshold_majority_voting', # "majority_voting",
                                                                                  threshold=sensitivity,
                                                                                  match_method='first',
                                                                                  target_apps_only=False,
                                                                                  top_n=1, output_dict=True,
                                                                                  skipping_force_stop=True)
        p, r, f1 = plot_prec_recall_by_slot(all_correct_pred_dict,
                                            all_wrong_pred_dict,
                                            all_missed_gt_dict,
                                            by="all", savefig=None)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

        print(all_correct_pred_dict)
        print(all_wrong_pred_dict)
        print(all_missed_gt_dict)
        print()
        
        pp.table_push([sensitivity, len(all_correct_pred_dict), len(all_wrong_pred_dict), len(all_missed_gt_dict)])

    pp.table_print()
    
    import lib.plot_builder as plot_builder
    plot_builder.save_longrun_precision_recall_f1_plot('longrun_p_r_f1_threshold', linspace, rs, ps, f1s, folder=DATASET_DIR)

    return ps, rs, f1s


## ================================================================================================


def relabel_filter_by_length(events, minimum_payload=200, ratio_app_not_satisfing_minimum_payload_length=0.25, do_not_filter=set()):
    out = dict()

    def add(device, app, action, samples):
        if not device in out:
            out[device] = dict()
        if not app in out[device]:
            out[device][app] = dict()
        if not action in out[device][app]:
            out[device][app][action] = []

        out[device][app][action].extend(samples)

    relabeled = []
    relabeled_close = []

    for device in events:
        for app in events[device]:
            for action in events[device][app]:

                below_minimum_payload = 0
                for sample in events[device][app][action]:
                    payload_length = sum([abs(s) for s in sample["ys"]])
                    if payload_length < minimum_payload:
                        below_minimum_payload += 1

                ratio_below = below_minimum_payload / len(events[device][app][action])

                always_keep = (app+"_"+action in do_not_filter)
                if not always_keep and action == "Close":
                    relabeled_close.append( app + "_" + action)
                    add(device, "NoApp", "NoAction", events[device][app][action])
                elif not always_keep and ratio_below > ratio_app_not_satisfing_minimum_payload_length:
                    relabeled.append( app + "_" + action)
                    add(device, "NoApp", "NoAction", events[device][app][action])
                else:
                    add(device, app, action, events[device][app][action])

    if len(relabeled_close) > 0:
        print("[filter-close] relabeling", len(relabeled_close), "actions as noise:", ",".join(relabeled_close))
    if len(relabeled) > 0:
        print("[filter<200] relabeling", len(relabeled), "actions as noise:", ",".join(relabeled))
    return out


def extract_features(xy, unique_from=46, unique_to=1006, unique_granularity=1, unique_deltas=[1005, 46, 0]):
    # dataset is {'xs': [packet1, packet2,...], 'ys': [packet1, packet2,...]} where x is time and y is size

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
    return f

# Anonymous used a naming convention different than the main dataset, on which we train
def rename_app_action(app_action):
    try:
        app, action = app_action.split('_')
    except:
        print("Can't split", app_action)
        sys.exit(1)

    if action=="open":
        action="Open"
    if action=="force-stop":
        action="Close"

    translations_app = dict()
    translations_app['Mobilis'] = 'Mobills'
    translations_app['UARecord'] = 'MapMyFitness'

    if app in translations_app:
        app = translations_app[app]

    translations_action = dict()
    translations_action['researchRecipy'] = 'SearchRecipe'
    translations_action['deterministicBrowse'] = 'Browse'
    translations_action['addWater'] = 'AddWater'
    translations_action['addCal'] = 'AddCalorie'
    translations_action['addProt'] = 'AddProteins'
    translations_action['addFat'] = 'AddFat'
    translations_action['addFood'] = 'AddFood'
    translations_action['addGlucose'] = 'AddGlucose'
    translations_action['addCarbs'] = 'AddCarbs'
    translations_action['addInsulin'] = 'AddInsulin'
    translations_action['browseMap'] = 'BrowseMap'
    translations_action['nightlife'] = 'NightLife'
    translations_action['fun'] = 'Leisure'
    translations_action['food'] = 'Restaurants'
    translations_action['coffee'] = 'Coffees'
    translations_action['run'] = 'Running'
    translations_action['shopping'] = 'Shopping'

    if action in translations_action:
        action = translations_action[action]

    if action == "Close":
        return NOISE_LABEL

    return app+"_"+action

def overlaps(a, b):
    return not (b[0] > a[1] or a[0] > b[1])

def overlap_length(a,b):
    if not overlaps(a, b):
        return 0
    return  min(a[1], b[1]) - max(a[0], b[0])

def build_features_labels_dataset(events, defense=None):
    data = []
    labels = []
    feature_names = []
    # shape: data is a [[Features], [Features], ...]
    #cost_aggregator = CostAggregate()
    cost_aggregator = None

    for device in events:
        for app in events[device]:
            for action in events[device][app]:
                label = app + "_" + action
                if action == "Close":
                    label = NOISE_LABEL # merge all this as a noise class
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
                    if feature_names is None:
                        feature_names = list(features_dict.keys())

                    data.append(features)
                    labels.append(label)

    return [data, labels, feature_names, cost_aggregator]

def extract_filename(f):
    f = f.replace('.csv', '').replace('.log', '')
    if '/' in f:
        f = f[f.rfind('/')+1:]
    return f

def zip_records_and_ground_truth(records, ground_truth, folder='data/'):
    records = set([extract_filename(f) for f in records])
    ground_truth = set([extract_filename(f) for f in ground_truth])

    paired_files = records.intersection(ground_truth)
    missing_files = sorted(list(records.union(ground_truth) - paired_files))

    for mf in missing_files:
        if mf in records:
            print("WARNING: {} - {} companion missing".format(mf, ".log"))
        if mf in ground_truth:
            print("WARNING: {} - {} companion missing".format(mf, ".csv"))

    paired_files = sorted(list(paired_files))
    return [[folder + f + ".csv", folder + GROUND_TRUTH_PATH_EXTENTION + f + ".log"] for f in paired_files]

def packets_to_timesize_tuples(packets, wprint=True):
    global master
    xy = dict(xs=[], ys=[])
    packets_ids = list(packets.keys())
    packets_ids.sort(reverse=False)
    # Ensure that the direction stays the same even if HuaweiWatch2 becomes master
    for packet_id in packets_ids:
        for layer in packets[packet_id]:
            master = extract_master(layer["Communication"])
            if master in POSSIBLE_MASTERS:
                direction = 1
            else:
                print("WARNING master not in Possible masters: '" + master + "'")
                direction = -1

            if not "master" in layer['Transmitter'].lower():
                direction *= -1
            xy['xs'].append(float(layer['Time']))
            xy['ys'].append(direction * int(layer['PayloadLength']))
    return xy


def dataset_file_to_xy_features(source_file):
    """
    Returns [(time,size)] for a given .csv source_file.
    Assumes packet_store has been filled (with parse_dataset) (not necessary if data is cached).
    """
    f = source_file+".xy"
    packet_store = build_datasets.packet_store

    xy_features = build_datasets.load_cache(f)
    if xy_features == None:
        if not source_file in packet_store or packet_store[source_file] == None:
            print("Cache miss, loading and cleaning file", source_file)

            build_datasets.parse_csv(source_file)
            build_datasets.packet_store_cleanup(packet_store[source_file])

            build_datasets.cache(source_file, packet_store[source_file])
        xy_features = packets_to_timesize_tuples(packet_store[source_file])
        build_datasets.cache(f, xy_features)
    return xy_features


def parse_ground_truth_file(f):
    out = []

    with open(f, "r") as file:
        csv_reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        next(csv_reader)

        for i, line in enumerate(csv_reader):
            if "finished" in line[1] or "ERROR" in line[1]:
                continue

            t = float(line[0])
            app_action = rename_app_action(line[1])

            t_bounds = [t - LOWER, t, t + UPPER]
            if app_action == "PlayStore_Browse":
                # Because Appstore apps takes longer time to open, we shift its bound
                t_bounds = [t - LOWER + 12, t, t + UPPER + 12]

            if app_action =="NoApp_NoAction" or app_action in DISCARDED_ACTION:
                continue

            out.append([t_bounds, app_action])

    return out


def action(s, target="Open", in_range=None):
    name = s.replace(DATASET_DIR, '').replace('.csv', '')
    parts = name.split('_')

    range_check_ok=True
    if in_range is not None:
        i = int(parts[5])
        low, high = in_range
        range_check_ok = (low <= i and i < high)

    return parts[0] == "HuaweiWatch2" and parts[2] == target and range_check_ok and parts[1] + "_" + parts[2] not in DISCARDED_ACTION

sources = []
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_Endomondo_BrowseMap_*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_Endomondo_Running_*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_AppInTheAir_Open_Classic_enc_9*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_DiabetesM_Add*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_FoursquareCityGuide*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_HealthyRecipes_SearchRecipe*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_Lifesum_Add*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_Qardio*.csv")
sources += glob.glob(DATASET_DIR+"HuaweiWatch2_PlayStore_Browse*.csv")
sources += [s for s in glob.glob(DATASET_DIR+"HuaweiWatch2*.csv") if action(s, "Open", [6000, 7000])]
sources += [s for s in glob.glob(DATASET_DIR+"HuaweiWatch2*.csv") if action(s, "Close")]
sources = list(set(sources))

# parse longrun folders
data_and_groundtruth_files = dict()
for d in DATASET_LONGRUN_HUMAN_PATTERNS_DIRS:
    slots_recorded = sorted(glob.glob(d + '*.csv', recursive=True))
    slots_groundtruth = sorted(glob.glob(d + GROUND_TRUTH_PATH_EXTENTION + '*.log', recursive=True))
    data_and_groundtruth_files[d] = zip_records_and_ground_truth(slots_recorded, slots_groundtruth, folder=d)



# parse ground truths
ground_truths = dict()
ground_truth_actions = set()
for d in data_and_groundtruth_files:
    for _, ground_truth in data_and_groundtruth_files[d]:
        f = extract_filename(ground_truth)
        ground_truths[f] = parse_ground_truth_file(ground_truth)
        for x in ground_truths[f]:
            ground_truth_actions.add(x[1])

ground_truth_actions = sorted(list(set(ground_truth_actions)))

# train a model from the non-longrun dataset
model = build_datasets.load_cache(CACHE_FILE)
if REBUILD or model is None:

    build_datasets.find_common_columns(sources)
    events, counts = build_datasets.cut_all_datasets_in_events(sources, folder=DATASET_DIR)

    counts = {}
    for s in sources:
        f = extract_filename(s)
        device, app, action, ble, enc, i = f.split('_')
        l = device+"_"+app+"_"+action
        if not l in counts:
            counts[l] = 0
        counts[l] += 1

    events = relabel_filter_by_length(events,
                minimum_payload=200,
                ratio_app_not_satisfing_minimum_payload_length=0.25,
                do_not_filter=set(ground_truth_actions))

    events = build_datasets.equilibrate_events_across_action(events)

    for a in events:
        for b in events[a]:
            for c in events[a][b]:
                print(a,b,c, len(events[a][b][c]))


    X, y, feature_names, _ = build_features_labels_dataset(events)
    all_trained_action = set(np.unique(np.array(y)).tolist())

    non_trained_actions = set(ground_truth_actions).difference(all_trained_action)
    if len(non_trained_actions) > 0:
        print("[warning] Could not train on", non_trained_actions)

    clf=RandomForestClassifier(n_estimators=100, random_state=None)
    clf.fit(X, y)

    model = [X, y, feature_names, clf, all_trained_action]
    build_datasets.cache(CACHE_FILE, model)

else:
    print("Loaded model from cache")
    [X,y,feature_names, clf, all_trained_action] = model
    Xtmp, ytmp = X, y


print("Classifier has {} classes: {}".format(len(all_trained_action), sorted(all_trained_action)))

# parse data from cache
data_capture = build_datasets.load_cache(CACHE_FILE2)
if True or REBUILD or data_capture is None:
    data_capture = dict()
    for d in data_and_groundtruth_files:
        print("Importing longrun packet traces in :" + d)

        recorded_traces = dict()
        data_files = [f[0] for f in data_and_groundtruth_files[d]]
        build_datasets.find_common_columns(data_files)

        for recorded_trace, _ in data_and_groundtruth_files[d]:
            filename = extract_filename(recorded_trace)
            time_serie = dataset_file_to_xy_features(recorded_trace)
            data_capture[filename] = time_serie

    build_datasets.cache(CACHE_FILE2, data_capture)

else:
    print("Loaded", len(data_capture.keys()), "longrun packet traces from cache")

print("Predicting the longrun capture")
predicted = predict_all(data_capture, max_delay_between_packets=10, window_size=15, dprint=False, min_space_between_critical_points=1)

print("Evaluate and find the best F1 score")

### Evaluate and find the best F1 score and its
ps, rs, f1s = plot_f1_prec_rec(predicted, ground_truths, clf)
