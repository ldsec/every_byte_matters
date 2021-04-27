
import sklearn.metrics
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np
from json import JSONEncoder
import pandas as pd
import os
import json
from tabulate import tabulate
import subprocess
import re
from operator import itemgetter

PLOT_DIR = 'plots/'

# plotting constants
CONFUSION_MATRIX_XTICKS_SIZE=20
CONFUSION_MATRIX_LABELS_SIZE=20

CONFUSION_MATRIX_HIGHRES_FIGSIZE=(18,16)
CONFUSION_MATRIX_HIGHRES_XTICKS_SIZE=14
CONFUSION_MATRIX_HIGHRES_LABELS_SIZE=20

FEATURE_IMPORTANCE_XTICKS_SIZE=14
FEATURE_IMPORTANCE_LABELS_SIZE=14

LONGRUN_LABEL_SIZE = 14

def print_classifier_score(score):
    def r(i):
        return str(round(i, 2))
    print(",".join([k+": "+r(score[k]) for k in score]))

def nice_feature_names(s):
    s = s.replace('ipt', 'IPT').replace('time_deltas', 'Δtime')
    s = s.replace('max_', 'max ').replace('min_', 'min ').replace('std_', 'std dev. ').replace('mean_', 'mean ').replace('count_', 'number of ')
    s = re.sub('_([0-9]+)_([0-9]+)', ' [\\1:\\2]', s)
    s = re.sub('2$', ">46B", s)
    s = re.sub('_46$', " >46B", s)
    s = s.replace('_non_null', '≠0')
    s = s.replace('number of sizes', 'number of pkts')
    s = s.replace('number of pkts_outgoing', 'number of outgoing pkts')
    s = s.replace('number of packets_outgoing', 'number of outgoing pkts')
    s = s.replace('number of pkts_incoming', 'number of incoming pkts')
    s = s.replace('number of packets_incoming', 'number of incoming pkts')
    s = s.replace(' sizes_outgoing', ' outgoing sizes')
    s = s.replace(' sizes_incoming', ' incoming sizes')
    s = s.replace('max outgoing sizes', 'max outgoing pkt size')
    s = s.replace('mean outgoing sizes', 'mean outgoing pkt size')
    s = s.replace('min outgoing sizes', 'min outgoing pkt size')
    s = s.replace('max incoming sizes', 'max incoming pkt size')
    s = s.replace('mean incoming sizes', 'mean incoming pkt size')
    s = s.replace('min incoming sizes', 'min incoming pkt size')
    s = s.replace('number of Δtime incoming', 'number of incoming Δtime')
    s = s.replace('IPT_incoming', 'IPT of incoming pkts')
    s = s.replace('IPT_outgoing', 'IPT of outgoing pkts')
    s = s.replace(']', ']B')

    s = s.replace('std dev. incoming sizes', 'std dev. incoming packet sizes')
    s = s.replace('std dev. sizes≠0', 'std dev. packet sizes')
    s = s.replace('max sizes≠0', 'max packet size')
    s = s.replace('min sizes≠0', 'min non-empty packet size')
    s = s.replace('mean sizes≠0', 'mean packet size')
    s = s.replace('number of pkts≠0', 'number of non-empty packets')
    s = s.replace('number of Δtime', 'number of packets')
    s = s.replace('number of packets_0', 'number of empty packets')
    s = s.replace('max incoming pkt size>46B', 'max incoming packet size')
    s = s.replace('max outgoing pkt size>46B', 'max outgoing packet size')
    s = s.replace('min Δtime >46B', 'min Δtime for packets>46B')
    s = s.replace('max Δtime >46B', 'max Δtime for packets>46B')
    s = s.replace('std dev. Δtime >46B', 'std dev. Δtime for packets>46B')
    s = s.replace('min Δtime_0', 'min Δtime for empty pkts')
    s = s.replace('max Δtime_0', 'max Δtime for empty pkts')
    s = s.replace('number of pkts_216', 'number of pkts [210;219]B')
    s = s.replace('number of pkts_170', 'number of pkts [170;179]B')
    return s


def save_longrun_precision_recall_f1_plot(filename, linspace, rs, ps, f1s, folder='data/'):
    # features_and_percentages is [(feature name, mean, std), ...]

    script_filename = filename + '.py'
    data_filename = filename + '.json'
    plot_filename = filename + '.png'

    data_to_serialize = [linspace, rs, ps, f1s]
    serialize(PLOT_DIR + data_filename, data_to_serialize, get_dataset_git_commit_id(folder))

    s = getHeader() + """
import numpy as np
import scikitplot as skplt

[linspace, rs, ps, f1s] = deserialize("{0}")

plt.title(None)
plt.plot(linspace, rs, label="recall")
plt.plot(linspace, ps, label="precision")
plt.plot(linspace, f1s, label= "F1")

plt.xlabel("Decision Maker threshold", fontsize=14)
plt.ylabel("score", fontsize=14)
plt.title("")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('{1}', format='png', dpi=180)
plt.savefig('{1}'.replace('.png', '.eps'), format='eps', dpi=180)
print("Written {1}/eps")

max_ = np.argmax(f1s)
print("maximum f1 score reached at ", linspace[max_], " threshold")
print(" precision: ", ps[max_])
print(" recall: ", rs[max_])
print(" f1: ", f1s[max_])
""".format(data_filename, plot_filename)

    if os.path.isfile(PLOT_DIR + script_filename):
        print("Not overwriting", PLOT_DIR +script_filename, "but using existing one...")
    else:
        with open(PLOT_DIR +script_filename, "w") as f:
            f.write(s)

    os.system("cd "+PLOT_DIR+" && python3 "+script_filename)


def feature_importance(filename, features_and_percentages, folder='data/'):
    # features_and_percentages is [(feature name, mean, std), ...]

    script_filename = filename + '.py'
    data_filename = filename + '.json'
    plot_filename = filename + '.png'

    features_and_percentages = sorted(features_and_percentages, key=itemgetter(1))

    serialize(PLOT_DIR + data_filename, features_and_percentages, get_dataset_git_commit_id(folder))

    s = getHeader() + """
import numpy as np
import scikitplot as skplt

features_and_percentages = deserialize("{0}")
xs = [nice_feature_names(x[0]) for x in features_and_percentages]
ys = [y[1] for y in features_and_percentages]
yerr = [[min(y[1], y[2]) for y in features_and_percentages], [y[2] for y in features_and_percentages]]

plt.title(None)
fig, ax = plt.subplots()
ax.barh(xs, ys, xerr=yerr)
ax.set_ylabel('Feature name', fontsize=14)
ax.set_xlabel('Feature importance', fontsize=14)
ax.tick_params(labelsize=14)
ax.grid(axis='x')
plt.tight_layout()

plt.savefig('{1}', format='png')
plt.savefig('{1}'.replace('.png', '.eps'), format='eps')
print("Written {1}/eps")
""".format(data_filename, plot_filename)

    if os.path.isfile(PLOT_DIR + script_filename):
        print("Not overwriting", PLOT_DIR +script_filename, "but using existing one...")
    else:
        with open(PLOT_DIR +script_filename, "w") as f:
            f.write(s)

    os.system("cd "+PLOT_DIR+" && python3 "+script_filename)

def rocauc(filename, y_test, predicted_probas, title='Receiver Operating Characteristic', folder='data/'):
    script_filename = filename + '.py'
    data_filename = filename + '.json'
    plot_filename = filename + '.png'


    data = [y_test, predicted_probas]
    serialize(PLOT_DIR + data_filename, data, get_dataset_git_commit_id(folder))

    s = getHeader() + """
import numpy as np
import scikitplot as skplt

[y_test, predicted_probas] = deserialize("{0}")
ax = skplt.metrics.plot_roc(y_test, np.asarray(predicted_probas), figsize=(7,6))
ax.set_title(None)
plt.tight_layout()
plt.savefig('{1}', format='png')
plt.savefig('{1}'.replace('.png', '.eps'), format='eps')
print("Written {1}/eps")
""".format(data_filename, plot_filename)

    if os.path.isfile(PLOT_DIR + script_filename):
        print("Not overwriting", PLOT_DIR +script_filename, "but using existing one...")
    else:
        with open(PLOT_DIR +script_filename, "w") as f:
            f.write(s)

    os.system("cd "+PLOT_DIR+" && python3 "+script_filename)


def latex_table(header, table):

    header_bold = ["\\textbf{"+t+"}" for t in header]
    table2 = [header_bold]
    table2.extend(table)

    latex_table = """\\begin{tabular}{lrrrr}
"""
    rows = [" & ".join(map(str,row)) for row in table2]
    latex_table += "".join(["    " + row + " \\\\\n" for row in rows])
    latex_table += """\\end{tabular}"""

    return latex_table

def latex_table_precision_recall_f1(y_true, y_pred, classes=None, labelFormat=""):

    # Only use the labels that appear in the data
    if classes is None:
        classes = unique_labels(y_true, y_pred)

    precision,recall,fscore,support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, labels=classes)

    header = ['Label', 'Precision', 'Recall', 'F1-score']#, 'Support']
    header_bold = ["\\textbf{"+t+"}" for t in header]
    table = [header_bold]
    i = 0

    def r(i):
        return round(i, 2)

    for thisClass in classes:
        label = thisClass
        if labelFormat != "":
            label = labelFormat +"{"+label+"}"
        table.append([label, r(precision[i]), r(recall[i]), r(fscore[i])]) #, support[i]])
        i += 1

    
    precision,recall,fscore,support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, labels=classes, average='micro')
    table.append(["\\emph{Average}", r(precision), r(recall), r(fscore)])#, len(y_true)])


    from tabulate import tabulate
    print(tabulate(table[1:], headers=header))

    latex_table = """\\begin{tabular}{lrrrr}
"""
    rows = [" & ".join(map(str,row)) for row in table]
    latex_table += "".join(["    " + row + " \\\\\n" for row in rows])
    latex_table += """\\end{tabular}"""

    return latex_table


def latex_table_defense(defense_results, labelFormat="", headerRow=True):

    header = ['Defense', 'Accuracy [\%]', 'Delay/pkt [s]', 'Extra dur. [s]', 'Padding [KB]', 'Dummy [KB]', 'Overhead [\%]']
    header_bold = ["\\textbf{"+t+"}" for t in header]
    table = [header_bold]

    if not headerRow:
        #table = []
        pass
        
    i = 0
    
    def r(i, decimals=1):
        if i==0.0 or i==0:
            return "-"
        return round(i, decimals)

    for defense in defense_results:
        name = "No defense"

        if defense is not None and defense != "null":
            name = defense

        acc = 100*defense_results[defense]['scores']['accuracy']
        delay1 = 0
        delay2 = 0
        pad = 0
        dummy = 0
        overhead = 0

        stats = defense_results[defense]['costs'].stats()
        if len(stats) != 0:
            [sum_after, sum_before, sum_dummies, sum_pad, oh_ind, dur_flat, dur_ind] = stats
            dummy = sum_dummies / 1024
            pad = sum_pad / 1024
            delay1 = dur_flat
            delay2 = dur_ind

            overhead = 0
            if sum_before != 0:
                overhead = 100 * (sum_after-sum_before)/sum_before

        if labelFormat != "":
            name = labelFormat +"{"+name+"}"

        table.append([name, r(acc), r(delay2), r(delay1), r(pad), r(dummy), r(overhead)]) #, support[i]])
        i += 1

    from tabulate import tabulate
    print(tabulate(table[1:], headers=header))

    latex_table = """\\begin{tabular}{lrrrrrr}
"""
    rows = [" & ".join(map(str,row)) for row in table]
    latex_table += "".join(["    " + row + " \\\\\n" for row in rows])
    latex_table += """\\end{tabular}"""

    return latex_table

def get_dataset_git_commit_id(folder='data/'):
    return subprocess.check_output(["git", "describe", "--always"], cwd=folder).decode('utf-8').strip()

def plot_confusion_matrix(y_true, y_pred, normalize=True, title=True, classes=None, noTextBox=False, figsize=(9, 8), dpi= 80, colorbar=False, labelfontsize=12, nolabel=True):

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=classes, normalize='true')

    # Only use the labels that appear in the data
    if classes is None:
        classes = unique_labels(y_true, y_pred)

    print(classes)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(cm, interpolation='none', aspect='auto', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
    #ax.tick_params(labelsize=labelfontsize)

    plt.xlabel('Predicted label', fontsize=labelfontsize)
    plt.ylabel('True label', fontsize=labelfontsize)

    if nolabel:
        plt.xlabel('', fontsize=labelfontsize)
        plt.ylabel('', fontsize=labelfontsize)

    if colorbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        cbar.ax.tick_params(labelsize=labelfontsize)  # set your label size here


    #plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title)
    

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if not noTextBox:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        fontsize=labelfontsize,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    return cm, fig, ax


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def serialize(uri, o, version=''):
    try:
        os.remove(uri)
    except:
        pass
    with open(uri, "w") as f:
        if version != '':
            f.write('#version: '+version+'\n')
        json.dump(o, f, cls=NumpyArrayEncoder)

def deserialize(uri, version=''):
    if os.path.isfile(uri):
        with open(uri, "r") as f:
            data = []
            for line in f:
                if not line.startswith('#version:'):
                    data.append(line)
            return json.loads(''.join(data))
    return None

def getHeader():
    return """
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.pyplot as plt
"""

def write_latex_table_precision_recall_f1(filename, y_true, y_pred, classes=None, labelFormat="", folder='data/'):
    table = latex_table_precision_recall_f1(y_true, y_pred, classes, labelFormat)
    with open(filename, "w") as f:
        f.write(table)

def write_latex_table_defense(filename, defense_results, labelFormat="", headerRow=True, folder='data/'):
    table = latex_table_defense(defense_results, labelFormat, headerRow)
    with open(filename, "w") as f:
        f.write(table)

    def r(i, decimals=2):
        return round(i, decimals)
    
    for defense in defense_results:
        costs_per_label = defense_results[defense]['costs'].detailed_stats()

        header = ['Label', 'Delay/pkt [s]', 'Extra dur. [s]', 'Padding [KB]', 'Dummy [KB]', 'Overhead [\%]']
        header_bold = ["\\textbf{"+t+"}" for t in header]
        table = []

        for label in costs_per_label:
            stats = costs_per_label[label]

            delay1 = 0
            delay2 = 0
            pad = 0
            dummy = 0
            overhead = 0

            if len(stats) != 0:
                [sum_after, sum_before, sum_dummies, sum_pad, oh_ind, dur_flat, dur_ind] = stats
                dummy = sum_dummies / 1024
                pad = sum_pad / 1024
                delay1 = dur_flat
                delay2 = dur_ind

                overhead = 0
                if sum_before != 0:
                    overhead = 100 * (sum_after-sum_before)/sum_before

            table.append([label, r(delay2), r(delay1), r(pad), r(dummy), r(overhead)]) #, support[i]])

        if len(table)>0:
            print("************** DETAILED STATS [{}] **************".format(defense))
            t2 = np.array([row[1:] for row in table])
            table.append([''] * 5)

            table.append(['min', r(np.min(t2[:, 0])), r(np.min(t2[:, 1])), r(np.min(t2[:, 2])), r(np.min(t2[:, 3])), r(np.min(t2[:, 4]))])
            table.append(['med', r(np.median(t2[:, 0])), r(np.median(t2[:, 1])), r(np.median(t2[:, 2])), r(np.median(t2[:, 3])), r(np.median(t2[:, 4]))])
            table.append(['mean', r(np.mean(t2[:, 0])), r(np.mean(t2[:, 1])), r(np.mean(t2[:, 2])), r(np.mean(t2[:, 3])), r(np.mean(t2[:, 4]))]) 
            table.append(['max', r(np.max(t2[:, 0])), r(np.max(t2[:, 1])), r(np.max(t2[:, 2])), r(np.max(t2[:, 3])), r(np.max(t2[:, 4]))])
            table.append(['std', r(np.std(t2[:, 0])), r(np.std(t2[:, 1])), r(np.std(t2[:, 2])), r(np.std(t2[:, 3])), r(np.std(t2[:, 4]))]) 

            print(tabulate(table, headers=header))


def confusion_matrix(filename, y_test, y_pred, title, folder='data/'):
    table_filename = filename + '.tex'
    script_filename = filename + '.py'
    data_filename = filename + '.json'
    plot_filename = filename + '.png'

    data = [y_test, y_pred]
    serialize(PLOT_DIR + data_filename, data, get_dataset_git_commit_id(folder))

    s = getHeader() + """
[y_test, y_pred] = deserialize("{0}")
plot_confusion_matrix(y_test, y_pred, normalize=True, title="{3}")
write_latex_table_precision_recall_f1("{2}", y_test, y_pred)
plt.tight_layout()
plt.savefig('{1}', format='png')
plt.savefig('{1}'.replace('.png', '.eps'), format='eps')
print("Written {1}/eps")
""".format(data_filename, plot_filename, table_filename, title)

    if os.path.isfile(PLOT_DIR + script_filename):
        print("Not overwriting", PLOT_DIR +script_filename, "but using existing one...")
    else:
        with open(PLOT_DIR +script_filename, "w") as f:
            f.write(s)

    os.system("cd "+PLOT_DIR+" && python3 "+script_filename)

def accuracy_over_time(filename, xs, ts_classic, ts_classic_e, ts_ble, ts_ble_e, folder='data/'):

    script_filename = filename + '.py'
    data_filename = filename + '.json'
    plot_filename = filename + '.png'

    data = [xs, ts_classic, ts_classic_e, ts_ble, ts_ble_e]
    serialize(PLOT_DIR + data_filename, data, get_dataset_git_commit_id(folder))

    s = getHeader() + """
[xs, ts_classic, ts_classic_e, ts_ble, ts_ble_e] = deserialize("{0}")
plt.figure()
plt.errorbar(xs, ts_classic, ts_classic_e, label='Bluetooth Classic')
plt.errorbar(xs, ts_ble, ts_ble_e, label='Bluetooth Low Energy')
plt.xlabel('Capture duration (T_cutoff)')
plt.ylabel('Classifier accuracy')
plt.xscale('log')
#plt.title('Attacker accuracy versus Capture duration')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('{1}', format='png')
plt.savefig('{1}'.replace('.png', '.eps'), format='eps')
print("Written {1}/eps")
""".format(data_filename, plot_filename)

    if os.path.isfile(PLOT_DIR +script_filename):
        print("Not overwriting", PLOT_DIR +script_filename, "but using existing one...")
    else:
        with open(PLOT_DIR + script_filename, "w") as f:
            f.write(s)

    os.system("cd "+PLOT_DIR+" && python3 "+script_filename)


def accuracy_vs_nsamples(filename, xs, ys_cla, ys_ble, folder='data/'):

    script_filename = filename + '.py'
    data_filename = filename + '.json'
    plot_filename = filename + '.png'

    data = [xs, ys_cla, ys_ble]
    serialize(PLOT_DIR + data_filename, data, get_dataset_git_commit_id(folder))

    s = getHeader() + """
[xs, ys_cla, ys_ble] = deserialize("{0}")
plt.figure()
plt.errorbar(xs, ys_cla, label='Bluetooth Classic')
plt.errorbar(xs, ys_ble, label='Bluetooth Low Energy')
plt.xlabel('Number of features')
plt.ylabel('Classifier accuracy')
plt.xscale('log')
#plt.title('Attacker accuracy versus Number of Features')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('{1}', format='png')
plt.savefig('{1}'.replace('.png', '.eps'), format='eps')
print("Written {1}/eps")
""".format(data_filename, plot_filename)

    if os.path.isfile(PLOT_DIR +script_filename):
        print("Not overwriting", PLOT_DIR +script_filename, "but using existing one...")
    else:
        with open(PLOT_DIR + script_filename, "w") as f:
            f.write(s)

    os.system("cd "+PLOT_DIR+" && python3 "+script_filename)


def defense_plot(filename, defense_results, folder='data/'):
    table_filename = filename + '.tex'
    script_filename = filename + '.py'
    data_filename = filename + '.json'
    plot_filename = filename + '.png'

    defense_results2 = dict()
    for defense in defense_results:
        defense_results2[defense] = dict(scores=defense_results[defense]['scores'],
        costs=defense_results[defense]['costs'].to_serialized_array())

    data = defense_results2
    serialize(PLOT_DIR + data_filename, data, get_dataset_git_commit_id(folder))

    s = getHeader() + """
from lib.defenses import CostAggregate
from lib.plot_builder import write_latex_table_defense
import numpy as np

defense_results2 = deserialize("{0}")
defense_results = dict()

for defense in defense_results2:
    defense_results[defense] = dict(scores=defense_results2[defense]['scores'],
    costs=CostAggregate().from_serialized_array(defense_results2[defense]['costs']))


write_latex_table_defense("{2}", defense_results, labelFormat="\\defense")

print("{1}/eps")
sys.exit(0)

names = []
accuracies = []
costs_dummy = []
costs_pad = []
delays = []

for defense in defense_results:
    if defense is None:
        names.append("No defense")
    else:
        names.append(defense)
    accuracies.append(defense_results[defense]['scores']['accuracy'])

    stats = defense_results[defense]['costs'].stats()
    if len(stats) == 0:
        costs_dummy.append(0)
        costs_pad.append(0)
        delays.append(0)
    else:
        [sum_after, sum_before, sum_dummies, sum_pad, oh_ind, dur_flat, dur_ind] = stats
        costs_dummy.append(sum_dummies / 1024)
        costs_pad.append(sum_pad / 1024)
        delays.append(dur_ind)


plt.bar(names, accuracies)
plt.xlabel('Defense')
plt.ylabel('Classifier accuracy')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("{1}", format='png')
plt.savefig("{1}".replace('.png', '.eps'), format='eps')

plt.clf()

plt.bar(names, delays)
plt.xlabel('Defense')
plt.ylabel('Mean delay added per packet [s]')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("{1}".replace('.png', '-delay.png'), format='png')
plt.savefig("{1}".replace('.png', '-delay.png').replace('.png', '.eps'), format='eps')

plt.clf()

width = 0.3
xs1 = np.arange(len(costs_dummy)) 
xs2 = [x + width for x in xs1] 

plt.bar(xs1, costs_dummy, label="Dummy messages", width=width,align='edge')
plt.bar(xs2, costs_pad, label="Padding bytes", width=width,align='edge')
plt.xticks(xs2, names) 
plt.xlabel('Defense')
plt.ylabel('Dummy traffic and Padding, per sample [KB]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("{1}".replace('.png', '-cost.png'), format='png')
plt.savefig("{1}".replace('.png', '-cost.png').replace('.png', '.eps'), format='eps')
""".format(data_filename, plot_filename, table_filename)

    if os.path.isfile(PLOT_DIR + script_filename):
        print("Not overwriting", PLOT_DIR +script_filename, "but using existing one...")
    else:
        with open(PLOT_DIR +script_filename, "w") as f:
            f.write(s)

    os.system("cd "+PLOT_DIR+" && python3 "+script_filename)