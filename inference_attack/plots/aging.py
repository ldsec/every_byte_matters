import os
import sys
import json
import numpy as np
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1, LONGRUN_LABEL_SIZE

import matplotlib.pyplot as plt

def min_val(d):
    v = 10
    out = None
    for key in d:
        if d[key] < v:
            v = d[key]
            out = (key, d[key])
    return out

def filter0(d):
    return d
    out = {}
    for key in d:
        if d[key] > 0.0:
            out[key]=d[key]
    return out

def plot_file(in_file, out_file):
    data_parsed = None
    with open(in_file, "r") as f:
        data = []
        for line in f:
            if not line.startswith('#'):
                data.append(line.strip())
        data_parsed = json.loads(''.join(data))

    train_test_same_day = dict()
    train0_test_i = dict()
    xs = []

    for row in data_parsed:
        f1scores = filter0(row['f1scores'])
        if row['test_day'] == 0 or (row['train_day'] != row['test_day']):
            train0_test_i[row['test_day']] = list(f1scores.values())


        print(row['test_day'], min_val(f1scores))
        
        xs.append(row['test_day'])

    xs = list(sorted(list(set(xs))))

    ys = []

    for x in xs:
        ys.append(train0_test_i[x])

    plt.clf()
    plt.title(None)
    plt.boxplot(ys, showfliers=False)
    plt.xlabel("Day", fontsize=LONGRUN_LABEL_SIZE)
    plt.ylabel("F1 Score", fontsize=LONGRUN_LABEL_SIZE)
    plt.title("")
    plt.xticks(np.arange(len(xs))+1, xs, fontsize=LONGRUN_LABEL_SIZE)
    plt.yticks(fontsize=LONGRUN_LABEL_SIZE)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_file+'.png', format='png')
    plt.savefig(out_file+'.eps', format='eps')
    print("Written "+out_file+".png/eps")


plot_file("aging.json", "aging")
