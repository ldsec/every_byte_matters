import os
import sys
import json
import numpy as np
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1, LONGRUN_LABEL_SIZE

import matplotlib.pyplot as plt


def plot_file(in_file, out_file):
    data_parsed = None
    with open(in_file, "r") as f:
        data = []
        for line in f:
            if not line.startswith('#version:'):
                data.append(line)
        data_parsed = json.loads(''.join(data))

    aggregated = dict()
    aggregated_low = dict()
    aggregated_high = dict()

    for row in data_parsed:
        if not row['loss'] in aggregated:
            aggregated[row['loss']] = []
            aggregated_low[row['loss']] = []
            aggregated_high[row['loss']] = []
        aggregated[row['loss']].append(row['accuracy'])
        aggregated_low[row['loss']].append(row['accuracy_low'])
        aggregated_high[row['loss']].append(row['accuracy_high'])

    keys = list(aggregated.keys())

    xs = keys
    ys = []
    yerr = []
    ys_low = []
    yerr_low = []
    ys_high = []
    yerr_high = []

    for x in xs:
        ys.append(np.mean(aggregated[x]))
        yerr.append(np.std(aggregated[x]))
        ys_low.append(np.mean(aggregated_low[x]))
        yerr_low.append(np.std(aggregated_low[x]))
        ys_high.append(np.mean(aggregated_high[x]))
        yerr_high.append(np.std(aggregated_high[x]))

    plt.clf()
    plt.title(None)
    plt.errorbar(xs, ys_high, yerr=yerr_high, label="High-volume apps", linestyle="--")
    plt.errorbar(xs, ys, yerr=yerr, label="All apps")
    plt.errorbar(xs, ys_low, yerr=yerr_low, label="Low-volume apps", linestyle=":")

    plt.xlabel("Packet loss [%]", fontsize=LONGRUN_LABEL_SIZE)
    plt.ylabel("Classifier accuracy [%]", fontsize=LONGRUN_LABEL_SIZE)
    plt.title("")
    plt.xticks(fontsize=LONGRUN_LABEL_SIZE)
    plt.yticks(fontsize=LONGRUN_LABEL_SIZE)
    plt.legend(fontsize=LONGRUN_LABEL_SIZE)
    plt.grid()
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(out_file+'.png', format='png')
    plt.savefig(out_file+'.eps', format='eps')
    print("Written "+out_file+".png/eps")

plot_file("deep-exp-loss2.json", "deep-exp-loss")
plot_file("deep-exp-loss-multitrain.json", "deep-exp-loss-multitrain")