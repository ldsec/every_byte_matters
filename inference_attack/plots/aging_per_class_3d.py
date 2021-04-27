import os
import sys
import json
import numpy as np
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1, LONGRUN_LABEL_SIZE

import matplotlib.pyplot as plt

data_parsed = None
with open('aging_per_class_3d.json', 'r') as f:
    data = []
    for line in f:
        if not line.startswith('#'):
            data.append(line.strip())
    data_parsed = json.loads(''.join(data))

f1_diff = data_parsed['f1_diff']
f1_diff_sorted = data_parsed['f1_diff_sorted']
sorted_class = data_parsed['sorted_class']


def shorten(arr):
    arr2 = []
    for x in arr:
        if x == "FoursquareCityGuide":
            arr2.append("Foursquare")
        else:
            arr2.append(x)
    return arr2

plt.title(None)

barlist = plt.bar(x = np.arange(len(f1_diff)), height = [100 * a for a in f1_diff_sorted], width=0.8, tick_label = "")
i = 0
while f1_diff_sorted[i] < 0:
    i += 1
[barlist[i].set_color('r') for i in range(i)]

plt.xlabel("Class", fontsize=LONGRUN_LABEL_SIZE)
plt.ylabel("F1 score gain [%]", fontsize=LONGRUN_LABEL_SIZE)
plt.title("")
plt.xticks(ticks=np.arange(len(f1_diff)), labels=shorten(sorted_class), fontsize=LONGRUN_LABEL_SIZE-4, rotation=90)
plt.yticks(fontsize=LONGRUN_LABEL_SIZE)
#plt.legend(fontsize=LONGRUN_LABEL_SIZE)
plt.grid(axis='y')
plt.ylim([-32, 3])
plt.tight_layout()
plt.savefig('aging_per_class_3d.png', format='png')
plt.savefig('aging_per_class_3d.eps', format='eps')
print("Written aging_per_class_3d.png/eps")