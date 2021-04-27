
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.pyplot as plt

import numpy as np
import scikitplot as skplt

features_and_percentages = deserialize("device-id-def-cla-add_dummies-fi.json")
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

plt.savefig('device-id-def-cla-add_dummies-fi.png', format='png')
plt.savefig('device-id-def-cla-add_dummies-fi.png'.replace('.png', '.eps'), format='eps')
print("Written device-id-def-cla-add_dummies-fi.png/eps")
