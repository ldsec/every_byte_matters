
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.pyplot as plt

import numpy as np
import scikitplot as skplt

features_and_percentages = deserialize("action-id-wearables-def-add_dummies-fi.json")
xs = [nice_feature_names(x[0]) for x in features_and_percentages]
ys = [y[1] for y in features_and_percentages]
yerr = [[min(y[1], y[2]) for y in features_and_percentages], [y[2] for y in features_and_percentages]]


plt.title(None)
fig, ax = plt.subplots()
ax.barh(xs, ys, xerr=yerr)
ax.set_ylabel('Feature name', fontsize=FEATURE_IMPORTANCE_LABELS_SIZE)
ax.set_xlabel('Feature importance', fontsize=FEATURE_IMPORTANCE_LABELS_SIZE)
ax.tick_params(labelsize=FEATURE_IMPORTANCE_XTICKS_SIZE)
ax.grid(axis='x')
plt.tight_layout()

plt.savefig('action-id-wearables-def-add_dummies-fi.png', format='png')
plt.savefig('action-id-wearables-def-add_dummies-fi.png'.replace('.png', '.eps'), format='eps')
print("Written action-id-wearables-def-add_dummies-fi.png/eps")
