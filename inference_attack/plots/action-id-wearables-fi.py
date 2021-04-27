
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt

import numpy as np
import scikitplot as skplt

features_and_percentages = deserialize("action-id-wearables-fi.json")
features_and_percentages = sorted(features_and_percentages, key=lambda x: -x[1])
features_and_percentages = features_and_percentages[:15]
features_and_percentages = sorted(features_and_percentages, key=lambda x: x[1])

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

plt.savefig('action-id-wearables-fi.png', format='png')
plt.savefig('action-id-wearables-fi.png'.replace('.png', '.eps'), format='eps')
print("Written action-id-wearables-fi.png/eps")
