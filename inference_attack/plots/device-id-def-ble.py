
import sys
sys.path.append("..")
from lib.plot_builder import deserialize, nice_feature_names, plot_confusion_matrix, write_latex_table_precision_recall_f1
import matplotlib.pyplot as plt

from lib.defenses import CostAggregate
from lib.plot_builder import write_latex_table_defense
import numpy as np

defense_results2 = deserialize("device-id-def-ble.json")
defense_results = dict()

for defense in defense_results2:
    defense_results[defense] = dict(scores=defense_results2[defense]['scores'],
    costs=CostAggregate().from_serialized_array(defense_results2[defense]['costs']))


write_latex_table_defense("device-id-def-ble.tex", defense_results, labelFormat="\defense")

print("device-id-def-ble.png/eps")
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
plt.savefig("device-id-def-ble.png", format='png')
plt.savefig("device-id-def-ble.png".replace('.png', '.eps'), format='eps')

plt.clf()

plt.bar(names, delays)
plt.xlabel('Defense')
plt.ylabel('Mean delay added per packet [s]')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("device-id-def-ble.png".replace('.png', '-delay.png'), format='png')
plt.savefig("device-id-def-ble.png".replace('.png', '-delay.png').replace('.png', '.eps'), format='eps')

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
plt.savefig("device-id-def-ble.png".replace('.png', '-cost.png'), format='png')
plt.savefig("device-id-def-ble.png".replace('.png', '-cost.png').replace('.png', '.eps'), format='eps')
