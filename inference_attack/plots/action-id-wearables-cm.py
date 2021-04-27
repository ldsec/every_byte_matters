
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

[y_test, y_pred] = deserialize("action-id-wearables-cm.json")

def replace(s):
    s = s.replace('SamsungGalaxyWatch', 'GalaxyWatch')
    return s

y_test = [replace(y) for y in y_test]
y_pred = [replace(y) for y in y_pred]

cm, fig, ax = plot_confusion_matrix(y_test, y_pred, nolabel=False,labelfontsize=CONFUSION_MATRIX_LABELS_SIZE, normalize=True, title="", noTextBox=True, figsize=(9, 8), colorbar=True)
plt.tight_layout()


def app(s):
    p = s.split('_')
    return p[0]+"_"+p[1]

uniq = sorted(list(set(y_test)))
boxes = []
i = 0
while i<len(uniq):
    j = i + 1
    while j<len(uniq) and app(uniq[i]) == app(uniq[j]):
        j += 1
    if j-i > 1 and not app(uniq[i]).endswith("NoApp"):
        boxes.append([app(uniq[i]), i, j-i])
    i = j

for app, start, width in boxes:
    ax.add_patch(patches.Rectangle((start-0.5, start-0.5),width,width,linewidth=2,edgecolor='r',linestyle='--',facecolor='none'))

ax.text(16, 35, "Same application", fontsize=CONFUSION_MATRIX_LABELS_SIZE, color='r')

if True:
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()

plt.savefig('action-id-wearables-cm.png', format='png')
plt.savefig('action-id-wearables-cm.png'.replace('.png', '.eps'), format='eps')
print("Written action-id-wearables-cm.png/eps")


import sklearn.metrics
from sklearn.utils.multiclass import unique_labels
classes = unique_labels(y_test, y_pred)
precision,recall,fscore,support = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, labels=classes)

i = 0

def r(i):
    return round(i, 2)

f1s_cla = []
f1s_ble = []

for thisClass in classes:
    if "Airpods" in thisClass or "AppleWatch" in thisClass or "Fossil" in thisClass or "GalaxyWatch" in thisClass or "HuaweiWatch2" in thisClass or "MDR" in thisClass:
        f1s_cla.append(fscore[i])
    else:
        f1s_ble.append(fscore[i])

    print([thisClass, r(precision[i]), r(recall[i]), r(fscore[i])]) 
    i += 1

print("F1 score classic:", np.mean(f1s_cla), f1s_cla)
print("F1 score BLE:", np.mean(f1s_ble), f1s_ble)