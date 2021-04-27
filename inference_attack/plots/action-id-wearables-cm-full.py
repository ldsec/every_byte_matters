
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

[y_test, y_pred] = deserialize("action-id-wearables-cm.json")

def replace(s):
    s = s.replace('SamsungGalaxyWatch', 'GalaxyWatch').replace('_NoApp_', '_').replace('_', ' ').replace('SearchRecipe', 'Search').replace('SamsungHealthApp', 'Health').replace('FossilExploristHR', 'Fossil').replace('MyFitnessPalApp', 'MyFitnessPal')
    return s

y_test = [replace(y) for y in y_test]
y_pred = [replace(y) for y in y_pred]


cm, fig, ax = plot_confusion_matrix(y_test, y_pred, normalize=True, title="", noTextBox=True, figsize=CONFUSION_MATRIX_HIGHRES_FIGSIZE, colorbar=True)
#write_latex_table_precision_recall_f1("action-id-wearables-cm.tex", y_test, y_pred, labelFormat="\\app")

def app(s):
    p = s.split(' ')
    return p[0]+" "+p[1]

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

ax.text(12, 25, "Same application", fontsize=CONFUSION_MATRIX_HIGHRES_LABELS_SIZE, color='r')
ax.tick_params(axis='both', which='major', labelsize=CONFUSION_MATRIX_HIGHRES_XTICKS_SIZE)
plt.tight_layout()

plt.savefig('action-id-wearables-cm-full.png', format='png')
plt.savefig('action-id-wearables-cm-full.png'.replace('.png', '.eps'), format='eps')
print("Written action-id-wearables-cm-full.png/eps")
