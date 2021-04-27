
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt

[y_test, y_pred] = deserialize("device-id-def-cla-pad-cm.json")

def replace(a):
    return a.replace('SamsungGalaxyWatch', 'SamsungGWatch').replace('MDR', 'Sony MDR')

y_test = [replace(y) for y in y_test]
y_pred = [replace(y) for y in y_pred]


a, b, ax = plot_confusion_matrix(y_test, y_pred, labelfontsize=CONFUSION_MATRIX_LABELS_SIZE, normalize=True, title="", nolabel=False)
write_latex_table_precision_recall_f1("device-id-def-cla-add_dummies-cm.tex", y_test, y_pred)
ax.tick_params(axis='both', which='major', labelsize=CONFUSION_MATRIX_XTICKS_SIZE)
plt.tight_layout()

write_latex_table_precision_recall_f1("device-id-def-cla-pad-cm.tex", y_test, y_pred)
plt.tight_layout()
plt.savefig('device-id-def-cla-pad-cm.png', format='png')
plt.savefig('device-id-def-cla-pad-cm.png'.replace('.png', '.eps'), format='eps')
print("Written device-id-def-cla-pad-cm.png/eps")
