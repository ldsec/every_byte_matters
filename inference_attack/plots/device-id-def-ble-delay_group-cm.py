
import sys
sys.path.append("..")
from lib.plot_builder import *
import matplotlib.pyplot as plt

[y_test, y_pred] = deserialize("device-id-def-ble-delay_group-cm.json")
def replace(a):
    return a.replace('SamsungGalaxyWatch', 'SamsungGWatch').replace('MDR', 'Sony MDR')

y_test = [replace(y) for y in y_test]
y_pred = [replace(y) for y in y_pred]


a, b, ax = plot_confusion_matrix(y_test, y_pred, labelfontsize=CONFUSION_MATRIX_LABELS_SIZE, normalize=True, title="", nolabel=False)
ax.tick_params(axis='both', which='major', labelsize=CONFUSION_MATRIX_XTICKS_SIZE)

write_latex_table_precision_recall_f1("device-id-def-ble-delay_group-cm.tex", y_test, y_pred)
plt.tight_layout()
plt.savefig('device-id-def-ble-delay_group-cm.png', format='png')
plt.savefig('device-id-def-ble-delay_group-cm.png'.replace('.png', '.eps'), format='eps')
print("Written device-id-def-ble-delay_group-cm.png/eps")
